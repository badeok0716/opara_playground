from functools import reduce
from operator import mul
from Opara import ModelProfiler

from collections import deque
import os

def launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    def is_mem_access_intensive(node):
        memory_intensive_op = ['add', 'cast', 'ceil', 'clip', 'concat', 'exp', 'floor', 'log', 
                            'gelu', 'neg', 'pow', 'reciprocal', 'relu', 'sigmoid', 'slice', 'relu'
                            'sqrt', 'sub', 'tanh', 'transpose', 'unsqueeze', 'view', 'avg_pool',
                            'reshape', 'max_pool', 'adaptive_avg_pool', 'adaptive_max_pool', 'premute',
                            'flatten', 'dropout', 'batch_norm', 'layer_norm', 'instance_norm', 'contiguous', 'ones', 'to']
        for op in memory_intensive_op:
            if op in node.name:
                # print("memory_intensive_op:", node.name)
                return True
        return False
    
    def pop_from_queue(q):
        ret_node_name = q[0]

        min_metric = 2
        for node_name in q:
            if len(nodes[node_name].info) > 0:
                achieved_occupancy = nodes[node_name].info[0]["args"]["est. achieved occupancy %"]
                blocksPerSM = nodes[node_name].info[0]["args"]["blocks per SM"]
                shared_memory = nodes[node_name].info[0]["args"]["shared memory"] / sharedMemPerBlock
                thread_num = reduce(mul, nodes[node_name].info[0]["args"]["block"]) / maxThreadsPerBlock
                registers_num = thread_num * nodes[node_name].info[0]["args"]["registers per thread"] / regsPerBlock
                request = [shared_memory, thread_num, registers_num]
                # metric = max(shared_memory, max(thread_num, registers_num))
                # metric = thread_num
                # metric = achieved_occupancy
                metric = shared_memory
                # if metric == min_metric:
                #     print("same metric:", node_name, ret_node_name)
                if metric < min_metric:
                    min_metric = metric
                    ret_node_name = node_name
        q.remove(ret_node_name)

        return ret_node_name
    
    memory_queue = deque()
    not_memory_queue = deque()
    for node_name, degree in in_degree.items():
        if degree == 0:
            if is_mem_access_intensive(nodes.get(node_name)):
                memory_queue.append(node_name)
            else:
                not_memory_queue.append(node_name)
    flag = True
    while memory_queue or not_memory_queue:

        flag = not flag
        if memory_queue and not_memory_queue:
            if flag:
                q = memory_queue
            else:
                q = not_memory_queue
        else:
            if memory_queue:
                q = memory_queue
            else:
                q = not_memory_queue
        cur_node_name = pop_from_queue(q)
        result.append(cur_node_name)

        for succ_node in nodes.get(cur_node_name).users:
            in_degree[succ_node.name] -= 1
            if in_degree[succ_node.name] == 0:
                if is_mem_access_intensive(nodes.get(succ_node.name)):
                    memory_queue.append(succ_node.name)
                else:
                    not_memory_queue.append(succ_node.name)
    return result


import json
import os

def get_resource_from_json(path):
    with open(path) as f:
        data = json.load(f)

    step_num = 0
    for event in data["traceEvents"]:
        if "torch/fx/interpreter.py(97): run" in event["name"] and "run_node" not in event["name"]:
            step_num += 1
    # print("step_num", step_num)
   
    # 获取run_node事件、kernel_launch事件、kernel事件
    run_node_events = []
    kernel_launch_events = []
    kernel_events = []
    for event in data["traceEvents"]:
        if "run_node" in event["name"]:
            run_node_events.append(event)

        if event["name"] == "cudaLaunchKernel":
            kernel_launch_events.append(event)

        if event.get("cat", "None") == "kernel":
            kernel_events.append(event)


    # 计算获取一个step中的run_node事件、kernel_launch事件、kernel事件
    one_step_range_of_node = len(run_node_events) // step_num
    one_step_range_of_kernel_launch = len(kernel_launch_events) // step_num
    one_step_range_of_kernel = len(kernel_events) // step_num
    start = step_num - 1
    end = step_num
    run_node_events = run_node_events[start*one_step_range_of_node:end*one_step_range_of_node]
    kernel_launch_events = kernel_launch_events[start*one_step_range_of_kernel_launch:end*one_step_range_of_kernel_launch]
    kernel_events = kernel_events[start*one_step_range_of_kernel:end*one_step_range_of_kernel]


    # 根据时间轴范围获取由node事件触发的kernel_launch事件
    node2kernels = []
    kernel_num = 0
    for i, node_event in enumerate(run_node_events):
        node2kernels.append([])
        for j, kernel_launch_event in enumerate(kernel_launch_events):
            if node_event["ts"] <= kernel_launch_event["ts"] and node_event["ts"] + node_event["dur"] >= kernel_launch_event["ts"]:
                node2kernels[i].append(kernel_events[j])
                kernel_num += 1

    max_block_nums = []
    sum_time = 0
    for i, kernel_events in enumerate(node2kernels):
        
        max_block_size = 4096
        for kernel_event in kernel_events:
            sum_time += kernel_event["dur"]
            cur_block_size = kernel_event["args"]["block"][0] * kernel_event["args"]["block"][1] * kernel_event["args"]["block"][2]
            max_block_size = min(max_block_size, cur_block_size)
        if max_block_size == 4096:
            max_block_size = 0
        max_block_nums.append(max_block_size)

    est_achieved_occupancy = 0
    for i, kernel_events in enumerate(node2kernels):
        for kernel_event in kernel_events:
            dur = kernel_event["dur"]
            est_achieved_occupancy += kernel_event["args"]["est. achieved occupancy %"] * dur
    est_achieved_occupancy = est_achieved_occupancy / sum_time
    sharedMemPerBlock = data['deviceProperties'][0]['sharedMemPerBlock']
    regsPerBlock = data['deviceProperties'][0]['regsPerBlock']
    maxThreadsPerBlock = data['deviceProperties'][0]['maxThreadsPerBlock']


    return node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock


def get_topo(fx_nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    nodes = {node.name: node for node in fx_nodes}
    in_degree = {node.name: 0 for node in nodes.values()}
    for node in nodes.values():
        for input_node in node.all_input_nodes:
            in_degree[node.name] += 1
    visited = set()
    result = []
    result = launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)

    return result, nodes

def recompile(model_class_name, graph_module, inputs):
    
    path = os.path.abspath(os.path.dirname(__file__))
    for i in inputs:
        model_class_name += "_" + str(i.shape)
    cur_slurm_node = os.environ['SLURM_NODELIST']
    os.makedirs(path + "/profile_result/2.0.0/", exist_ok=True)
    path += "/profile_result/2.0.0/" + model_class_name + "_" + cur_slurm_node + ".pt.trace.json"
    if os.path.exists(path) is False:
        ModelProfiler.profile(graph_module, inputs, path)
    node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock = get_resource_from_json(path)

    for i, node in enumerate(graph_module.graph.nodes):
        if not hasattr(node, 'info'):
            setattr(node, 'info', node2kernels[i])

    result, torch_nodes = get_topo(graph_module.graph.nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)

    size = len(result)
    for i in range(size - 1):
        torch_nodes[result[i]].append(torch_nodes[result[i+1]])
    graph_module.graph.lint()
    graph_module.recompile()
