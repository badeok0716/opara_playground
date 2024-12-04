import torch
import numpy as np
import torch._dynamo as dynamo
import os 

cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    cache.zero_()

def run_torch_model(model, inputs, iterations, warm_ups):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]    
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    # Warmup steps
    with torch.inference_mode():
        for _ in range(warm_ups):
            model(*inputs) # don't record time
        torch.cuda.synchronize()
        for i in range(iterations):
            flush_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record()
            y = model(*inputs)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        std = np.std(times)
        output_str = ('Time of native PyTorch:', str(np.mean(times)) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
    return y, times

def run_torch_compile_model(model, inputs, iterations, warm_ups):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]    
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    model = torch.compile(model)
    # Warmup steps
    with torch.inference_mode():
        for _ in range(warm_ups):
            model(*inputs) # don't record time
        torch.cuda.synchronize()
        for i in range(iterations):
            flush_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record()
            y = model(*inputs)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        std = np.std(times)
        output_str = ('Time of native PyTorch:', str(np.mean(times)) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
    return y, times

def run_sequence_graph(symbolic_traced, inputs, iterations, warm_ups, start_index, end_index):
    with torch.inference_mode():
        for i in range(warm_ups):
            symbolic_traced(*inputs)

    with torch.inference_mode():
        g1 = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g1):
            out_torch_graph_without_stream = symbolic_traced(*inputs)

        time_list = []
        torch.cuda.synchronize()
        for i in range(iterations):
            start = torch.cuda.Event(enable_timing=True)  # the times
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            g1.replay()
            end.record()
            end.synchronize()
            tim = start.elapsed_time(end)
            time_list.append(tim)
        average_time = np.mean(time_list[start_index: end_index])
        std = np.std(time_list[start_index: end_index])
        output_str = ('Time of sequential CUDA Graph:', str(average_time) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
        return out_torch_graph_without_stream, time_list[start_index: end_index]


def run_parallel_graph(Opara, inputs, iterations, warm_ups, start_index, end_index):
    time_list = []

    for i in range(iterations):
        flush_cache()
        torch.cuda._sleep(1_000_000)
        start = torch.cuda.Event(enable_timing=True)  # the times
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = Opara(*inputs)
        end.record()
        end.synchronize()
        tim = start.elapsed_time(end)
        time_list.append(tim)
    average_time = np.mean(time_list[start_index: end_index])
    std = np.std(time_list[start_index: end_index])
    output_str = ('Time of Opara:', str(average_time) + ' ms', "std: " + str(std) )
    print('{:<30} {:<20} {:<20}'.format(*output_str))
    return output, time_list[start_index: end_index]
