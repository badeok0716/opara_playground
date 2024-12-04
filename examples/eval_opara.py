import os, sys
import torch
from examples.infer_op import run_torch_model, run_torch_compile_model, run_sequence_graph, run_parallel_graph
from examples.io_op import write_pkl
def prepare_deepfm(batch_size, device):
    from examples.NCF import DeepFM
    cate_fea_nuniqs = [100*(i+1) for i in range(32)] 
    nume_fea_size = 16  
    model = DeepFM(cate_fea_nuniqs, nume_fea_size, emb_size=8, hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]).to(device).eval()

    X_sparse = torch.randint(0, 100, (batch_size, len(cate_fea_nuniqs))).to(device)  # 生成随机的类别特征输入
    X_dense = torch.rand(batch_size, nume_fea_size).to(device)  # 生成随机的数值特征输入
    inputs = (X_sparse, X_dense)

    return model, inputs

def prepare_googlenet(batch_size, device):
    import torchvision
    model = torchvision.models.googlenet().to(device).eval()

    x = torch.randint(low=0, high=256, size=(batch_size, 3, 224, 224), dtype=torch.float32).to(device)
    inputs = (x,)
    return model, inputs

def prepare_nasnet(batch_size, device):
    import pretrainedmodels
    model_name = 'nasnetalarge'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').to(device).eval()
    x = torch.randint(low=0, high=256, size=(batch_size, 3, 331, 331), dtype=torch.float32).to(device)
    inputs = (x,)
    return model, inputs

def eval_model(model, inputs, method, iterations, warm_ups, check_eq=False):
    if method == 'torch':
        _, tl = run_torch_model(model, inputs, iterations, warm_ups)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif method == 'torch_compile':
        _, tl = run_torch_compile_model(model, inputs, iterations, warm_ups)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif method == 'sequence':
        _, tl = run_sequence_graph(model, inputs, iterations, warm_ups, 0, iterations)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif method == 'opara':
        from Opara import GraphCapturer
        Opara = GraphCapturer.capturer(inputs, model)
        output, tl = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, iterations)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        # check equality
        if check_eq:
            res = output[0]
            if res.dtype == torch.float16:
                res = res.float()
            y, _ = run_torch_model(model, inputs, iterations, warm_ups)
            assert torch.allclose(y, output[0], rtol=1e-05, atol=1e-05, equal_nan=False)
    
    return tl, mem

model_preparer_dict = {
    'deepfm': prepare_deepfm,
    'googlenet': prepare_googlenet,
    'nasnet': prepare_nasnet,
}

def print_args(args, node_name):
    print("="*40)
    print(f"Node: {node_name}")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print("="*40)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Opara example')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warm_ups', type=int, default=1000, help='warm up iterations')
    parser.add_argument('--iterations', type=int, default=5000, help='iterations')
    parser.add_argument('--model', type=str, default='deepfm', choices=['deepfm', 'googlenet', 'nasnet'], help='model name')
    parser.add_argument('--method', type=str, default='opara', choices=['torch', 'torch_compile', 'sequence', 'opara'], help='method')
    args = parser.parse_args()
    
    node_name = os.environ['SLURM_NODELIST']
    device = torch.device('cuda:0')
    # log arguments
    print_args(args, node_name)

    # check exists
    pathkey = f'{node_name}_{args.batch_size}_{args.warm_ups}_{args.iterations}/'
    path2dir = f'/home/deokjae/Opara/examples/results_2.0.0/' + pathkey
    path2pkl = path2dir + f'{args.model}_{args.method}.pkl'
    os.makedirs(path2dir, exist_ok=True)
    if os.path.exists(path2pkl):
        print(f'Exists: {path2pkl}')
        sys.exit()

    # prepare model and inputs
    model_preparer = model_preparer_dict[args.model]
    model, inputs = model_preparer(args.batch_size, device)
    
    # eval inference time and memory
    tl, mem = eval_model(model, inputs, args.method, args.iterations, args.warm_ups)    
    results = {
        'time_list': tl,
        'memory': mem,
    }

    # write pkl
    write_pkl(results, path2pkl)