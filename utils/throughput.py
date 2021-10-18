import torch
import torch.optim
import torch.utils.data
import torch


def throughput(model, img_size, args):
        
    inputs = torch.randn(args.batch_size, 3, img_size, img_size).cuda(args.gpu)
    repetitions=100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    Throughput =   (repetitions*128)/total_time
    
    
    return Throughput
            