import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess

from MobileNetV2 import MobileNetV2

# benchmark settings
parser = argparse.ArgumentParser(description='PyTorch MobileNetV2 Benchmark')
parser.add_argument('--batch_size', type=int, default=96,
                    help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--profile', action='store_true', default=False,
                    help='enable profiler')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

input_sizes = [3, 224, 224]
steps = 5 # nb of steps in loop to average perf
nDryRuns = 1


if args.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    kernel = 'cudnn'
    p = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv', 
                                shell=True)
    device_name = str(p).split('\\n')[1]
else:
    kernel = 'nn'
    p = subprocess.check_output('cat /proc/cpuinfo | grep name | head -n 1',
                                shell = True)
    device_name = str(p).split(':')[1][:-3]

print('Running on device: %s' % (device_name))


def main():
    t = time.time()

    c, h, w = input_sizes[0], input_sizes[1], input_sizes[2]
    data_ = torch.randn(args.batch_size, c, h, w)
    target_ = torch.arange(1, args.batch_size + 1).long()        
    net = MobileNetV2()
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum =0.9, weight_decay=0.00004)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        data_, target_ = data_.cuda(), target_.cuda()
        net.cuda()
        criterion = criterion.cuda()
        
    net.eval()
        
    print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' % (
            'MobileNetV2', kernel, args.batch_size, c, h, w))
    data, target = Variable(data_), Variable(target_)
        
    for i in range(nDryRuns):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update

    time_fwd, time_bwd, time_upt = 0, 0, 0
        
    for i in range(steps):
        optimizer.zero_grad()   # zero the gradient buffers
        t1 = time.time()
        output = net(data)
        t2 = time.time()
        loss = criterion(output, target)
        loss.backward()
        t3 = time.time()
        optimizer.step()    # Does the update
        t4 = time.time()
        time_fwd = time_fwd + (t2 - t1)
        time_bwd = time_bwd + (t3 - t2)
        time_upt = time_upt + (t4 - t3)
      
    time_fwd_avg = time_fwd / steps * 1000
    time_bwd_avg = time_bwd / steps * 1000
    time_upt_avg = time_upt / steps * 1000
        
    # update not included!
    time_total = time_fwd_avg + time_bwd_avg
    
    print("%-30s %10s %10.2f %10.2f" % (kernel, ':forward:', time_fwd_avg, args.batch_size*1000/time_fwd_avg))
    print("%-30s %10s %10.2f" % (kernel, ':backward:', time_bwd_avg))
    print("%-30s %10s %10.2f" % (kernel, ':update:', time_upt_avg))
    print("%-30s %10s %10.2f %10.2f" % (kernel, ':total:', time_total, args.batch_size*1000/time_total))
        

if __name__ == '__main__':
    if args.profile:
        with torch.autograd.profiler.profile() as prof:
            main()
        profile_log = device_name.replace(' ', '_')
        f = open('%s-profile.txt' % (profile_log), 'w')
        f.write(prof.__str__())
    else:
        main()
