import argparse
import torch.nn as nn
import torch
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
import numpy

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.D = 3
        self.channels = 64
        self.msfa_size = [4, 4]
        self.iters = args.iters

        self.conv1 = nn.Conv2d(args.num_bands+1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, args.num_bands, 3, 1, 1)

        self.relu = nn.ReLU()

        self.proj_net = Proj_Net(args)
        self.degrade_r = Degrade_R(args)
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, pan), 1)
        fused = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))) + pan.repeat(1, mosaic_interpolate.shape[1], 1, 1)
        
        for _ in range(args.iters):
            pan_from_hrms = self.degrade_r(fused)
            mosaic_from_hrms = degrade_dm(fused, msfa_kernel)
            delta_mosaic = mosaic - mosaic_from_hrms
            delta_pan = pan - pan_from_hrms
            delta_hrhsi = self.proj_net(delta_mosaic, delta_pan)
            fused = fused + delta_hrhsi
        return fused

def degrade_dm(hrms, msfa_kernel):
    x = torch.nn.functional.conv2d(hrms, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=hrms.shape[1])
    x = torch.nn.functional.pixel_shuffle(x, 4)

    return x

class Degrade_R(nn.Module):
    def __init__(self, args):
        super(Degrade_R, self).__init__()

        self.spec_res = nn.Conv2d(args.num_bands, 1, 3, 1, 1)
    
    def forward(self, hrms):
        y = self.spec_res(hrms)
        return y

class Proj_Net(nn.Module):
    def __init__(self, args):
        super(Proj_Net, self).__init__()
        self.D = 3
        self.channels = 64
        self.msfa_size = [4, 4]

        self.conv1 = nn.Conv2d(args.num_bands+1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, args.num_bands, 3, 1, 1)

        self.relu = nn.ReLU()
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        return y

def get_param_and_flops_of_each_layer(net, net_name, params, flops, memory):
    keys = list(net._modules.keys())
    if keys == []:
        if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ReLU) \
        or isinstance(net, torch.nn.PReLU) or isinstance(net, torch.nn.ELU) \
        or isinstance(net, torch.nn.LeakyReLU) or isinstance(net, torch.nn.ReLU6) \
        or isinstance(net, torch.nn.Linear) or isinstance(net, torch.nn.MaxPool2d) \
        or isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.BatchNorm2d) \
        or isinstance(net, torch.nn.Upsample) or isinstance(net, torch.nn.ConvTranspose2d):
            params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            param_this_layer = str(params_num)
            if params_num // 10 ** 6 > 0:
                param_this_layer = str(round(params_num / 10 ** 6, 2)) + 'M'
            elif params_num // 10 ** 3:
                param_this_layer = str(round(params_num / 10 ** 3, 2)) + 'k'
            kernel_size = str(tuple(net.weight.shape[-2:])) if hasattr(net, 'weight') else ''
            stride = str(net.stride) if hasattr(net, 'stride') else ''
            input_shape = str(tuple(net.input_shape)) if hasattr(net, 'input_shape') else ''
            output_shape = str(tuple(net.output_shape)) if hasattr(net, 'output_shape') else ''
            module_type = str(type(net)).split('.')[-1][:-2]
            mem = str(net.__mem__ // 1e6) + "MB" if hasattr(net, '__mem__') else ''
            if "f_and_g" in net_name:
                print('{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format(net_name, module_type, input_shape,
                                                                                    output_shape, kernel_size, stride,
                                                                                    param_this_layer,
                                                                                    flops_to_string(2*net.__flops__), mem))
            else:
                print('{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format(net_name, module_type, input_shape,
                                                                                    output_shape, kernel_size, stride,
                                                                                    param_this_layer,
                                                                                    flops_to_string(net.__flops__), mem))
            params.append(params_num)
            flops.append(net.__flops__)
            memory.append(net.__mem__)
    else:
        for key in keys:
            get_param_and_flops_of_each_layer(net._modules[key],
                                              net_name=net_name + '.' + key,
                                              params=params,
                                              flops=flops,
                                              memory=memory)

if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.num_bands = 16
    args.iters = 2
    net = Network(args).cuda()
    batch_size = 1

    pan, mosaic = torch.FloatTensor(batch_size, 1, 1024, 1024).cuda(), torch.FloatTensor(batch_size, 1, 512, 512).cuda()
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).cuda()
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25

    model = add_flops_counting_methods(net)
    model.eval().start_flops_count()
    t = time.time()
    with torch.no_grad():
        out = model(mosaic, pan)
    params, flops, memory = [], [], []
    print(
        '{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format('module', 'type', 'input_shape', 'output_shape',
                                                                        'kernel', 'stride', 'params', 'flops', 'mem'))
    get_param_and_flops_of_each_layer(model, '', params, flops, memory)
    total_params, total_flops, total_mem = 0, 0, 0
    for param in params:
        total_params += param
    if total_params // 10 ** 6 > 0:
        total_params = str(round(total_params / 10 ** 6, 2)) + 'M'
    elif total_params // 10 ** 3:
        total_params = str(round(total_params / 10 ** 3, 2)) + 'k'
    total_params = str(total_params)
    for flop in flops:
        total_flops += flop
    for mem in memory:
        total_mem += mem

    infer_time = time.time() - t
    print('Flops:  {}'.format(flops_to_string(total_flops / batch_size)))
    print('Params: {}'.format(total_params))
    print('Memory usage: {} GB'.format(total_mem / batch_size / 1e9))
    print('Infer time: {}s'.format(infer_time / batch_size))
