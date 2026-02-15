from model import Network, Degrade_R, degrade_dm
import torch
import os
import numpy
import scipy.io as scio
import cv2
import argparse
from tqdm.contrib import tzip
import tqdm
import utils
import h5py
import random

numpy.random.seed(22)

def main(args):
    dir_dataset = os.path.join("./", args.dataset)
    if not os.path.exists(dir_dataset):
        os.mkdir(dir_dataset)

    dir_idx = os.path.join(dir_dataset, str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)

    device = "cpu" if args.cpu == True else f"cuda:{args.device}"

    fuse_net = Network(args)
    degrade_r = Degrade_R(args)

    if args.load_model == "":
        print("==> No fuse_net checkpoint loaded!")
    else:
        print("==> Load fuse_net checkpoint: {}".format(args.load_model))
        csd = torch.load(args.load_model, map_location="cpu")
        fuse_net.load_state_dict(csd["fuse_net"], strict=False)
        degrade_r.load_state_dict(csd[f"degrade_r"], strict=False)
        
    fuse_net = fuse_net.to(device)
    fuse_net.eval()
    degrade_r = degrade_r.to(device)
    degrade_r.eval()

    data_path = os.path.join(args.data_path, args.dataset, "test")
    ids, mosaics, pans, gts = [], [], [], []

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25

    if args.data_id == []:
        data_names = os.listdir(data_path)
    else:
        data_names = [file for file in os.listdir(data_path) if file in args.data_id]
    for data_name in tqdm.tqdm(data_names):
        data_dir = os.path.join(data_path, data_name)
        if args.dataset == "CAVE":
            hrms = scio.loadmat(data_dir)['b']
            hrms = hrms[:, :, 12:28]
        elif args.dataset == "pavia":
            hrms = scio.loadmat(data_dir)["pavia"]
        elif args.dataset == "chikusei":
            hrms = scio.loadmat(data_dir)["chikusei"]
        hrms = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                    :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio)].astype(numpy.float32)
        
        # MS simulate
        # downsampling
        ms_blur_tensor = torch.from_numpy(hrms).permute(2, 0, 1).unsqueeze(0)
        lrms_tensor = torch.nn.functional.avg_pool2d(ms_blur_tensor, 2, 2)
        lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

        # mosaicing
        mosaic = utils.MSFA_filter(lrms, MSFA)

        # PAN simulate
        spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
        spe_res /= spe_res.sum()
        pan = numpy.sum(hrms * spe_res, axis=-1, keepdims=True)

        mosaics.append(mosaic)
        pans.append(pan)
        gts.append(hrms)
        ids.append(data_name.split(".")[0])

    dir_mat = os.path.join(dir_idx, "result", "mat")
    if not os.path.exists(dir_mat):
        os.makedirs(dir_mat)
        
    for cnt, (idx, mosaic, pan) in enumerate(tqdm.tqdm(tzip(ids, mosaics, pans))):
        with torch.no_grad():
            try:
                mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                pan_tensor = torch.from_numpy(pan.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                hrms_tensor = fuse_net(mosaic_tensor, pan_tensor).detach()
            except:
                mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).cpu()
                pan_tensor = torch.from_numpy(pan.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).cpu()
                fuse_net = fuse_net.cpu()
                degrade_r = degrade_r.cpu()
                hrms_tensor = fuse_net(mosaic_tensor, pan_tensor).detach()

                fuse_net = fuse_net.to(device)
                degrade_r = degrade_r.to(device)

            hrms_tensor = hrms_tensor.detach()[0].cpu()

        hrms = hrms_tensor.permute(1, 2, 0).numpy()

        if args.mosaic_save == True:
            mosaic_numpy = numpy.zeros((mosaic.shape[0]//MSFA.shape[0], mosaic.shape[1]//MSFA.shape[1], MSFA.shape[0]*MSFA.shape[1])).astype(mosaic.dtype)
            for i in range(MSFA.shape[0]):
                for j in range(MSFA.shape[1]):
                    mosaic_numpy[:, :, i*MSFA.shape[1]+j] = mosaic[i::MSFA.shape[0], j::MSFA.shape[1], 0]
            if not os.path.exists(os.path.join(dir_mat, "mosaic")):
                os.mkdir(os.path.join(dir_mat, "mosaic"))
            scio.savemat(os.path.join(dir_mat, "mosaic", f"{idx}.mat"), {'mosaic': mosaic_numpy})
        if args.pan_save == True:
            if not os.path.exists(os.path.join(dir_mat, "pan")):
                os.mkdir(os.path.join(dir_mat, "pan"))
            scio.savemat(os.path.join(dir_mat, "pan", f"{idx}.mat"), {'pan': pan})
        if args.gt_save == True:
            gt = gts[cnt]
            if not os.path.exists(os.path.join(dir_mat, "gt")):
                os.mkdir(os.path.join(dir_mat, "gt"))
            scio.savemat(os.path.join(dir_mat, "gt", f"{idx}.mat"), {'gt': gt})
        if not os.path.exists(os.path.join(dir_mat, "fused")):
            os.mkdir(os.path.join(dir_mat, "fused"))
        scio.savemat(os.path.join(dir_mat, "fused", f"{idx}.mat"), {'fused': hrms})

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--mosaic_save', action='store_true', default=False, help='Determine whether to generate mosaic data.')
    parser.add_argument('--pan_save', action='store_true', default=False, help='Determine whether to generate pan data.')
    parser.add_argument('--demosaic_save', action='store_true', default=False, help='Determine whether to generate demosaic data.')
    parser.add_argument('--gt_save', action='store_true', default=False, help='Determine whether to generate gt data. Effective only when in simulated dataset.')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--dataset', type=str, default="CAVE", help='Type of satellite data.')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--noise_level', nargs="+", type=float, default=[0.01, 0.05], help='Noise level when generating simulated data.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--data_id', type=str, default=[], nargs="+",
                        help='Index of which data to be tested. If empty, then all be selected.')
    parser.add_argument('--load_model', type=str, default='', help='The pandemosaicing model to be loaded.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Determine whether to cpu.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')

    args = parser.parse_args()
    main(args)