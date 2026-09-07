import numpy
import os
import scipy.io as scio
import cv2
import torch
from torch.utils.data import Dataset
import pickle
import tqdm
from tqdm.contrib import tzip
import h5py
import utils
import scipy
import math, random


def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches


def crop_to_patch_4d(img, size, stride):
    H, W = img.shape[1:3]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[:, h: h + size, w: w + size, :]
                for f in range(patch.shape[0]):
                    patches.append(patch[f])
    return patches


class MakeDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan, self.hrms = [], [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            hrms_imgs = os.listdir(base_path)

            if type == "train":
                numpy.random.seed(42)
            elif type == "test":
                numpy.random.seed(22)

            for hrms_name in tqdm.tqdm(hrms_imgs):
                if args.dataset == "CAVE":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["b"]
                    hrms = hrms[:, :, 12:28]
                elif args.dataset == "pavia":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["pavia"]
                elif args.dataset == "chikusei":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["chikusei"]
                hrms = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio)].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])
                                    
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
                
                spatial_ratio = pan.shape[0] // mosaic.shape[0]
                if type == "train":
                    mosaic_patches = crop_to_patch(
                        mosaic, args.train_size // spatial_ratio, args.stride // spatial_ratio
                    )
                    pan_patches = crop_to_patch(
                        pan, args.train_size, args.stride
                    )

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.pan.append(pan)
                    self.hrms.append(hrms)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan, self.hrms], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan, self.hrms = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        if self.hrms != []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32)).permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32)).permute(2, 0, 1)
            hrms = torch.from_numpy(self.hrms[index].astype(numpy.float32)).permute(2, 0, 1)
            return mosaic, pan, hrms
        else:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32)).permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32)).permute(2, 0, 1)
            return mosaic, pan