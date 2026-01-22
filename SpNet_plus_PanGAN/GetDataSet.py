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
import pyexr
import utils
from scipy import signal

def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches

class MakeDatasetforDemosaic(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        self.train_size = args.train_size
        if not os.path.exists(cache_path):
            self.mosaic, self.target = [], []
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
                elif args.dataset == "ICVL":
                    hrms = h5py.File(os.path.join(base_path, hrms_name))["rad"][:]
                    hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                    hrms /= hrms.max((0, 1))
                elif args.dataset == "Kaist":
                    hrms = pyexr.open(os.path.join(base_path, hrms_name)).get()
                hrms_select_bands = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        12:28].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])
                
                # MS simulate
                # downsampling
                ms_blur_tensor = torch.from_numpy(hrms_select_bands).permute(2, 0, 1).unsqueeze(0)
                lrms_tensor = torch.nn.functional.avg_pool2d(ms_blur_tensor, 2, 2)
                lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

                # mosaicing
                mosaic = utils.MSFA_filter(lrms, MSFA)

                # PAN simulate
                spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
                spe_res /= spe_res.sum()
                pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)

                mosaic_reshape = utils.MSFA_filter_inv(mosaic, MSFA)
                mosaic_reshape = mosaic_reshape[:mosaic_reshape.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                                :mosaic_reshape.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio)]
    
                mosaic_down = utils.MSFA_filter(mosaic_reshape, MSFA)

                if type == "train":
                    mosaic_down_patches = crop_to_patch(mosaic_down, args.train_size, args.stride)
                    mosaic_patches = crop_to_patch(mosaic_reshape, args.train_size, args.stride)

                    self.mosaic += mosaic_down_patches
                    self.target += mosaic_patches
                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.target.append(lrms)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.target], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.target = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
        mosaic = mosaic.permute(2, 0, 1)
        target = torch.from_numpy(self.target[index].astype(numpy.float32))
        target = target.permute(2, 0, 1)

        return mosaic, target
        
class MakeDatasetforPansharpening(Dataset):
    def __init__(self, args, type="train", demosaic_net=None):
        self.train_size = args.train_size
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.lrms, self.pan, self.hrms = [], [], [], []
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
                elif args.dataset == "ICVL":
                    hrms = h5py.File(os.path.join(base_path, hrms_name))["rad"][:]
                    hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                    hrms /= hrms.max((0, 1))
                elif args.dataset == "Kaist":
                    hrms = pyexr.open(os.path.join(base_path, hrms_name)).get()
                hrms_select_bands = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        12:28].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])
                # MS simulate
                # downsampling
                ms_blur_tensor = torch.from_numpy(hrms_select_bands).permute(2, 0, 1).unsqueeze(0)
                lrms_tensor = torch.nn.functional.avg_pool2d(ms_blur_tensor, 2, 2)
                lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

                # mosaicing
                mosaic = utils.MSFA_filter(lrms, MSFA)

                # PAN simulate
                spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
                spe_res /= spe_res.sum()
                pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)

                mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    try:
                        demosaic = demosaic_net(mosaic_tensor.to(f"cuda:{args.device}"))[0].cpu().numpy().transpose(1, 2, 0)
                    except:
                        demosaic = demosaic.cpu()
                        demosaic = demosaic_net(mosaic_tensor)[0].cpu().numpy().transpose(1, 2, 0)
                        demosaic = demosaic.to(f"cuda:{args.device}")

                spatial_ratio = pan.shape[0] // demosaic.shape[0]
                if type == "train":
                    demosaic_patches = crop_to_patch(demosaic, args.train_size//spatial_ratio, args.stride//spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)

                    self.lrms += demosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.lrms.append(demosaic)
                    self.pan.append(pan)
                    self.hrms.append(hrms_select_bands)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.lrms, self.pan, self.hrms], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.lrms, self.pan, self.hrms = pickle.load(f)

    def __len__(self):
        return len(self.lrms)

    def __getitem__(self, index):
        if self.hrms != []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            lrms = torch.from_numpy(self.lrms[index].astype(numpy.float32))
            lrms = lrms.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)
            hrms = torch.from_numpy(self.hrms[index].astype(numpy.float32))
            hrms = hrms.permute(2, 0, 1)

            return mosaic, lrms, pan, hrms
        else:
            lrms = torch.from_numpy(self.lrms[index].astype(numpy.float32))
            lrms = lrms.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)

            return lrms, pan