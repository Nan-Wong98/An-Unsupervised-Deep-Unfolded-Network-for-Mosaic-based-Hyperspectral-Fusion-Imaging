# An Unsupervised Deep Unfolded Network for Mosaic-based Hyperspectral Fusion Imaging (UDUN)

[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-Paper-00629B?style=flat-square&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11614557)

Official PyTorch implementation of the paper: **"An Unsupervised Deep Unfolded Network for Mosaic-based Hyperspectral Fusion Imaging (UDUN)"**.

## 📖 Introduction

In this repository, we provide the code for our proposed **UDUN** along with implementations of **6 state-of-the-art (SOTA) competing methods**. 

Our goal is to provide a comprehensive benchmark for mosaiced and PAN image fusion. All methods share a **unified framework** and consistent execution logic, making it easy for researchers to reproduce results and compare performance.

## 📂 Project Structure

This repository contains **7 individual projects** (sub-folders), including:

- **Proposed Method:** `Ours (UDUN)`
- **Competing Methods:** `EFN`, `EBVIF`, `PPID_PanGAN`, `PPID_VBPN`, `LSAN_PanGAN`, `LSAN_VBPN`.

### Unified Workflow
Each project folder follows the exact same file structure and logic:

| File Name | Function |
| :--- | :--- |
| `GetDataSet.py` | 🛠 **Data Preparation:** Generates training/testing data from raw datasets. |
| `train.py` | 🚀 **Training:** Trains the model. |
| `generate.py` | 💾 **Inference:** Generates fusion results using pretrained weights. |
| `test.py` | 📊 **Evaluation:** Calculates quantitative metrics (PSNR, SAM, ERGAS, Q2n, QNR etc.). |
| `visualize.py` | 🎨 **Visualization:** Visualizes the generated HSI and MAE map results. |

## 📦 Pretrained Weights
Pretrained weights of all the fusion methods are packaged in [Release](https://github.com/Nan-Wong98/Equivariant-High-Resolution-Hyperspectral-Imaging-via-Mosaiced-and-PAN-Image-Fusion/releases). Please refer to the directory mapping below:

### Competing Methods
Two stage: demosaicing + pansharpening

| Component | CAVE | Chikusei | Real-world | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Demosaicing** | Folder "1" | Folder "3" | Folder "5" | Used for LSAN |
| **Pansharpening**| Folder "2" | Folder "4" | Folder "6" | Used for PanGAN, VBPN |

> **Note:** PPID uses a traditional demosaicing algorithm and does not require pretrained weights for the first stage.

### Proposed Method (UDUN)
**EFN**, **EBVIF**, and **UDUN** are the one-step fusion framework.

| Method | CAVE | Chikusei | Real-world |
| :--- | :---: | :---: | :---: |
| EFN | Folder "1" | Folder "2" | Folder "3" |
| EBVIF | Folder "1" | Folder "2" | Folder "3" |
| **UDUN (Proposed)** | Folder "1" | Folder "2" | Folder "3" |


## 💾 Data Preparation

We utilize three widely-used hyperspectral datasets (CAVE, Chikusei, Pavia Center) for our experiments. Please refer to the following instructions to prepare the data.

The dataset is organized as follows:
```
Dataset/
├── CAVE/
│ ├── train/
│ └── test/
├── Chikusei/
│ ├── train/
│ └── test/
├── PaviaC/
│ ├── train/
│ └── test/
```

### 1. Download Links
Due to the file size, we host the prepared datasets on Baidu Netdisk:

* **CAVE Dataset**
    * 📥 [Download Link](https://pan.baidu.com/s/1DrPQRold0AAR89hU9sfQ_Q)
    * 🔑 Access Code: `bk9p`

* **Chikusei Dataset**
    * 📥 [Download Link](https://pan.baidu.com/s/1wBFLZcR1lk0iqtDBtfdxlQ)
    * 🔑 Access Code: `42b3`

* **PaviaC Dataset**
    * 📥 [Download Link](https://pan.baidu.com/s/1N_SpGFF0QyhMTqKlfbYdCA)
    * 🔑 Access Code: `ba2r`

## ⚙️ Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
````

<details>

*   h5py==3.15.1
*   Imath==0.0.2
*   matplotlib==3.10.8
*   numpy==2.4.1
*   opencv_python==4.13.0.90
*   OpenEXR==3.4.4
*   pytorch_msssim==1.0.0
*   scipy==1.17.0
*   torch==2.10.0+cu126
*   torchvision==0.25.0+cu126
*   tqdm==4.67.1

</details>

## 📝 Citation

If you find this code or our dataset useful for your research, please verify strictly and cite our paper:
```
@article{wang2026unsupervised,
  title={An Unsupervised Deep Unfolded Network for Mosaic-based Hyperspectral Fusion Imaging},
  author={Wang, Nan and Mo, Aiping and Dian, Renwei and Wei, Jujue and Li, Shutao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```

## 📧 Contact
If any question, please contact with me.

E-mail: wangn@hnu.edu.cn

## 🙏 Acknowledgments
This project is based on the Equivariant Fusion Network (EFN) framework from our previous work:

"Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion" (IEEE TIP 2026)📥[Github](https://github.com/Nan-Wong98/Equivariant-High-Resolution-Hyperspectral-Imaging-via-Mosaiced-and-PAN-Image-Fusion)
