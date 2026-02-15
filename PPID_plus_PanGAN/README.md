# 📖 Introduction

This competing approach consists of two sequential stages:

1. **Demosaicing** using **PPID**
2. **Pansharpening** using **PanGAN**

Notably, the demosaicing network does **not** require training.  
Only the pansharpening network (**PanGAN**) is trained in this framework.

---

## 📂 Dataset Structure

The dataset is organized as follows:
```
Dataset/
├── CAVE/
│ ├── train/
│ └── test/
├── ICVL/
│ ├── train/
│ └── test/
└── real_world/
├── train/
└── test/
```

## 🚀 Usage

The command lines for **training**, **generation**, **testing**, and **visualization** are provided below.

# 1. Train
## 🟢 CAVE 
### pansharpening
```
python train_pansharpening_simulate.py --idx 2 --data_path ../../DataSet --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 3
```

## 🔵 Pavia
### pansharpening
```
python train_pansharpening.py --idx 4 --data_path ../../DataSet --dataset pavia --epochs 50 --train_size 64 --stride 16 --batch_size 8 --lr_decay --save_freq 5 --device 4
```

## 🟠 Chikusei
### pansharpening
```
python train_pansharpening.py --idx 6 --data_path ../../DataSet --dataset chikusei --epochs 100 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 5 --device 0
```

# 2. Generate
## 🟢 CAVE
### generate all
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_ps_model ./2/model/best_758.pth
```

### generate single
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_ps_model ./2/model/best_758.pth --data_id jelly_beans_ms.mat
```

## 🔵 Pavia
### generate all
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset pavia --load_ps_model ./4/model/best_37.pth
```

### generate single
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset pavia --load_ps_model ./4/model/best_37.pth --data_id 1.mat
```

## 🟠 Chikusei
### generate all
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset chikusei --load_ps_model ./6/model/best_90.pth
```

### generate single
```
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset chikusei --load_ps_model ./6/model/best_90.pth --data_id 2.mat
```

# 3. test
## 🟢 Cave
### test all
```
python test.py --idx 1 --data_path ./CAVE/1/result/mat/
```

### test single
```
python test.py --idx 1 --data_path ./CAVE/1/result/mat/ --data_id jelly_beans_ms.mat
```
## 🔵 Pavia
### test all
```
python test.py --idx 1 --data_path ./pavia/1/result/mat/
```

### test single
```
python test.py --idx 1 --data_path ./pavia/1/result/mat/ --data_id 1.mat
```
## 🟠 Chikusei
### test all
```
python test.py --idx 1 --data_path ./chikusei/1/result/mat/
```

### test single
```
python test.py --idx 1 --data_path ./chikusei/1/result/mat/ --data_id 2.mat
```

# 4. Visualize
## 🟢 CAVE
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2 --mae_level 64 --sam_level 64
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --data_type upmosaic --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_type upmosaic --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2
```

## 🔵 Pavia
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./pavia/1/result/mat/fused/ --save_path ./pavia/1/result/rgb/fused/ --data_type fused --data_id 1.mat --detach --detach_size 25 25  --detach_coordinate 20 60 20 60 --boxcolor b --boxwidth 1
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./pavia/1/result/mat/fused/ --mat_path_for_diff ./pavia/1/result/mat/gt/ --save_path ./pavia/1/result/rgb/diffmap/ --data_id 1.mat --detach --detach_size 25 25  --detach_coordinate 20 60 20 60 --boxcolor b --boxwidth 1 --mae_level 64 --sam_level 64
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --data_type upmosaic --mat_path ./pavia/1/result/mat/mosaic/ --save_path ./pavia/1/result/rgb/upmosaic/ --data_type upmosaic --data_id 1.mat --detach --detach_size 25 25  --detach_coordinate 20 60 20 60 --boxcolor b --boxwidth 1
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./pavia/1/result/mat/pan/ --save_path ./pavia/1/result/rgb/pan/ --data_type pan --data_id 1.mat --detach --detach_size 25 25  --detach_coordinate 20 60 20 60 --boxcolor b --boxwidth 1
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./pavia/1/result/mat/gt/ --save_path ./pavia/1/result/rgb/gt/ --data_type mosaic --data_id 1.mat --detach --detach_size 25 25  --detach_coordinate 20 60 20 60 --boxcolor b --boxwidth 1
```

## 🟠 Chikusei
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/fused/ --save_path ./chikusei/1/result/rgb/fused/ --data_type fused --data_id 2.mat --detach --detach_size 50 50  --detach_coordinate 100 250 100 250 --boxcolor b --boxwidth 2
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/fused/ --mat_path_for_diff ./chikusei/1/result/mat/gt/ --save_path ./chikusei/1/result/rgb/diffmap/ --data_id 2.mat --detach --detach_size 50 50  --detach_coordinate 100 250 100 250 --boxcolor b --boxwidth 2 --mae_level 64 --sam_level 64
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --data_type upmosaic --mat_path ./chikusei/1/result/mat/mosaic/ --save_path ./chikusei/1/result/rgb/upmosaic/ --data_type upmosaic --data_id 2.mat --detach --detach_size 50 50  --detach_coordinate 100 250 100 250 --boxcolor b --boxwidth 2
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/pan/ --save_path ./chikusei/1/result/rgb/pan/ --data_type pan --data_id 2.mat --detach --detach_size 50 50  --detach_coordinate 100 250 100 250 --boxcolor b --boxwidth 2
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/gt/ --save_path ./chikusei/1/result/rgb/gt/ --data_type mosaic --data_id 2.mat --detach --detach_size 50 50  --detach_coordinate 100 250 100 250 --boxcolor b --boxwidth 2
```
