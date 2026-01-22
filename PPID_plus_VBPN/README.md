# Train
## CAVE
``` 
# pansharpening
python train_pansharpening.py --idx 2 --data_path ../../DataSet --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 4
```

## ICVL
```
# pansharpening
python train_pansharpening.py --idx 4 --data_path ../../DataSet --dataset ICVL --epochs 20 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --device 2
```

## Kaist
```
# pansharpening
python train_pansharpening.py --idx 6 --data_path ../../DataSet --dataset Kaist --epochs 20 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --device 7
```

# Generate
## CAVE
```
# all
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_ps_model ./2/model/best_138.pth

# assign
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet --dataset CAVE --load_ps_model ./2/model/best_138.pth --data_id jelly_beans_ms.mat
```

## ICVL
```
# all
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_ps_model ./4/model/best_36.pth

# assign
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet --dataset ICVL --load_ps_model ./4/model/best_36.pth --data_id peppers_0503-1330.mat
```

## Kaist
```
# all
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset Kaist --load_ps_model ./6/model/best_14.pth

# assign
python generate.py --idx 1 --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet --dataset Kaist --load_ps_model ./6/model/best_14.pth --data_id scene21_reflectance.exr
```

# test
## Cave
```
# test all
python test.py --idx 1 --data_path ./CAVE/1/result/mat/ --simulate

# test assign
python test.py --idx 1 --data_path ./CAVE/1/result/mat/ --data_id jelly_beans_ms.mat
```
## ICVL
```
# test all
python test.py --idx 1 --data_path ./ICVL/1/result/mat/ --simulate

# test assign
python test.py --idx 1 --data_path ./ICVL/1/result/mat/ --data_id peppers_0503-1330.mat
```
## Kaist
```
# test all
python test.py --idx 1 --data_path ./Kaist/1/result/mat/ --simulate

# test assign
python test.py --idx 1 --data_path ./Kaist/1/result/mat/ --data_id scene21_reflectance.mat
```

# Visualize
## Cave
```
# fused
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2

# diffmap
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2 --mae_level 64 --sam_level 64

# upmosaic
python visualize.py --visual_task rgb --spatial_ratio 8 --data_type upmosaic --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2 --data_type upmosaic

# pan
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2

# gt
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id jelly_beans_ms.mat --detach --detach_size 50 50  --detach_coordinate 180 180 180 180 --boxcolor b --boxwidth 2
``` 

## ICVL
```
# fused
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --save_path ./ICVL/1/result/rgb/fused/ --data_type fused --data_id peppers_0503-1330.mat --detach --detach_size 128 128 --detach_coordinate 400 350 400 350 --boxcolor b --boxwidth 5  

# diffmap
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --mat_path_for_diff ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/diffmap/ --data_id peppers_0503-1330.mat --detach --detach_size 128 128 --detach_coordinate 400 350 400 350 --boxcolor b --boxwidth 5 --mae_level 16 --sam_level 16

# upmosaic
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/mosaic/ --save_path ./ICVL/1/result/rgb/upmosaic/ --data_type upmosaic --data_id peppers_0503-1330.mat --detach --detach_size 128 128 --detach_coordinate 400 350 400 350 --boxcolor r --boxwidth 5

# pan
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/pan/ --save_path ./ICVL/1/result/rgb/pan/ --data_type pan --data_id peppers_0503-1330.mat --detach --detach_size 128 128 --detach_coordinate 400 350 400 350 --boxcolor r --boxwidth 5

# gt
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/gt/ --data_type mosaic --data_id peppers_0503-1330.mat --detach --detach_size 128 128 --detach_coordinate 400 350 400 350 --boxcolor b --boxwidth 5
```

## KAISt
```
# fused
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./Kaist/1/result/mat/fused/ --save_path ./Kaist/1/result/rgb/fused/ --data_type fused --data_id scene21_reflectance.mat --detach --detach_size 256 256 --detach_coordinate 1250 2450 1250 2450 --boxcolor b --boxwidth 8

# diffmap
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./Kaist/1/result/mat/fused/ --mat_path_for_diff ./Kaist/1/result/mat/gt/ --save_path ./Kaist/1/result/rgb/diffmap/ --data_id scene21_reflectance.mat --detach --detach_size 256 256 --detach_coordinate 1250 2450 1250 2450 --boxcolor b --boxwidth 8 --mae_level 16 --sam_level 100

# upmosaic
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./Kaist/1/result/mat/mosaic/ --save_path ./Kaist/1/result/rgb/upmosaic/ --data_type upmosaic --data_id scene21_reflectance.mat --detach --detach_size 256 256 --detach_coordinate 1250 2450 1250 2450 --boxcolor r --boxwidth 8

# pan
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./Kaist/1/result/mat/pan/ --save_path ./Kaist/1/result/rgb/pan/ --data_type pan --data_id scene21_reflectance.mat --detach --detach_size 256 256 --detach_coordinate 1250 2450 1250 2450 --boxcolor r --boxwidth 8

# gt
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./Kaist/1/result/mat/gt/ --save_path ./Kaist/1/result/rgb/gt/ --data_type mosaic --data_id scene21_reflectance.mat --detach --detach_size 256 256 --detach_coordinate 1250 2450 1250 2450 --boxcolor b --boxwidth 8