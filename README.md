# ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics

International Conference on Computer Vision (ICCV) 2019 (Oral)

Zhiyuan Zhang, Binh-Son Hua, Sai-Kit Yeung.

## Introduction
This is the code release of our paper about building a convolution and neural network for point cloud learning such that the training is fast and accurate. We address this problem by learning point features in regions called 'shells', which resolves point orders and produces local features altogether. Please find the details of the technique in our [project page](https://hkust-vgd.github.io/shellnet/).

If you found this paper useful in your research, please cite:
```
@inproceedings{zhang-shellnet-iccv19,
    title = {ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics},
    author = {Zhiyuan Zhang and Binh-Son Hua and Sai-Kit Yeung},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```

## Installation
The code is based on [PointCNN](https://github.com/yangyanli/PointCNN). Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators in the `tf_ops` folder.  

The code has been tested with Python 3.6, TensorFlow 1.13.2, CUDA 10.0 and cuDNN 7.3 on Ubuntu 14.04.

## Code Explanation
The core convolution, ShellConv, and the neural network, ShellNet, are defined in [shellconv.py](shellconv.py).

### Convolution Parameters
Let us take the `sconv_params` from [s3dis.py](setting/seg_s3dis.py) as an example:
```
ss = 8 
sconv_param_name = ('K', 'D', 'P', 'C')
sconv_params = [dict(zip(sconv_param_name, sconv_param)) for sconv_param in
                [
                 (ss*4, 4, 512, 128),
                 (ss*2, 2, 128, 256),
                 (ss*1, 1, 32, 512)]]
```
`ss` indicates the shell size which is defined as the number of points contained in each shell. Each element in `sconv_params` is a tuple of `(K, D, P, C)`, where `K` is the neighborhood size, `D` is number of shells, `P` is the representative point number in the output, and `C` is the output channel number.  Each tuple specifies the parameters of one `ShellConv` layer, and they are stacked to create a deep network.

### Deconvolution Parameters
Similarly, for deconvolution, let us look at `sdconv_params` from [s3dis.py](setting/seg_s3dis.py):
```
sdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
sdconv_params = [dict(zip(sdconv_param_name, sdconv_param)) for sdconv_param in
                [
                (ss*1,  1, 2, 1),
                (ss*2,  2, 1, 0),
                (ss*4,  4, 0, -1)]]
```
Each element in `sdconv_params` is a tuple of `(K, D, pts_layer_idx, qrs_layer_idx)`, where `K` and `D` have the same meaning as that in `sconv_params`, `pts_layer_idx` specifies the output of which `ShellConv` layer (from the `sconv_params`) will be the input of this `ShellDeConv` layer, and `qrs_layer_idx` specifies the output of which `ShellConv` layer (from the `sconv_params`) will be forwarded and fused with the output of this `ShellDeConv` layer. The `P` and `C` parameters of this `ShellDeConv` layer is also determined by `qrs_layer_idx`. Similarly, each tuple specifies the parameters of one `ShellDeConv` layer, and they are stacked to create a deep network.



## Usage
### Classification

To train a ShellNet model to classify shapes in the ModelNet40 dataset:
```
cd data_conversions
python3 ./download_datasets.py -d modelnet
cd ..
python3 train_val_cls.py
python3 test_cls_modelnet40.py -l log/cls/xxxx
```
Our pretrained model can be downloaded [here](https://gohkust-my.sharepoint.com/:u:/g/personal/saikit_ust_hk/EQsJFdSR6YJMsco8Dqss_3kBgTKD8rWfHQcZt7yR9rigTQ?e=3Bj4mK). Please put it to `log/cls/modelnet_pretrained` folder to test.


### Segmentation
We perform segmentation with various datasets, as follows.

#### ShapeNet
```
cd data_conversions
python3 ./download_datasets.py -d shapenet_partseg
python3 ./prepare_partseg_data.py -f ../../data/shapenet_partseg
cd ..
python3 train_val_seg.py -x seg_shapenet
python3 test_seg_shapenet.py -l log/seg/shellconv_seg_shapenet_xxxx/ckpts/epoch-xxx
cd evaluation
python3 eval_shapenet_seg.py -g ../../data/shapenet_partseg/test_label -p ../../data/shapenet_partseg/test_pred_shellnet_1 -a
```

#### ScanNet
Please refer to ScanNet [homepage](http://www.scan-net.org) and PointNet++ preprocessed [data](https://github.com/charlesq34/pointnet2/tree/master/scannet) to download ScanNet. After that, the following script can be used for training and testing:
```
cd data_conversions
python3 prepare_scannet_seg_data.py
python3 prepare_scannet_seg_filelists.py
cd ..
python3 train_val_seg.py -x seg_scannet
python3 test_seg_scannet.py -l log/seg/shellconv_seg_scannet_xxxx/ckpts/epoch-xxx
cd evaluation
python3 eval_scannet.py -d <path to *_pred.h5> -p <path to scannet_test.pickle>
```

#### S3DIS
Please download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html#Download). The following script performs training and testing:
```
cd data_conversions
python3 prepare_s3dis_label.py
python3 prepare_s3dis_data.py
python3 prepare_s3dis_filelists.py
cd ..
python3 train_val_seg.py -x seg_s3dis
python3 test_seg_s3dis.py -l log/seg/shellconv_seg_s3dis_xxxx/ckpts/epoch-xxx
cd evaluation
python3 s3dis_merge.py -d <path to *_pred.h5>
python3 eval_s3dis.py
```
Please notice that these command just for `Area 1` validation. Results on other Areas can be computed by modifying the `filelist` and `filelist_val` in [s3dis.py](setting/seg_s3dis.py).

#### Semantic3D
You can download our preprocessed hdf5 files and labels [here](https://gohkust-my.sharepoint.com/:u:/g/personal/saikit_ust_hk/Ea_S7Yb-n7JLp1wK5xFihfYBI6vAzHWaA548ytu5k84kdQ?e=YXRcaK). Then:
```
python3 train_val_seg.py -x seg_semantic3d
python3 test_seg_semantic3d.py -l log/seg/shellconv_seg_semantic3d_xxxx/ckpts/epoch-xxx
cd evaluation
python3 semantic3d_merge.py -d <path to *_pred.h5> -v <reduced or full>
```

If you prefer to process the data by yourself, here are the steps we used. In general, this data preprocessing of this dataset is more involved. First, please download the original [Semantic3D dataset](http://www.semantic3d.net/view_dbase.php). We then downsample the data using this [script](https://github.com/intel-isl/Open3D-PointNet2-Semantic3D). Finally, we follow PointCNN's [script](https://github.com/yangyanli/PointCNN/tree/master/data_conversions) to split the data into training and validation set, and prepare the .h5 files. 
## License
This repository is released under MIT License (see LICENSE file for details).