


# *SCORE*
The PyTorch implementation for the SCORE: Unsupervised moving infrared small target detector adaptation by invariant and discriminative feature learning.

This project takes the [DFAR](https://github.com/xiaomingxige/DFAR) detector as an example. Similar steps can be followed for the [SSTNet](https://github.com/UESTC-nnLab/SSTNet) and [STMENet](https://github.com/UESTC-nnLab/STME) detectors.

## 1. Pre-request
### 1.1. Environment
```bash
conda create -n SCORE python=3.10.11
conda activate SCORE
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

git clone --depth=1 https://github.com/xiaomingxige/SCORE
cd SCORE
pip install -r requirements.txt
```
### 1.2. Other dependencies
For other environment dependencies, please refer to the [DFAR](https://github.com/xiaomingxige/DFAR) setup instructions, or follow the configuration steps provided below.

**Build DCNv2**
```bash
cd nets/ops/dcn/
# You may need to modify the paths of cuda before compiling.
bash build.sh
```


### 1.3. Datasets
Our experiments are conducted on three datasets: **DAUB**, **IRDST**, and **SIRSTD**. Six domain adaptation experiments are performed: **DAUB→ IRDST**, **DAUB→ SIRSTD**, **IRDST→ DAUB**,  **IRDST→ SIRSTD**, **SIRSTD→ DAUB**, and **SIRSTD→ IRDST**. 

We would like to thank [SSTNet](https://github.com/UESTC-nnLab/SSTNet)  for providing DAUB and  IRDST datasets download links:
- **DAUB**: [Download Link](https://pan.baidu.com/s/1nNTvjgDaEAQU7tqQjPZGrw?pwd=saew) (Extraction Code: saew)
- **IRDST**: [Download Link](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr) (Extraction Code: rrnr)

 **Note:** We found that the original annotations in the IRDST dataset are not perfectly aligned with the targets and exhibit a systematic shift toward the lower-right direction. 
To address this issue, we corrected the annotations by shifting all bounding boxes one pixel toward the upper-left direction. The revised annotations (`coco_train_re_IRDST.txt` and `coco_val_re_IRDST.txt`)  are provided in this repository.
 An example comparison is shown below：
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc9e59787c52470d9790e1c4e9d33b36.png#pic_center)




 **SIRSTD:** In the original [SIRSTD](https://github.com/aurora-sea/SIRSTD) dataset, all images are stored in a single directory without sequence organization. To facilitate training and evaluation, we provide a restructured version where the images are grouped into sequences. In addition, we have converted the annotations to the COCO format (`coco_train_SIRSTD.txt` and `coco_val_SIRSTD.txt`). Download link for the reorganized dataset:
 [Download Link](https://pan.baidu.com/s/1DTMfmYM1M0JXgYeem7ENHw) (Extraction Code: SCOR)

## 2. Pre-trained source model
Follow the DFAR training pipeline to obtain the source-domain model weights on the three datasets. We also provide the pre-trained weights in `./model_data/oracle/.`

 **Note:** To ensure a fair comparison among the DFAR, SSTNet, and STMENet detectors, we modified the input resolution of DFAR from the original 544×544 to 512×512. In addition, the optimizer settings of DFAR were adjusted to match those used by SSTNet and STMENet. The modified configuration can be found in `DAUB_to_SIRSTD.py`.
## 3. Train
Taking **DAUB_to_SIRSTD** as an example, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u  DAUB_to_SIRSTD.py >  DAUB_to_SIRSTD.out &
```

For other transfer scenarios, you can proceed in a similar manner after modifying the corresponding file paths (source_train_annotation_path, target_train_annotation_path, target_val_annotation_path, and model_path).
## 4. Test
We utilize 1 NVIDIA GeForce RTX 4090D GPU for testing. For the **DAUB_to_SIRSTD**：
```bash
CUDA_VISIBLE_DEVICES=0 python vid_DAUB_to_SIRSTD.py  
```
## 5. Visualization
For the **DAUB_to_SIRSTD**：
```bash
python predict_DAUB_to_SIRSTD.py
```
## Citation
If you find this project is useful for your research, please cite:

```bash
@article{LUO2026106686,
title = {Unsupervised moving infrared small target detector adaptation by invariant and discriminative feature learning},
journal = {Infrared Physics & Technology},
volume = {157},
pages = {106686},
year = {2026},
author = {Dengyan Luo and Yanping Xiang and Hu Wang and Yan Gan and Mao Ye},
}
```


## Acknowledgements
This work is based on [DFAR](https://github.com/xiaomingxige/DFAR), [SSTNet](https://github.com/UESTC-nnLab/SSTNet), [IRDST](https://xzbai.buaa.edu.cn/datasets.html), and [ST-Trans](https://github.com/aurora-sea/SIRSTD). Thank them for sharing the codes or datasets.

