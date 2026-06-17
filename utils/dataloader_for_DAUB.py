import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time
import torch
from .utils import  preprocess_input, cvtColor, add_gaussian_noise, calculate_suitable_std

    
def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

import random

def add_gaussian_noise(image, mean=0, std=25):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = cv2.add(image, gaussian_noise.astype(np.uint8))
    return noisy_image

def augmentation(images, boxes, h, w, hue=.1, sat=0.7, val=0.4):
    # images [5, w, h, 3], bbox [:,4]
    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = rand()<.5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    images      = np.array(images, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    for i in range(len(images)):
        hue, sat, val   = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
        dtype           = images[i].dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        images[i] = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)
    return np.array(images, dtype=np.float32), np.array(boxes, dtype=np.float32)







import glob
class target_seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5, type='train', length=None):
        super(target_seqDataset, self).__init__()
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.aug = True
        else:
            self.aug = False

        self.img_idx = []
        self.anno_idx = []

        with open(dataset_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)

            if length is not None:
                self.times = math.ceil(length / self.length)
                self.length = length
            else:
                self.times = 1

            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]))

        self.radius = self.num_frame // 2

    
    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]  
        self.suffix = '.' + file_name.split('.')[-1]

        label_data = self.anno_idx[index] 
        label_data = np.array(label_data, dtype=np.float32) 

        image_id = int(file_name.split("/")[-1][:-4]) 
        image_path = file_name.replace(file_name.split("/")[-1], '')  
        
        # idx_list = list(range(image_id - self.num_frame + 1, image_id + 1))
        idx_list = list(range(image_id - self.radius, image_id + self.radius + 1))
        images_list = sorted(glob.glob(image_path + f'/*{self.suffix}'))

        nfs = len(images_list)
        idx_list = np.clip(idx_list, 0, nfs-1) 


        sub_images_list = []
        for id in idx_list:
            img = Image.open(image_path + str(id) + self.suffix)
            
            img = cvtColor(img)
            iw, ih = img.size
            

            w_scale = w / iw
            h_scale = h / ih
            new_img = img.resize((w, h), Image.BICUBIC)
            image_data.append(np.array(new_img, np.float32))

            ######################### 
            img = Image.open(image_path + str(id) + self.suffix)
            img = img.convert('L')
            sub_images_list.append(np.array(img, np.float32))

        ############################# 
        tar = np.abs(sub_images_list[-1] - sub_images_list[-2]) + np.abs(sub_images_list[-1] - sub_images_list[-3]) 
        + np.abs(sub_images_list[-1] - sub_images_list[-4]) + np.abs(sub_images_list[-1] - sub_images_list[-5])

        idx = np.argsort(np.abs(tar).ravel())[::-1]
        tar.ravel()[idx[int(0.001 * ih * iw):(ih * iw)]] = 0
        nonzero_coords = np.argwhere(tar != 0)

        if len(label_data) > 0:
            np.random.shuffle(label_data)  
            label_data[:, [0, 2]] = label_data[:, [0, 2]] * w_scale
            label_data[:, [1, 3]] = label_data[:, [1, 3]] * h_scale

            
            label_data[:, 0:2][label_data[:, 0:2]<0] = 0
            label_data[:, 2][label_data[:, 2]>w] = w
            label_data[:, 3][label_data[:, 3]>h] = h

            # discard invalid box
            # box_w = label_data[:, 2] - label_data[:, 0]
            # box_h = label_data[:, 3] - label_data[:, 1]
            # label_data = label_data[np.logical_and(box_w>1, box_h>1)] 

        image_data = np.array(image_data) # (5, h, w, 3)
        label_data = np.array(label_data, dtype=np.float32) # (1, 5)

        if self.aug is True:
            pass
            # image_data, label_data[:, :4] = augmentation(image_data, label_data[:, :4], h, w)
        return image_data, label_data, nonzero_coords, (ih, iw)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, box, nonzero_coords, raw_img_shape = self.get_data(index // self.times)
        images = np.transpose(preprocess_input(images), (3, 0, 1, 2))  # t, h, w, c --> c, t, h, w 
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box, nonzero_coords, raw_img_shape



def target_dataset_collate(batch):
    images = []
    bboxes = []

    nonzero_coords_list = []
    input_shape_list = []
    for img, box, nonzero_coords, input_shape in batch:
        images.append(img)
        bboxes.append(box)

        nonzero_coords_list.append(nonzero_coords)
        input_shape_list.append(input_shape)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes, nonzero_coords_list, input_shape_list




import math
class source_seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5, type='train', length=None):
        super(source_seqDataset, self).__init__()
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.aug = True
        else:
            self.aug = False

        self.img_idx = []
        self.anno_idx = []

        with open(dataset_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)

            if length is not None:
                self.times = math.ceil(length / self.length)
                self.length = length
            else:
                self.times = 1

            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]))
        
        self.radius = self.num_frame // 2
    
    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index] 
        self.suffix = '.' + file_name.split('.')[-1]

        label_data = np.copy(self.anno_idx[index]) 
        label_data = np.array(label_data, dtype=np.float32) 

        image_id = int(file_name.split("/")[-1][:-4])  
        image_path = file_name.replace(file_name.split("/")[-1], '')  


        # idx_list = list(range(image_id - self.num_frame + 1, image_id + 1))
        idx_list = list(range(image_id - self.radius, image_id + self.radius + 1))
        images_list = sorted(glob.glob(image_path + f'/*{self.suffix}'))

        nfs = len(images_list)
        idx_list = np.clip(idx_list, 0, nfs-1) 

        for id in idx_list: 
            img = Image.open(image_path + str(id) + self.suffix)
            
            img = cvtColor(img)
            iw, ih = img.size


            w_scale = w / iw
            h_scale = h / ih
            new_img = img.resize((w, h), Image.BICUBIC)
            image_data.append(np.array(new_img, np.float32))

        if len(label_data) > 0:
            np.random.shuffle(label_data) 
            label_data[:, [0, 2]] = label_data[:, [0, 2]] * w_scale
            label_data[:, [1, 3]] = label_data[:, [1, 3]] * h_scale

            
            label_data[:, 0:2][label_data[:, 0:2]<0] = 0
            label_data[:, 2][label_data[:, 2]>w] = w
            label_data[:, 3][label_data[:, 3]>h] = h
            # discard invalid box
            # box_w = label_data[:, 2] - label_data[:, 0]
            # box_h = label_data[:, 3] - label_data[:, 1]
            # label_data = label_data[np.logical_and(box_w>1, box_h>1)] 

        image_data = np.array(image_data) # (5, h, w, 3)
        label_data = np.array(label_data, dtype=np.float32) # (1, 5)

        if self.aug is True:
            pass
            # image_data, label_data[:, :4] = augmentation(image_data, label_data[:, :4], h, w)
        return image_data, label_data, (ih, iw)


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, box, raw_img_shape = self.get_data(index // self.times)
        images = np.transpose(preprocess_input(images), (3, 0, 1, 2))  # t, h, w, c --> c, t, h, w 
        # if len(box) != 0:
        #     box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        #     box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box, raw_img_shape



def dataset_collate(batch):
    images = []
    bboxes = []
    raw_img_shape_list = []
    for img, box, raw_img_shape in batch:
        images.append(img)
        bboxes.append(box)
        raw_img_shape_list.append(raw_img_shape)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes, raw_img_shape_list

