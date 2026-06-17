import numpy as np
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image
    # return np.clip(image, -1, 1)



# def preprocess_input(new_img):
#     # new_img /= 255.0
#     # new_img -= np.array([0.485, 0.456, 0.406])
#     # new_img /= np.array([0.229, 0.224, 0.225])

#     max_value = np.max(new_img)
#     min_value = np.min(new_img)
#     # new_img = (new_img-min_value)/(max_value-min_value)*255  # 归一化到0-1 
#     new_img = (new_img-min_value)/(max_value-min_value)  # 归一化到0-1  
#     return new_img




#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 130)
    print('|%25s | %100s|' % ('keys', 'values'))
    print('-' * 130)
    for key, value in kwargs.items():
        print('|%25s | %100s|' % (str(key), str(value)))
    print('-' * 130)




import cv2
def add_gaussian_noise(image, mean=0, std=25):
    """
    为图像添加高斯噪声
    :param image: 输入图像（numpy数组格式）
    :param mean: 高斯分布的均值，默认为0
    :param std: 高斯分布的标准差，默认为25
    :return: 添加高斯噪声后的图像
    """
    # 获取图像的形状
    rows, cols, channels = image.shape
    
    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, std, (rows, cols, channels))
    gaussian_noise = gaussian_noise.reshape(rows, cols, channels)
    
    # 将噪声添加到图像中
    noisy_image = cv2.add(image, gaussian_noise.astype(np.uint8))
    return noisy_image



def calculate_suitable_std(image, target_snr=40):
    mean_image = np.mean(image)
    std = mean_image / target_snr
    return std
