import os

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from PIL import Image
from tqdm import tqdm


from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decode_outputs, non_max_suppression
from .utils_map import get_coco_map, get_map


from .dataloader_for_DAUB import source_seqDataset, dataset_collate
from torch.utils.data import DataLoader


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []

        self.source_loss = []
        self.source_ada_loss = []
        self.consistency_loss = []
        self.target_loss = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)        
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, source_loss, source_ada_loss, consistency_loss, target_loss):
        self.losses.append(loss)
        self.source_loss.append(source_loss)    
        self.source_ada_loss.append(source_ada_loss)        
        self.consistency_loss.append(consistency_loss)        
        self.target_loss.append(target_loss)        


        with open(os.path.join(self.log_dir, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_source_loss.txt"), 'a') as f:
            f.write(str(source_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_source_ada_loss.txt"), 'a') as f:
            f.write(str(source_ada_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_consistency_loss.txt"), 'a') as f:
            f.write(str(consistency_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_target_loss.txt"), 'a') as f:
            f.write(str(target_loss))
            f.write("\n")

        self.writer.add_scalar('train_loss', loss, epoch)
        self.writer.add_scalar('source_loss', source_loss, epoch)
        self.writer.add_scalar('source_ada_loss', source_ada_loss, epoch)
        self.writer.add_scalar('consistency_loss', consistency_loss, epoch)
        self.writer.add_scalar('target_loss', target_loss, epoch)
        self.loss_plot()



    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.source_loss, 'green', linewidth = 2, label='source_loss')
        plt.plot(iters, self.source_ada_loss, 'blue', linewidth = 2, label='source_ada_loss')
        plt.plot(iters, self.consistency_loss, 'black', linewidth = 2, label='consistency_loss')
        plt.plot(iters, self.target_loss, 'orange', linewidth = 2, label='target_loss')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")



class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, map_out_path=".temp_map_out", 
                 max_boxes=100, confidence=0.05, nms_iou=0.5, 
                 letterbox_image=False, 
                 MINOVERLAP=0.5, eval_flag=True, period=5,
                 source_train_annotation_path=None, val_source_train_dataset_length=None
                 ):
        super(EvalCallback, self).__init__()
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines

        self.log_dir            = log_dir
        self.cuda               = cuda
       
        self.map_out_path       = os.path.join(log_dir, map_out_path)
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

        self.source_train_annotation_path = source_train_annotation_path
        self.val_source_train_dataset_length = val_source_train_dataset_length


    def get_map_txt(self, image_id, images, class_names, map_out_path,
                    source_images):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(images[0])[0:2])

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        images       = [cvtColor(image) for image in images]

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = [resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image) for image in images]
        


        image_data = [np.transpose(preprocess_input(   np.array(image, dtype='float32')  ), (2, 0, 1)) for image in image_data] 
        image_data = np.stack(image_data, axis=1)  
        image_data  = np.expand_dims(image_data, 0)  
 
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

                source_images = source_images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images) 

            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, image_shape, self.letterbox_image, 
                                          conf_thres=self.confidence, nms_thres=self.nms_iou)
            if results[0] is None: 
                return 
            
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]


        top_100     = np.argsort(top_label)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            # 取一共6位（包括0和小数点）
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        source_train_dataset = source_seqDataset(self.source_train_annotation_path, self.input_shape[0], 5, 'train', 
                                                 length=self.val_source_train_dataset_length)
        shuffle = True

        source_DataLoader             = iter(DataLoader(source_train_dataset, shuffle = shuffle, batch_size =1, num_workers = 1, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=None))
        

        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")

            for annotation_line in tqdm(self.val_lines):
                souce_batch = next(source_DataLoader)
                source_images, _, _ = souce_batch[0], souce_batch[1], souce_batch[2]

                
                line        = annotation_line.split()
                image_id    = line[0].split('/')[-2] + '-' + line[0].split('/')[-1][:-4] 
  
                #------------------------------#
                #   读取图像
                #------------------------------#
                images = get_history_imgs(line[0])
                images = [Image.open(item) for item in images]

                #------------------------------#
                #   获得真实框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, images, self.class_names, self.map_out_path, 
                                 source_images)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id +".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                print('get_coco_map ################################')
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                print('get_map ################################')

                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            print()





import glob
def get_history_imgs(line, radius=2):  
    dir_path = line.replace(line.split('/')[-1], '')  
    file_type = line.split('.')[-1]  # bmp
    index = int(line.split('/')[-1][:-4])  # 如0

    images_list = sorted(glob.glob(dir_path + f'/*.{file_type}'))
    nfs = len(images_list)

    idx_list = list(range(index - radius, index + radius + 1))
    idx_list = np.clip(idx_list, 0, nfs-1)

    
    images = []
    for id in idx_list:
        images.append(dir_path + str(id) + '.' + file_type)
    return images



