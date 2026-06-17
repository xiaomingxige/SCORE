import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
import torch.nn as nn
import torch.nn.functional as F


import numpy as np




def fft_domain_adaptation(source_images, target_images):
    source_images = source_images.float()
    target_images = target_images.float()
    
    B, C, H, W = source_images.shape
    
    adapted_source_images = torch.zeros_like(source_images)
    
    for c in range(C):
        source_channel = source_images[:, c, :, :]
        target_channel = target_images[:, c, :, :]
        
        source_fft = torch.fft.rfft2(source_channel)
        target_fft = torch.fft.rfft2(target_channel)
        
        target_magnitude = torch.abs(target_fft)
        
        adapted_fft = target_magnitude * torch.exp(1j * torch.angle(source_fft))
        
        adapted_channel = torch.fft.irfft2(adapted_fft)
        
        adapted_source_images[:, c, :, :] = adapted_channel.real
    return adapted_source_images


def expand_box(left, top, right, bottom, area_ratio):
    current_width = right - left
    current_height = bottom - top

    size_ratio = area_ratio ** 0.5  
    new_width = current_width * size_ratio
    new_height = current_height * size_ratio

    new_left = left - (new_width - current_width) / 2
    new_top = top - (new_height - current_height) / 2
    new_right = right + (new_width - current_width) / 2
    new_bottom = bottom + (new_height - current_height) / 2
    return new_left, new_top, new_right, new_bottom



def adjust_box(source_raw_img_shape_list, target_raw_img_shape_list, source_targets, b, w, h):
    source_out_lable_data = []
    for b_idx in range(b):
        source_raw_img_shape = source_raw_img_shape_list[b_idx]
        target_raw_img_shape = target_raw_img_shape_list[b_idx]
        source_ih, source_iw = source_raw_img_shape[0], source_raw_img_shape[1]
        target_ih, target_iw = target_raw_img_shape[0], target_raw_img_shape[1]
        

        out_lable_data = []
        source_target = source_targets[b_idx]
        num, _ = source_target.shape

        for num_idx in range(num):
            box = source_target[num_idx, :]
            left, top, right, bottom = box[0], box[1], box[2], box[3]
            left, top, right, bottom = left.item(), top.item(), right.item(), bottom.item()


            area_ratio = (source_iw * source_ih) / (target_ih * target_iw)
            left, top, right, bottom = expand_box(left, top, right, bottom, area_ratio=area_ratio)
            out_lable_data.append([left, top, right, bottom, 0])
        out_lable_data = np.array(out_lable_data, dtype=np.float32) # (1, 5)

        if len(out_lable_data) > 0:
            out_lable_data[:, 0:2][out_lable_data[:, 0:2]<0] = 0
            out_lable_data[:, 2][out_lable_data[:, 2]>w] = w
            out_lable_data[:, 3][out_lable_data[:, 3]>h] = h
            # discard invalid box
            # box_w = out_lable_data[:, 2] - out_lable_data[:, 0]
            # box_h = out_lable_data[:, 3] - out_lable_data[:, 1]
            # out_lable_data = out_lable_data[np.logical_and(box_w>1, box_h>1)] 
        out_lable_data = np.array(out_lable_data, dtype=np.float32) # (1, 5)

        if len(out_lable_data) != 0:
            out_lable_data[:, 2:4] = out_lable_data[:, 2:4] - out_lable_data[:, 0:2]
            out_lable_data[:, 0:2] = out_lable_data[:, 0:2] + (out_lable_data[:, 2:4] / 2)
        source_out_lable_data.append(out_lable_data)

    source_out_lable_data = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in source_out_lable_data]
    return source_out_lable_data



import random
import copy
def expand_list(lst, target_length):
    expanded_list = lst.copy()  
    while len(expanded_list) < target_length:
        random_element = random.choice(lst)  
        expanded_list.append(copy.deepcopy(random_element))  
    return expanded_list


from utils.utils_bbox import decode_outputs, non_max_suppression
import numpy as np
def in_box(nonzero_coords, box):
    for coord in nonzero_coords:
        y, x = coord
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return True
    return False


def generate_pseudo_label(target_ada_images, target_images, model_teacher, input_shape_list, nonzero_coords_list, b, w, h, 
                          input_shape=(512, 512), num_classes=1, letterbox_image=False, confidence=0.001, nms_iou=0.1):
    pseudo_label_list = []
    pseudo_label_images_list = []
    ########################## 输出
    for b_idx in range(b):
        input_images = target_ada_images[b_idx, :, :, :, :]
        input_images = input_images.unsqueeze(0)  # 增加batch维度
        outputs = model_teacher(input_images)

        image_shape = input_shape_list[b_idx]
        outputs = decode_outputs(outputs, input_shape)

        outputs = non_max_suppression(outputs.detach(), num_classes, input_shape, image_shape, letterbox_image, 
                                        conf_thres=confidence, nms_thres=nms_iou)

        if outputs[0] is None: 
            continue

        top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
        top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
        top_boxes   = outputs[0][:, :4]
        
        pseudo_label = []
        for i, c in enumerate(top_label):
            # predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            # score           = top_conf[i]

            top, left, bottom, right = box

            # # ################## 
            area = (right - left) *   (bottom - top) 
            if area / (image_shape[0] * image_shape[1]) > 0.001 or area / (image_shape[0] * image_shape[1]) < 0.0001:
                continue
            # ################## 
            if not in_box(nonzero_coords=nonzero_coords_list[b_idx], box=(left, top, right, bottom)):
                continue

            ############ 
            pseudo_label.append([left, top, right, bottom, 0])
        pseudo_label = np.array(pseudo_label, dtype=np.float32) 

        ################ 缩放标签
        w_scale = w / image_shape[1]  
        h_scale = h / image_shape[0]  
        if len(pseudo_label) > 0:
            np.random.shuffle(pseudo_label)  

            pseudo_label[:, [0, 2]] = pseudo_label[:, [0, 2]] * w_scale
            pseudo_label[:, [1, 3]] = pseudo_label[:, [1, 3]] * h_scale
            pseudo_label[:, 0:2][pseudo_label[:, 0:2]<0] = 0
            pseudo_label[:, 2][pseudo_label[:, 2]>w] = w
            pseudo_label[:, 3][pseudo_label[:, 3]>h] = h
            # discard invalid box
            # box_w = pseudo_label[:, 2] - pseudo_label[:, 0]
            # box_h = pseudo_label[:, 3] - pseudo_label[:, 1]
            # pseudo_label = pseudo_label[np.logical_and(box_w>1, box_h>1)] 
        pseudo_label = np.array(pseudo_label, dtype=np.float32) # (1, 5)

        ################ 
        if len(pseudo_label) != 0:
            pseudo_label[:, 2:4] = pseudo_label[:, 2:4] - pseudo_label[:, 0:2]
            pseudo_label[:, 0:2] = pseudo_label[:, 0:2] + (pseudo_label[:, 2:4] / 2)

        pseudo_label_images_list.append(target_images[b_idx:b_idx+1, :, :, :, :])
        pseudo_label_list.append(pseudo_label)
    return pseudo_label_images_list, pseudo_label_list




def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, 
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, 
                  model_teacher=None, teacher_optimizer=None
                  ):
    loss = 0
    val_loss = 0
    source_loss = 0
    source_ada_loss = 0
    consistency_loss = 0
    target_loss = 0
    
    epoch_step = epoch_step // 2 

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    model_teacher.eval()

    my_loss = nn.L1Loss()
    for iteration, (souce_batch, target_batch) in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        source_images, source_targets, source_raw_img_shape_list = souce_batch[0], souce_batch[1], souce_batch[2]
        target_images, target_targets, nonzero_coords_list, target_raw_img_shape_list = target_batch[0], target_batch[1], target_batch[2], target_batch[3]

        b, c, t, h, w = source_images.shape
        
        ################## 
        source_out_lable_data = adjust_box(source_raw_img_shape_list, target_raw_img_shape_list, source_targets, b, w, h)

        with torch.no_grad():
            if cuda:
                source_images = source_images.cuda(local_rank)
                target_images = target_images.cuda(local_rank)

                # target_targets = [ann.cuda(local_rank) for ann in target_targets]
                source_out_lable_data = [ann.cuda(local_rank) for ann in source_out_lable_data]

                
        optimizer.zero_grad()
        if not fp16:  
            ############### source domian
            ##### 
            source_outputs, source_motion_loss  = model_train(source_images)
            source_yololoss = yolo_loss(source_outputs, source_out_lable_data)

            ##### 
            source_ada_images = source_images.clone()
            for t_idx in range(t):
                source_ada_images[:, :, t_idx, :, :] = fft_domain_adaptation(source_ada_images[:, :, t_idx, :, :], target_images[:, :, t_idx, :, :]) 

            source_ada_outputs, source_ada_motion_loss  = model_train(source_ada_images)
            source_ada_yololoss = yolo_loss(source_ada_outputs, source_out_lable_data)
            ##### 
            loss_consistency = my_loss(source_yololoss, source_ada_yololoss)


            ############### target domian
            ##### 
            target_ada_images = target_images.clone()
            for t_idx in range(t):
                target_ada_images[:, :, t_idx, :, :] = fft_domain_adaptation(target_ada_images[:, :, t_idx, :, :], source_images[:, :, t_idx, :, :]) 

            pseudo_label_images_list, pseudo_label_list = generate_pseudo_label(target_ada_images, target_images, model_teacher, target_raw_img_shape_list, nonzero_coords_list, b, w, h)


            if len(pseudo_label_list) == 0:  
                continue
            if len(pseudo_label_list) < b:  
                pseudo_label_images_list = expand_list(pseudo_label_images_list, b)
                pseudo_label_list = expand_list(pseudo_label_list, b)


            pseudo_label_images_list = torch.cat(pseudo_label_images_list, dim=0)  
            pseudo_label_list = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in pseudo_label_list]


            target_outputs, target_motion_loss   = model_train(pseudo_label_images_list)
            target_yololoss = yolo_loss(target_outputs, pseudo_label_list)
            

            ###############总损失
            loss_value = source_yololoss + source_motion_loss + \
                source_ada_yololoss + source_ada_motion_loss + \
                0.1 * loss_consistency + \
                0.01 * (target_yololoss + target_motion_loss)

            loss_value.backward()
            optimizer.step()
            
            ######### 
            model_teacher.zero_grad()
            teacher_optimizer.step()

        if ema:
            ema.update(model_train)
        loss += loss_value.item()
        source_loss += (source_yololoss.item() + source_motion_loss.item())
        source_ada_loss += (source_ada_yololoss.item() + source_ada_motion_loss.item())
        consistency_loss += loss_consistency.item()
        target_loss += (target_yololoss.item() + target_motion_loss.item())
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss' : loss / (iteration + 1), 
                                'source_loss' : source_yololoss.item() + source_motion_loss.item(), 
                                'source_ada_loss' : source_ada_yololoss.item() + source_ada_motion_loss.item(), 
                                'consistency_loss' : loss_consistency.item(), 
                                'target_loss' : target_yololoss.item() + target_motion_loss.item(), 
                                'lr'  : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()



    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss/epoch_step, 
            source_loss/epoch_step, source_ada_loss/epoch_step, consistency_loss/epoch_step, target_loss/epoch_step)
 
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Train Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))