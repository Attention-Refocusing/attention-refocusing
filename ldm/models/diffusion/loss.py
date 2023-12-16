import math
import torch
from ldm.models.diffusion.gaussian_smoothing import GaussianSmoothing
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
def loss_one_att_outside(attn_map,bboxes, object_positions,t):
    loss = 0
    object_number = len(bboxes)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
   
    for obj_idx in range(object_number):
        
        for obj_box in bboxes[obj_idx]:
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1.
            mask_out = 1. - mask
            index = (mask == 1.).nonzero(as_tuple=False)
            index_in_key = index[:,0]* H + index[:, 1]
            att_box = torch.zeros_like(attn_map)
            att_box[:,index_in_key,:] = attn_map[:,index_in_key,:]

            att_box = att_box.sum(axis=1) / index_in_key.shape[0]
            att_box = att_box.reshape(-1, H, H)
            activation_value = (att_box* mask_out).reshape(b, -1).sum(dim=-1) #/ att_box.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value)
            
    return loss / object_number

def caculate_loss_self_att(self_first, self_second, self_third, bboxes, object_positions, t, list_res=[256], smooth_att = True,sigma=0.5,kernel_size=3 ):
    all_attn = get_all_self_att(self_first, self_second, self_third)
    cnt = 0
    total_loss = 0
    for res in list_res:
        attn_maps = all_attn[res]
        for attn in attn_maps:
            total_loss += loss_one_att_outside(attn, bboxes, object_positions,t)
            cnt += 1

    return total_loss /cnt


def get_all_self_att(self_first, self_second, self_third):
    result = {256:[], 1024:[], 4096:[], 64:[], 94:[],1054:[] ,286:[],4126:[] }
    # import pdb; pdb.set_trace()
    all_att = [self_first, self_second, self_third]
    for self_att in all_att:
        for att in self_att:
            if att != []:
                temp = att[0]
                for attn_map in temp:
                    current_res = attn_map.shape[1]
                    # print(current_res)
                    result[current_res].append(attn_map)
    return result

def get_all_attention(attn_maps_mid, attn_maps_up , attn_maps_down, res):
    result  = []
    
    for attn_map_integrated in attn_maps_up:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if H == res:
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    for attn_map_integrated in attn_maps_mid:

    # for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated[0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H==res):
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_down:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        if attn_map == []: continue
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H==res):
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    
    result = torch.cat(result, dim=0)
    result = result.sum(0) / result.shape[0]
    return result


def caculate_loss_att_fixed_cnt(attn_maps_mid, attn_maps_up, attn_maps_down, bboxes, object_positions, t, res=16, smooth_att = True,sigma=0.5,kernel_size=3 ):
    attn = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)
    
    obj_number = len(bboxes)
    total_loss = 0
   
    attn_text = attn[:, :, 1:-1]
    attn_text *= 100
    attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
    current_res =  attn.shape[0]
    H = W = current_res
    
    
    min_all_inside = 1000
    max_outside = 0
    
    for obj_idx in range(obj_number):
        num_boxes= 0
        
        for obj_position in object_positions[obj_idx]:
            true_obj_position = obj_position - 1
            att_map_obj = attn_text[:,:, true_obj_position]
            if smooth_att:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                att_map_obj = smoothing(input).squeeze(0).squeeze(0)
            other_att_map_obj = att_map_obj.clone()
            att_copy = att_map_obj.clone()

            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            
            
                if att_map_obj[y_min: y_max, x_min: x_max].numel() == 0: 
                    max_inside=1.
                    
                else:
                    max_inside = att_map_obj[y_min: y_max, x_min: x_max].max()
                if max_inside < 0.1:
                    total_loss += 6*(1. - max_inside)
                elif max_inside < 0.2:
                    total_loss += 1. - max_inside
                elif t < 15:
                    total_loss += 1. - max_inside
                if max_inside < min_all_inside:
                    min_all_inside = max_inside
                
                # find max outside the box, find in the other boxes
                
                att_copy[y_min: y_max, x_min: x_max] = 0.
                other_att_map_obj[y_min: y_max, x_min: x_max] = 0.
            
            for obj_outside in range(obj_number):
                if obj_outside != obj_idx:
                    for obj_out_box in bboxes[obj_outside]:
                        x_min_out, y_min_out, x_max_out, y_max_out = int(obj_out_box[0] * W), \
                            int(obj_out_box[1] * H), int(obj_out_box[2] * W), int(obj_out_box[3] * H)
                        
                
                        if other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].numel() == 0: 
                            max_outside_one= 0
                        else:
                            max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].max()
                        
                        att_copy[y_min_out: y_max_out, x_min_out: x_max_out] = 0.
                        
                        if max_outside_one > 0.15:
                            total_loss += 4 * max_outside_one
                        elif max_outside_one > 0.1:
                            total_loss += max_outside_one
                        elif t<15:
                            total_loss += max_outside_one
                        if max_outside_one > max_outside:
                            max_outside = max_outside_one
            
            max_background = att_copy.max()
            total_loss += len(bboxes[obj_idx]) *max_background /2.
                
    return total_loss/obj_number, min_all_inside, max_outside


def caculate_ground(ground_first, ground_second, ground_third,bboxes,
                                object_positions, t ):
    attn_ground = get_all_self_att(ground_first, ground_second, ground_third)
    attn_maps = attn_ground[286]
    loss = 0
    loss_self = 0
    object_number = len(bboxes)

    # import pdb; pdb.set_trace()
    attn_map = torch.mean(torch.stack(attn_maps), dim=0)
    # for attn_map in attn_maps:
    if t < 15:
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):

            for obj_box in bboxes[obj_idx]:
                mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1.
                mask_out = 1. - mask
                index = (mask == 1.).nonzero(as_tuple=False)

                index_in_key = index[:,0]* H + index[:, 1]
                att_box = torch.zeros_like(attn_map)
                att_box[:,index_in_key,:] = attn_map[:,index_in_key,:]
                box_ids = np.arange(0,object_number,1)
               
                att_box = att_box.sum(axis=1) / index_in_key.shape[0]
                att_box_square = att_box[:,:256].reshape(-1, H, H)

                activation_value = (att_box_square*mask_out ).reshape(b, -1).sum(dim=-1) #/ att_box.reshape(b, -1).sum(dim=-1)
                cp_att_box= torch.zeros(size=(att_box.shape[0], 30))
                cp_att_box[:,:] = att_box[:,256:]
                cp_att_box[:,obj_idx] = 0
                loss_self += torch.mean(activation_value)
                loss += cp_att_box.amax(dim=1).mean()


    return loss / object_number, loss_self/ object_number

