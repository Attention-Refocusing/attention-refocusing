import cv2
import os
import os.path
import sys
import numpy as np
import pandas as pd
import sys
import pickle

import json
import os
def detect_color_hue_based(hue_value):
    if hue_value < 15:
        color = "red"
    elif hue_value < 22:
        color = "orange"
    elif hue_value < 39:
        color = "yellow"
    elif hue_value < 78:
        color = "green"
    elif hue_value < 131:
        color = "blue"
    else:
        color = "red"

    return color

coco_class_idx = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane":4, "bus":5, "train":6, "truck":7, "boat":8, "traffic light":9, "fire hydrant":10,
    "stop sign":11, "parking meter":12, "bench":13, "bird":14, "cat":15, "dog":16, "horse":17, "sheep":18, "cow":19, "elephant":20, "bear":21, "zebra":22,
    "giraffe":23, "backpack":24, "umbrella":25, "handbag":26, "tie":27, "suitcase":28, "frisbee":29, "skis":30, "snowboard":31, "sports ball":32,
    "kite":33, "baseball bat":34, "baseball glove":35, "skateboard":36, "surfboard":37, "tennis racket":38, "bottle":39, "wine glass":40, "cup":41, 
    "fork":42, "knife":43, "spoon":44, "bowl":45, "banana":46, "apple":47, "sandwich":48, "orange":49, "broccoli":50, "carrot":51, "hot dog":52, "pizza":53, 
    "donut":54, "cake":55, "chair":56, "couch":57, "potted plant":58, "bed":59, "dining table":60, "toilet":61, "tv":62, "laptop":63, "mouse":64, "remote":65, 
    "keyboard":66, "cell phone":67, "microwave":68, "oven":69, "toaster":70, "sink":71, "refrigerator":72, "book":73, "clock":74, "vase":75, "scissors":76, 
    "teddy bear":77, "hair drier":78, "toothbrush":79
}

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_list = []
    for sample in gt_data:
        # Objects:
        objs = [sample['obj1'], sample['obj2']]
        for i in range(3, 5):
            if type(sample['obj'+str(i)]) is str:  # check if there is an object
                objs.append(sample['obj'+str(i)])

        # Colors:
        colors = [sample['color1'], sample['color2']]
        for i in range(3, 5):
            if type(sample['color' + str(i)]) is str:  # check if there is other colors
                colors.append(sample['color' + str(i)])

        gt_list.append({"prompt": sample['meta_prompt'], "objs": objs, "colors": colors})

    return gt_list


def load_pred(pred_masks_names, iter_idx, data_len, gt_data):
    img_masks_names_dict = {}
    for idx in range(data_len):
        prompt = gt_data[idx]["prompt"]
        # img_name = str(idx).zfill(5)+"_"+str(iter_idx).zfill(2)
        img_name = str(idx )+"_"+str(iter_idx) +'_' + prompt.replace(' ','_') 
        # import pdb; pdb.set_trace()
        img_masks_names = [pred_masks_name for pred_masks_name in pred_masks_names if img_name[:-1] in pred_masks_name]
        img_masks_names_dict[img_name] = img_masks_names

    return img_masks_names_dict


def cal_acc(gt_data, img_masks_names_dict, level, iter_idx, t2i_out_dir, in_masks_folder):
    true_counter = 0
    total_num_objs = 0
    mul = int(len(gt_data) /3)
    gt_data_per_level = gt_data[level * mul:(level + 1) * mul]
    for idx in range(len(gt_data_per_level)):  # loop on samples
        if idx + (level * mul) == 500 or idx + (level * mul) == 131: continue
        gt_objs = gt_data_per_level[idx]["objs"]
        total_num_objs += len(gt_objs)
        gt_colors = gt_data_per_level[idx]["colors"]
        prompt = gt_data_per_level[idx]["prompt"]
        img_name = str(idx + (level * mul))+"_"+str(iter_idx) +'_' + prompt.replace(' ','_') 
        img = cv2.imread(os.path.join(t2i_out_dir, img_name)+".png")
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_frame = hsv_frame[:, :, 0]
        img_masks_names_per_sample = img_masks_names_dict[img_name]
        # import pdb; pdb.set_trace()
        for obj_idx in range(len(gt_objs)):  # loop on GT objs
            # 1) make sure the classes are correct:
            gt_obj_id = coco_class_idx[gt_objs[obj_idx]]
            img_masks_name_per_class = []
            for img_masks_name in img_masks_names_per_sample:
                if int(img_masks_name.split("_")[-1].split(".")[0]) == gt_obj_id:
                    img_masks_name_per_class.append(img_masks_name)
            if len(img_masks_name_per_class):
                # found some predictions match GT class
                # 2) make sure the color is correct:
                for i in range(len(img_masks_name_per_class)):
                    mask = cv2.imread(os.path.join(in_masks_folder, img_masks_name_per_class[i]), cv2.IMREAD_GRAYSCALE)
                    mask = mask / 255.0
                    mask = mask.astype(np.uint8)  # [0->1]
                    hsv_frame_masked = np.multiply(hsv_frame, mask)
                    avg_hue = hsv_frame_masked.sum() / np.count_nonzero(hsv_frame_masked)  # average hue component
                    detected_color = detect_color_hue_based(avg_hue)
                    if detected_color == gt_colors[obj_idx]:
                        true_counter += 1
                        break
    return 100*true_counter / total_num_objs


if __name__ == "__main__":
    """
    Example:
    python hue_based_color_classifier.py  'T2I_benchmark/data/colors/output/sd_v1' 'T2I_benchmark/data/colors/colors_composition_prompts.csv' 't2i_benchmark/data/t2i_out/sd_v1/colors'
    """
    in_masks_folder = sys.argv[1]
    gt_csv = sys.argv[2]
    t2i_out_dir = sys.argv[3]
    # Load GT:
    gt_data = load_gt(csv_pth=gt_csv)
    pred_masks_names = os.listdir(in_masks_folder)
    iter_num = 1
    avg_acc = []
    acc_per_level = {0: [], 1: [], 2: []}
    for iter_idx in range(iter_num):
        for level in range(3):
            # Load Predictions:
            img_masks_names_dict = load_pred(pred_masks_names=pred_masks_names, iter_idx=iter_idx, data_len=len(gt_data), gt_data=gt_data)
            # Calculate the counting Accuracy:
            acc = cal_acc(gt_data, img_masks_names_dict, level=level, iter_idx=iter_idx, t2i_out_dir=t2i_out_dir, in_masks_folder=in_masks_folder)
            avg_acc.append(acc)
            print("Accuracy ", iter_idx, ": ", acc, "%")
            # Per level:
            acc_per_level[level].append(acc)
    all_results = {'acc':acc_per_level, 'avg':sum(avg_acc) / len(avg_acc)}
    root=in_masks_folder.split('demo')[0] + in_masks_folder.split('demo')[1]
    with open(os.path.join(root+ "result.json"), 'w') as f:
            json.dump(all_results, f, sort_keys=True, indent=4)  
    for level in range(3):
        print("----------------------------")
        if level == 0:
            print("   Easy level Results   ")
        elif level == 1:
            print("   Medium level Results   ")
        elif level == 2:
            print("   Hard level Results   ")
        print("precision: ", (sum(acc_per_level[level]) / len(acc_per_level[level])), "%")
    print("----------------------------")
    print("   Average level Results   ")
    print("Averaged Accuracy: ", (sum(avg_acc) / len(avg_acc)), "%")
