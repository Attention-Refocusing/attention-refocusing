import os
import openai
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen
import pandas as pd
import pickle
import json 

openai.api_key = os.getenv("OPENAI_API_KEY")
def read_example_prompts(file_path):
    
    with open(file_path, 'r') as f:
       data = f.read()
    return data

def generate_box(text):
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 1024 * 1024 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"four cats in the field\", \ncat: (220, 318, 380, 460)\ncat: (440, 220, 714, 460)\ncat: (858, 242, 1002, 560)\ncat: (350, 694, 606, 846)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (482, 200, 804, 592)\ndog: (428, 634, 820, 970)\nthe prompt: \"five balls in the room\"\nball: (148, 560, 386, 824)\nball: (84, 138, 420, 404)\nball: (588, 104, 922, 368)\nball: (620, 436, 912, 672)\nball: ( 640, 750, 896, 964)\nthe prompt: \"a cat sitting on the car\"\nBecause the cat sitting on the car so the car bellow the cat and cat in the surface of car, therefore the result\ncat: (305, 384, 590, 600)\ncar: (100, 600, 928, 906)\n"
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image, spacial relationship of the objects should be followed in the description and size of the objects should be as large as possible. Each object in the image is one rectangle or square box in the layout. You should return each object and the corresponding coordinate of its boxes. the size of the image is 512 * 512.\nthe prompt :\"four cats in the field\", \ncat: (110, 159, 190, 230)\ncat: (220, 110, 357, 230)\ncat: (429, 121, 501, 280)\ncat: (175, 347, 303, 423)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (241, 100, 402, 296)\ndog: (217, 317, 410, 485)\nthe prompt: \"five balls in the room\"\nball: (74, 280, 193, 412)\nball: (42, 69, 210, 202)\nball: (294, 52, 461, 184)\nball: (310, 218, 456, 336)\nball: ( 320, 375, 448, 482)\nthe prompt: \"a cat sitting on the car\"\nBecause the cat sitting on the car so the car bellow the cat and cat in the surface of car, therefore the result\ncat: (153, 192, 295, 300)\ncar: (50, 300, 464, 453)\n"
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 1024 * 1024 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"three cats in the field\", \ncat: (58, 131, 456, 450)\ncat: (185, 559, 608, 857)\ncat: (539, 106, 962, 527)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (93, 305, 463, 776)\ndog: (540, 185, 938, 682)\nthe prompt: \"four balls in the room\"\nball: (65, 52, 487, 492)\nball: (566, 134, 1004, 513)\nball: (517, 610, 887, 998)\nball: (57, 573, 427, 981)\n"
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 512 * 512 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"three cats in the field\", \ncat: (51, 82, 399, 279)\ncat: (288, 128, 472, 299)\ncat: (27, 355, 418, 494)\nthe prompt: \"a cat on the left of a dog on the road\"\ncat: (63, 196, 223, 394)\ndog: (289, 131, 466, 360)\nthe prompt: \"four balls in the room\"\nball: (72, 81, 254, 243)\nball: (316, 44, 483, 218)\nball: (287, 295, 453, 462)\nball: (50, 323, 196, 484)\nthe prompt: \"A donut to the right of a toilet\"\ndonut: (287, 140, 467, 335)\ntoilet: (31, 97, 216, 286)\nthe prompt: \"A cat sitting on the top of a car\"\ncar: (94, 236, 414, 407)\ncat: (124, 139, 273, 252)\nthe prompt: \"A dog underneath a tree\"\ndog: (133, 232, 308, 445)\ntree: (121, 29, 324, 258)\n"
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 1024 * 1024 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"three cats in the field\", \ncat: (103, 164, 799, 559)\ncat: (577, 257, 944, 599)\ncat: (53, 710, 837, 988)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (125, 392, 447, 789))\ndog: (578, 263, 933, 721)\nthe prompt: \"four balls in the room\"\nball: (144, 163, 508, 487)\nball: (633, 87, 967, 437)\nball: (574, 591, 906, 925)\nball: (100, 647, 392, 969)\n"
    # example_prompt = read_example_prompts('prompt.txt')
    example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 512 * 512 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"three cats in the field\", \ncat: (51, 82, 399, 279)\ncat: (288, 128, 472, 299)\ncat: (27, 355, 418, 494)\nthe prompt: \"a cat on the left of a dog on the road\"\ncat: (63, 196, 223, 394)\ndog: (289, 131, 466, 360)\nthe prompt: \"four balls in the room\"\nball: (72, 81, 254, 243)\nball: (316, 44, 483, 218)\nball: (287, 295, 453, 462)\nball: (50, 323, 196, 484)\nthe prompt: \"A donut to the right of a toilet\"\ndonut: (287, 140, 467, 335)\ntoilet: (31, 97, 216, 286)\nthe prompt: \"A cat sitting on the top of a car\"\ncar: (94, 236, 414, 407)\ncat: (124, 139, 273, 252)\nthe prompt: \"A dog underneath a tree\"\ndog: (133, 232, 308, 445)\ntree: (121, 29, 324, 258)\nthe prompt: \"a small ball is put on the top of a box on the table. there is a red vase on the right of the box on the table\"\nsmall ball: (92, 30, 165, 134)\nbox: (93, 132 , 205, 324, 310)\nred vase: (214, 164, 297, 301)\ntable: (36, 259, 418, 463)\n"
    print('example ', example_prompt)
    prompt = example_prompt  + text
    # call api
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # complete text
    completed_text = response['choices'][0]['text']
    print('text', completed_text)
    # getting boxes and object names
    boxes = completed_text.split('\n')
    d = {}
    name_objects = []
    boxes_of_object = []
    # convert boxes from string to int
    # import pdb; pdb.set_trace()
    for b in boxes:
        if b == '': continue
       
        b_split = b.split(":")
        name_objects.append(b_split[0])
        boxes_of_object.append(text_list(b_split[1]))

    
    return name_objects, boxes_of_object
def load_json(path_file):
    with open(path_file) as f:
        data = json.load(f)
    return data

def generate_box_gpt4(inputs):
    # messages =[{"role": "user", "content": "You should use spatial and numerical understanding, I will provide the description of an image, you should output the corresponding layout of this image including . The size of the image is 512 * 512."}, {"role",: "user", "content": "three cats in the field"}, {"role": "system", "content": "a cat: (51, 82, 399, 279)\na cat: (288, 128, 472, 299)\na cat: (27, 355, 418, 494)"}]
    # messages.append({"role": "user", "content": "A donut to the right of a toilet"})
    # messages.append({"role": "system", "content": "a donut: (287, 140, 467, 335)\na toilet: (31, 97, 216, 286)"})
    # messages.append({"role": "user", "content": "a cat in the middle of a car and a chair"})
    # messages.append({"role": "system", "content": "a car: (10, 128, 202, 384)\na cat: (218, 176, 294, 336)\na chair: (330, 224, 482, 320)"})
    # messages.append({"role": "system", "content": "The size of boxes should  reflect the relative sizes of the objects in the real world.  So the size of car bigger than chair, and chair is bigger than cat."})
    # constraint = 'Can you return names and corresponding coodinate locations of objects with the prompt: '
    message = 'Provide box coordinates for an image with' + inputs

    messages = load_json('examples/example3.json')
    messages.append(
            {"role": "user", "content": message},
        )
    chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
    completed_text = chat.choices[0].message.content
    print('text gpt', completed_text)
    boxes = completed_text.split('\n')
    d = {}
    name_objects = []
    boxes_of_object = []
    for b in boxes:
        if b == '': continue
        if not '(' in b: continue 
        b_split = b.split(":")
        name_objects.append(b_split[0])
        boxes_of_object.append(text_list(b_split[1]))
    return name_objects, boxes_of_object

def draw_box_2(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    for i, bbox in enumerate(boxes):
        for box in bbox:
            if i==0:
               
                draw.rectangle([(box[0] * 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='red', width=6)
                
            elif i==1:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='green', width=6)
            else:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='blue', width=6)
    image.save(os.path.join(output_folder, img_name))

def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >0: 
                if  row[1] == t:
                    list_prompts.append(row)
    return list_prompts

def read_txt_label(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for x in f:
            x = x.replace(' \n', '')
            x = x.replace('\n', '')
            x = x.split(',')
            labels.update({x[0]: x[2]})
    return labels

def draw_box(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        mean_box_x, mean_box_y = int((box[0] + box[2] )/ 2) + int((box[1] + box[3] )/ 2)
        draw.text((mean_box_x, mean_box_y), t, fill=200,font=font )
    image.save(os.path.join(output_folder, img_name))

def save_img(folder_name, img, prompt, iter_id, img_id):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.jpg'
    img.save(os.path.join(folder_name, img_name))

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []

    for sample in gt_data:
        meta.append([sample['meta_prompt']])
        syn_prompt.append([sample['synthetic_prompt']])
    return meta, syn_prompt

def load_box(pickle_file):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f)
    return data
def read_txt_hrs(filename):
    result = []
    with open(filename) as f: 
        for x in f:
            result.append([x.replace('\n','')])
    return result

def format_box(names, boxes):
    result_name = []
    resultboxes = []
    for i, name in enumerate(names):
        name = remove_numbers(name)
        result_name.append('a ' + name.replace('_',' '))
        if name == 'person': 
            boxes[i] = boxes[i]
        resultboxes.append([boxes[i]])
    return result_name, np.array(resultboxes)

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d

def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att
