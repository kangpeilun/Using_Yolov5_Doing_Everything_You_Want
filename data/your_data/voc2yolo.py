# coding:utf-8

from __future__ import print_function

import os
import random
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import platform
import shutil


'''
    将voc数据集格式转换为yolo格式
'''

def check_dir(path):
    '''检查是否缺少文件夹'''
    if not os.path.exists(path):
        os.mkdir(path)


def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(filename):
    classes_dict = {}
    # insect.names
    # 切记这个要和你的.names文件同名
    with open("your_data.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[class_name] = idx

    width, height, objects = xml_reader(filename)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2 + x) * 0.5 / width
        cy = (y2 + y) * 0.5 / height
        w = (x2 - x) * 1. / width
        h = (y2 - y) * 1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)


    check_dir('labels')
    txt_name = filename.replace(".xml", ".txt").replace("Annotations", "labels")
    with open(txt_name, "w") as f:
        f.writelines(lines)


def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG', 'png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename) + "\n"
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist


def imglist2file(imglist, val_split=0.1):
    # val_split 将百分比数据作为验证集，默认将 10% 的数据集作为 验证集
    length = len(imglist)
    random.seed(10101)
    random.shuffle(imglist)
    random.seed(None)
    val_split = val_split
    train_num = length-int(length*val_split)
    train_list = imglist[:train_num]
    valid_list = imglist[train_num:]

    print('num of train_list',len(train_list))
    print('num of valid_list',len(valid_list))
    
    check_dir('train')
    check_dir('./train/images')
    check_dir('./train/labels')
    
    check_dir('valid')
    check_dir('./valid/images')
    check_dir('./valid/labels')

    for train_file in tqdm(train_list):
        if platform.system()=='Windows':
            dir, file_name = train_file.replace('\n','').split('\\')
            train_file =  train_file.replace('\n','')
            label_path = os.path.join('.\\', 'labels', file_name.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                shutil.copy(train_file, '.\\train\\images')
                shutil.copy(label_path, '.\\train\\labels')

        elif platform.system()=='Linux':
            _, dir, file_name = train_file.replace('\n','').split('/')
            train_file =  train_file.replace('\n','')
            label_path = os.path.join('./', 'labels', file_name.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                shutil.copy(train_file, './train/images')
                shutil.copy(label_path, './train/labels')

    print('\n训练集划分完毕！')


    for valid_file in tqdm(valid_list):
        if platform.system()=='Windows':
            dir, file_name = valid_file.replace('\n','').split('\\')
            valid_file =  valid_file.replace('\n','')
            label_path = os.path.join('.\\', 'labels', file_name.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                shutil.copy(valid_file, '.\\valid\\images')
                shutil.copy(label_path, '.\\valid\\labels')

        elif platform.system()=='Linux':
            _, dir, file_name = valid_file.replace('\n','').split('/')
            label_path = os.path.join('./', 'labels', file_name.replace('.jpg', '.txt'))
            valid_file =  valid_file.replace('\n','')
            if os.path.exists(label_path):
                shutil.copy(valid_file, './valid/images')
                shutil.copy(label_path, './valid/labels')
    print('\n验证集划分完毕！')


if __name__ == "__main__":
    # voc数据集中的xml文件夹
    xml_path_list = glob.glob("./Annotations/*.xml")
    for xml_path in tqdm(xml_path_list):
        voc2yolo(xml_path)

    # xml文件对应图片文件夹
    imglist = get_image_list("./JPEGImages")
    imglist2file(imglist)