import os
import shutil
import glob
import json

json_ori = 'instances_00_train2017.json'
json_save = 'train00.json'
json_ori = 'instances_01_train2017.json'
json_save = 'train01.json'
json_ori = 'instances_00_test2017.json'
json_save = 'val00.json'
# json_ori = 'instances_01_test2017.json'
# json_save = 'val01.json'

json_ori_path = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_tiny_origin/annotations_cx_new/' + json_ori
json_save_path = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/' + json_save

with open(json_ori_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
images_wlist=[]
annotations_wlist=[]
categories_wlist=[]

for i, text in enumerate(data['images']):
    text_oriname = text['file_name']
    split_name = text_oriname.split('/')
    text['width']=640
    text['height']=512
    text['file_name'] = split_name[0]+'_'+split_name[2]
    images_wlist.append(text)

for i,text in enumerate(data['annotations']):
    annotations_wlist.append(text)
for i,text in enumerate(data['categories']):
    categories_wlist.append(text)
type_w = 'instances'

ann_js={}
ann_js['images']=images_wlist
ann_js['annotations']=annotations_wlist
ann_js['type']=type_w
ann_js['categories']=categories_wlist
json.dump(ann_js,open(json_save_path,'w'),indent=4)
print('finished')