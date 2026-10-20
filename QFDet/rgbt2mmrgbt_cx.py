import os
import shutil
import glob
import json

# 1 图像改名
# image_path = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val/thermal/*.jpg'
# image_savepath =  '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val/thermal0/'
# image_list = glob.glob(image_path)
# for i,image in enumerate(image_list):
#     image_name = os.path.basename(image)
#     split_name = image_name.split('_')
#     image_name0 = split_name[0]+'_'+split_name[1]+'_'+split_name[2]+'_'+split_name[4]
#     image_newname = image_savepath + image_name0
#     shutil.copy(image, image_newname)
#     if i%1000==0:
#         print('{}/{}'.format(i,len(image_list)))
# print('finished')

# 2 json更改
json_ori_path = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_tiny_origin/annotations_cx_new/instances_00_train2017.json'
json_save_path = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_tiny_origin/annotations_cx_new/instances_00_train2017_new.json'

with open(json_ori_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
images_wlist=[]
annotations_wlist=[]
categories_wlist=[]

for i, text in enumerate(data['images']):
    # text_oriname = text['file_name']
    # split_name = text_oriname.split('/')
    text['width']=640
    text['height']=512
    # text['file_name'] = split_name[0]+'_'+split_name[2]
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