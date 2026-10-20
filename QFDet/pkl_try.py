import pickle

# 假设你有一个名为 'example.pkl' 的文件
file_path = '/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/test/out_00.pkl'
file_path2 = '/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/test/result.pkl'

# 使用with语句确保文件正确关闭
with open(file_path, 'rb') as file:
    # 加载.pkl文件内容
    data = pickle.load(file)
with open(file_path2, 'rb') as file:
    # 加载.pkl文件内容
    data2 = pickle.load(file)
# 现在 data 变量包含了.pkl文件的内容
# 你可以打印出来或者进行其他操作
print(data)
