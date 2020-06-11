import os

file_path = "/home/chenli/head_counts/head_data/head_crop5/0_background/"

# 同名导致会覆盖,文件会减少,如果命名为相同的,最好加上一个编号
file_names = os.listdir(file_path)
index = 15000
for i,file_name in enumerate(file_names):
    os.renames(file_path+file_name, file_path+str(i+index)+'.jpg')
    print(file_path+str(i+index)+'.jpg')