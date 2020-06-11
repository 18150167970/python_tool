import os
import shutil

file_path = "/home/chenli/head_counts/head_data/head_crop5/0_background/"
save_file_path = "/home/chenli/head_counts/head_data/headcount/test/0_background/"

file_names = os.listdir(file_path)
for i, file_name in enumerate(file_names):
    # if i%10==0:
    shutil.move(file_path + file_name, save_file_path + file_name)
    print(save_file_path + file_name)
