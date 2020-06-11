import os
import shutil

file_path = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/train_data/ground_truth/"
image_path = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/train_data/images/"

save_file_path = "/home/chenli/work/crowd_counting/dataset/ours/train_data/ground_truth/"
save_image_path = "/home/chenli/work/crowd_counting/dataset/ours/train_data/images/"

file_names = os.listdir(file_path)
for i, file_name in enumerate(file_names):
    if os.path.exists(file_path + file_name):
        shutil.copy(file_path + file_name, save_file_path+ file_name)
        shutil.copy(image_path + file_name[:-4]+".jpg", save_image_path+file_name[:-4]+".jpg")
