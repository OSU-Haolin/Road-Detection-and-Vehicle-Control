import os
import time
import fnmatch
from tqdm import tqdm
import shutil


def find_files(directory, pattern):
    """
    Method to find target files in one directory, including subdirectory
    :param directory: path
    :param pattern: filter pattern
    :return: target file path list
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                file_list.append(filename)

    return file_list


project_path_source = '/home/lx/program/LORD-Lidar-Orientated-Road-Detection/dataset/training/'
project_path_destination = '/home/lx/program/LORD-Lidar-Orientated-Road-Detection/local/'
source_type_name = {
    'image': "image_2/",
    'truth_img': "gt_image_2/",
    'ADT': "ADI/",
    'FrontView': "Lidar2FV/",
    'Lidar_proj': "image_2_proj/",
    'surface_normals': "image_2_sn"
}
destination_type_name = {
    'image': "image_2/",
    'truth_img': "gt_image_2/",
    'ADT': "ADI/",
    'Lidar_proj': "img_proj/",
    'surface_normals': "surface_normals/"
}


def copy_file(source, destination):
    for img_path in tqdm(find_files(project_path_source + source, '*.png')):
        _, img_name = os.path.split(img_path)
        temp_name = img_name[-10:-4]
        temp_num = int(temp_name) % 2
        if temp_num == 0:
            shutil.copy(img_path, project_path_destination + 'train/' + destination + img_name)
        elif temp_num == 1:
            shutil.copy(img_path, project_path_destination + 'val/' + destination + img_name)


def main():
    # copy_file(source_type_name['image'], destination_type_name['image'])
    # copy_file(source_type_name['truth_img'], destination_type_name['truth_img'])
    # copy_file(source_type_name['ADT'], destination_type_name['ADT'])
    # copy_file(source_type_name['Lidar_proj'], destination_type_name['Lidar_proj'])
    copy_file(source_type_name['surface_normals'], destination_type_name['surface_normals'])


if __name__ == "__main__":
    main()
