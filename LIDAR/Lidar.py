import os
import cv2
import time
import numpy as np
from function import find_files, cal_proj_matrix, load_img, load_lidar, project_lidar2img, generate_FV
from tqdm import tqdm
import argparse
import argparse as args


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Lidar2image')
    argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--out_path_Lidar2Proj', required=True, help='path of Proj image output')
    argparser.add_argument('--out_path_Lidar2FV', required=True, help='path of frontview output')
    # argparser.add_argument('--type', required=True, help='Type of Lidar representation.choose: proj, FV, etc')
    args = argparser.parse_args()
    #############defaut path refer##############
    # --data_path
    # .../Road-Detection-and-Vehicle-Control/RD/dataset/
    # --out_path_Lidar2Proj
    # .../Road-Detection-and-Vehicle-Control/LIDAR/outputs/Lidar2Proj/
    # --out_path_Lidar2FV
    # .../Road-Detection-and-Vehicle-Control/LIDAR/outputs/Lidar2FV/
    if not os.path.exists(args.out_path_Lidar2Proj):
        os.makedirs(args.out_path_Lidar2Proj)
    if not os.path.exists(args.out_path_Lidar2FV):
        os.makedirs(args.out_path_Lidar2Proj)
    print('- Original Lidar Sources are from: %s' % args.data_path)
    print('- Lidar2image results will be saved at: %s' % args.out_path_Lidar2Proj)
    print('- Lidar2FV results will be saved at: %s' % args.out_path_Lidar2FV)
    # Calib File
    CALIB = args.data_path+"training/calib/"
    ################# PARAMETER ####################
    CAM_ID = 2
    flag = 'xyz'
    # Source File
    IMG_PATH = args.data_path+"training/image_2/"
    LIDAR_PATH = args.data_path+"training/velodyne/"
    # Save File
    SIMG_PATH = args.out_path_Lidar2Proj
    # Batch Process
    #-----------------------------------IMG_Process------------------------------------------------------------
    time_cost = []
    for img_path in tqdm(find_files(IMG_PATH, '*.png')):
        _, img_name = os.path.split(img_path)
        pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
        calib_path = CALIB + img_name[:-4] + '.txt'
        # print ("Working on", img_name[:-4])
        start_time = time.time()

        # Load img & pc
        img = load_img(img_path)
        pc = load_lidar(pc_path)
        pc_distance = pc.copy()
        if flag == 'x':
            pass
        elif flag == 'xy':
            pc_distance[..., 0] = np.sqrt(pc[..., 0] ** 2 + pc[..., 1] ** 2)
        elif flag == 'xyz':
            pc_distance[..., 0] = np.sqrt(pc[..., 0] ** 2 + pc[..., 1] ** 2 + pc[..., 2] ** 2)

        # Project & Generate Image & Save
        p_matrix = cal_proj_matrix(calib_path, CAM_ID)
        points = project_lidar2img(img, pc_distance, p_matrix)

        pcimg = img.copy()
        depth_max = np.max(pc_distance[:, 0])
        for idx, i in enumerate(points):
            color = int((pc_distance[idx, 0] / depth_max) * 255)
            # cv2.rectangle(pcimg, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
            cv2.circle(pcimg, (int(i[0]), int(i[1])), 1, (0, 0, color), -1)
        cv2.imwrite(SIMG_PATH + img_name, pcimg)
        end_time = time.time()
        time_cost.append(end_time - start_time)

    print("Mean_time_cost:", np.mean(time_cost))
    cv2.destroyAllWindows()
    # -----------------------FV Process---------------------------------------------------------------------------
    for img_path in tqdm(find_files(IMG_PATH, '*.png')):
        _, img_name = os.path.split(img_path)
        pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
        # print ("Working on", img_name[:-4])
        start_time = time.time()

        HRES = 0.35  # horizontal resolution (assuming 20Hz setting)
        VRES = 0.4  # vertical res
        VFOV = (-24.9, 2.0)  # Field of view (-ve, +ve) along vertical axis
        val = 'depth'
        cmap = 'jet'
        savepath = args.out_path_Lidar2FV + img_name
        Y_FUDGE = 5  # y fudge factor for velodyne HDL 64E
        path_temp = args.data_path + 'training/velodyne/' + img_name[:-4] + '.bin'
        pointcloud = np.fromfile(path_temp, dtype=np.float32, count=-1).reshape([-1, 4])
        generate_FV(pointcloud, VRES, HRES, VFOV, val, cmap, savepath, Y_FUDGE)
        end_time = time.time()
        time_cost.append(end_time - start_time)


    print("Mean_time_cost:", np.mean(time_cost))
    print('all samples were processed')

if __name__ == "__main__":
    main()
