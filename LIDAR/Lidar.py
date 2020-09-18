import os
import cv2
import time
import numpy as np
from function import show_img, find_files, cal_proj_matrix, load_img, load_lidar, project_lidar2img, generate_FV
from pca_norm import PCA, get_pca_o3d, get_surface_normals, get_surface_normals_o3d, norm_create
from tqdm import tqdm
import argparse


def main():
    # parsing arguments
    args = argparse.ArgumentParser(description='Lidar2image')
    args.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    args.add_argument('--out_path_Lidar2img', required=True, help='path of image output')
    args.add_argument('--out_path_Lidar2FV', required=True, help='path of frontview output')
    args.add_argument('--out_path_Surface_normal', required=True, help='path of surface normals output')

    # argparser.add_argument('--type', required=True, help='Type of Lidar representation.choose: proj, FV, etc')
    args.data_path='/home/lx/program/LORD-Lidar-Orientated-Road-Detection/dataset/'
    args.out_path_Lidar2img='/home/lx/program/LORD-Lidar-Orientated-Road-Detection/outputs/Lidar2img/'
    args.out_path_Lidar2FV = '/home/lx/program/LORD-Lidar-Orientated-Road-Detection/outputs/Lidar2FV/'
    args.out_path_Surface_normal='/home/lx/program/LORD-Lidar-Orientated-Road-Detection/SURFACE/'
    
    #############defaut path refer##############
    # --data_path
    # .../Road-Detection-and-Vehicle-Control/RD/dataset/
    # --out_path_Lidar2Proj
    # .../Road-Detection-and-Vehicle-Control/LIDAR/outputs/Lidar2Proj/
    # --out_path_Lidar2FV
    # .../Road-Detection-and-Vehicle-Control/LIDAR/outputs/Lidar2FV/
    if not os.path.exists(args.out_path_Lidar2img):
        os.makedirs(args.out_path_Lidar2img)
    if not os.path.exists(args.out_path_Lidar2FV):
        os.makedirs(args.out_path_Lidar2FV)
    print('- Original Lidar Sources are from: %s' % args.data_path)
    print('- Lidar2image results will be saved at: %s' % args.out_path_Lidar2img)
    print('- Lidar2FV results will be saved at: %s' % args.out_path_Lidar2FV)
    print('- Surface Normals results will be saved at: %s' % args.out_path_Surface_normal)
    # Calib File
    CALIB = args.data_path+"training/calib/"
    # # ################# PARAMETER ####################
    CAM_ID = 2
    flag = 'xyz'
    # Source File
    IMG_PATH = args.data_path+"training/image_2/"
    LIDAR_PATH = args.data_path+"training/velodyne/"
    # Save File
    SIMG_PATH = args.out_path_Lidar2img
    SNORM_PATH=args.out_path_Surface_normal

    # Batch Process
    #-----------------------------------IMG_Process------------------------------------------------------------
    # time_cost = []
    # for img_path in tqdm(find_files(IMG_PATH, '*.png')):
    #     _, img_name = os.path.split(img_path)
    #     pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
    #     calib_path = CALIB + img_name[:-4] + '.txt'
    # # print ("Working on", img_name[:-4])
    #     start_time = time.time()
    # #
    # #     # Load img & pc
    #     img = load_img(img_path)
    #     pc = load_lidar(pc_path)
    #     pc_distance = pc.copy()
    #     if flag == 'x':
    # 	    pass
    #     elif flag == 'xy':
    #         pc_distance[..., 0] = np.sqrt(pc[..., 0] ** 2 + pc[..., 1] ** 2)
    #     elif flag == 'xyz':
    #         pc_distance[..., 0] = np.sqrt(pc[..., 0] ** 2 + pc[..., 1] ** 2 + pc[..., 2] ** 2)
    #
    #     # Project & Generate Image & Save
    #     p_matrix = cal_proj_matrix(calib_path, CAM_ID)
    #     points = project_lidar2img(img, pc_distance, p_matrix)
    #
    #     pcimg = img.copy()
    #     depth_max = np.max(pc_distance[:, 0])
    #     for idx, i in enumerate(points):
    #         color = int((pc_distance[idx, 0] / depth_max) * 255)
    #         # cv2.rectangle(pcimg, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
    #         cv2.circle(pcimg, (int(i[0]), int(i[1])), 1, (0, 0, color), -1)
    #     cv2.imwrite(SIMG_PATH + img_name, pcimg)
    #     end_time = time.time()
    #     time_cost.append(end_time - start_time)
    #
    # print("Mean_time_cost:", np.mean(time_cost))
    # cv2.destroyAllWindows()

    # -----------------------FV Process---------------------------------------------------------------------------
    # time_cost=[]
    # HRES = 0.35  # horizontal resolution (assuming 20Hz setting)
    # VRES = 0.4  # vertical res
    # VFOV = (-24.9, 2.0)  # Field of view (-ve, +ve) along vertical axis
    # val = 'depth'
    # cmap = 'jet'
    # Y_FUDGE = 5  # y fudge factor for velodyne HDL 64E
    # for img_path in tqdm(find_files(IMG_PATH, '*.png')):
    #     _, img_name = os.path.split(img_path)
    #     pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
    #     # print ("Working on", img_name[:-4])
    #     start_time = time.time()
    #     savepath = args.out_path_Lidar2FV + img_name
    #     path_temp = args.data_path + 'training/velodyne/' + img_name[:-4] + '.bin'
    #     pointcloud = np.fromfile(path_temp, dtype=np.float32, count=-1).reshape([-1, 4])
    #     generate_FV(pointcloud, VRES, HRES, VFOV, val, cmap, savepath, Y_FUDGE)
    #     end_time = time.time()
    #     time_cost.append(end_time - start_time)
    #
    #
    # print("Mean_time_cost:", np.mean(time_cost))
    
    # -----------------------Surface normals Process--------------------------------------------------------------
    time_cost=[]

    for img_path in tqdm(find_files(IMG_PATH, '*.png')):
        _, img_name = os.path.split(img_path)
        pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
        calib_path = CALIB + img_name[:-4] + '.txt'
        # print ("Working on", img_name[:-4])
        start_time=time.time()
        img_template=cv2.imread(img_path)
        (r,c,num)=img_template.shape
        zeros = np.zeros([r,c])
        cv2.imwrite("./1.png", zeros)
        img = cv2.imread("./1.png")
        img1 = img.copy()
        norm_create(pc_path)
        rawdata=np.load("./surface_norm.npy").reshape([6,-1])
        lidar_points=rawdata[:3,:]
        v_max=rawdata.max(axis=1)
        v_min=rawdata.min(axis=1)
        range_x=v_max[3]-v_min[3]
        range_y = v_max[4] - v_min[4]
        range_z = v_max[5] - v_min[5]
        pc = lidar_points.T
        pc_distance = pc.copy()

        p_matrix = cal_proj_matrix(calib_path, CAM_ID)
        points = project_lidar2img(img1, pc_distance, p_matrix)

        pcimg = img1.copy()
        depth_max = np.max(pc_distance[:, 0])
        normal_vector=rawdata.T
        for idx, i in enumerate(points):
            color1 = int(((normal_vector[idx, 3]-v_min[3]) / range_x) * 255)
            color2 = int(((normal_vector[idx, 4] - v_min[4]) / range_y) * 255)
            color3 = int(((normal_vector[idx, 5] - v_min[5]) / range_z) * 255)
    # cv2.rectangle(pcimg, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
            if(int(i[1])<r and int(i[0])<c and int(i[1])>=0 and int(i[0])>=0):
                pcimg[int(i[1]),int(i[0]),0]=color1
                pcimg[int(i[1]),int(i[0]),1]=color2
                pcimg[int(i[1]),int(i[0]),2]=color3
                cv2.circle(img_template, (int(i[0]), int(i[1])), 1, (color1, color2, color3), -1)

        cv2.imwrite(SNORM_PATH + 'outputs/' + img_name, pcimg)
        img_gray=cv2.cvtColor(pcimg,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(SNORM_PATH + 'outputs_gray/' + img_name, img_gray)
        cv2.imwrite(SNORM_PATH + 'outputs_proj/' + img_name, img_template)
        end_time = time.time()
        time_cost.append(end_time - start_time)
    print("Mean_time_cost:", np.mean(time_cost))
    cv2.destroyAllWindows()
    print('all samples were processed')
if __name__ == "__main__":
    main()
