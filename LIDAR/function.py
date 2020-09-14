import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import fnmatch
from tqdm import tqdm
from pprint import pprint


# **********************************************************#
#                    Basic Function                        #
# **********************************************************#

def show_img(name, img):
    """
    Show the image

    Parameters:
        name: name of window
        img: image
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(0)


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


# **********************************************************#
#                     Load Function                        #
# **********************************************************#

def load_calib_cam2cam(filename, debug=False):
    """
    Only load R_rect & P_rect for need
    Parameters: filename of the calib file
    Return:
        R_rect: a list of r_rect(shape:3*3)
        P_rect: a list of p_rect(shape:3*4)
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

    R_rect = []
    P_rect = []

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-4] == "R_rect":
            r_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r_r = np.reshape(r_r, (3, 3))
            R_rect.append(r_r)
        elif title[:-4] == "P_rect":
            p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            p_r = np.reshape(p_r, (3, 4))
            P_rect.append(p_r)

    if debug:
        print("R_rect:")
        pprint(R_rect)

        print()
        print("P_rect:")
        pprint(P_rect)

    return R_rect, P_rect


def load_calib_lidar2cam(filename, debug=False):
    """
    Load calib parameters for LiDAR2Cam
    Parameters: filename of the calib file
    Return:
        tr: shape(4*4)
            [  r   t
             0 0 0 1]
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-1] == "R":
            r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r = np.reshape(r, (3, 3))
        if title[:-1] == "T":
            t = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            t = np.reshape(t, (3, 1))

    tr = np.hstack([r, t])
    tr = np.vstack([tr, np.array([0, 0, 0, 1])])

    if debug:
        print()
        print("Tr:")
        print(tr)

    return tr


def load_calib(filename, debug=False):
    """
    Load the calib parameters which has R_rect & P_rect & Tr in the same file
    Parameters:
        filename: the filename of the calib file
    Return:
        R_rect, P_rect, Tr
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

        P_rect = []
    for line in lines:
        title = line.strip().split(' ')[0]
        if len(title):
            if title[0] == "R":
                R_rect = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                R_rect = np.reshape(R_rect, (3, 3))
            elif title[0] == "P":
                p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                p_r = np.reshape(p_r, (3, 4))
                P_rect.append(p_r)
            elif title[:-1] == "Tr_velo_to_cam":
                Tr = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                Tr = np.reshape(Tr, (3, 4))
                Tr = np.vstack([Tr, np.array([0, 0, 0, 1])])

    return R_rect, P_rect, Tr


def load_img(filename, debug=False):
    """
    Load the image
    Parameter:
        filename: the filename of the image
    Return:
        img: image
    """
    img = cv2.imread(filename)

    if debug: show_img("Image", img)

    return img


def load_lidar(filename, debug=False):
    """
    Load the PointCloud
    Parameter:
        filename: the filename of the PointCloud
    Return:
        points: PointCloud associated with the image
    """
    # N*4 -> N*3
    points = np.fromfile(filename, dtype=np.float32)
    points = np.reshape(points, (-1, 4))
    points = points[:, :3]
    points.tofile("./temp_pc.bin")

    # Remove all points behind image plane (approximation)
    cloud = PyntCloud.from_file("./temp_pc.bin")
    cloud.points = cloud.points[cloud.points["x"] >= 0]
    points = np.array(cloud.points)

    if debug:
        print(points.shape)

    return points


# **********************************************************#
#                   Process Function                       #
# **********************************************************#
def cal_proj_matrix(filename, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Image
    Parameters:
        filename: filename of the calib file
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Image
    """
    # Load Calib Parameters
    R_rect, P_rect, tr = load_calib(filename, debug)

    # Calculation
    R_cam2rect = np.hstack([np.array([[0], [0], [0]]), R_rect])
    R_cam2rect = np.vstack([np.array([1, 0, 0, 0]), R_cam2rect])

    P_lidar2img = np.matmul(P_rect[camera_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print()
        print("P_lidar2img:")
        print(P_lidar2img)

    return P_lidar2img


def project_lidar2img(img, pc, p_matrix, debug=False):
    """
    Project the LiDAR PointCloud to Image
    Parameters:
        img: Image
        pc: PointCloud
        p_matrix: projection matrix
    """
    # Dimension of data & projection matrix
    dim_norm = p_matrix.shape[0]
    dim_proj = p_matrix.shape[1]

    # Do transformation in homogenuous coordinates
    pc_temp = pc.copy()
    if pc_temp.shape[1] < dim_proj:
        pc_temp = np.hstack([pc_temp, np.ones((pc_temp.shape[0], 1))])
    points = np.matmul(p_matrix, pc_temp.T)
    points = points.T

    temp = np.reshape(points[:, dim_norm - 1], (-1, 1))
    points = points[:, :dim_norm] / (np.matmul(temp, np.ones([1, dim_norm])))

    # Plot
    if debug:
        depth_max = np.max(pc[:, 0])
        for idx, i in enumerate(points):
            color = int((pc[idx, 0] / depth_max) * 255)
            # cv2.rectangle(img, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
            cv2.circle(img, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (0, 0, color), -1)
        show_img("Test", img)

    return points

def generate_FV(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    # x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    # x_img -= x_min              # Shift
    # x_max = 360.0 / h_res       # Theoretical max x value after shifting
    x_min = -180.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 180.0 / h_res
    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution

    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.axis('off')

    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV
    plt.style.use('dark_background')
    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()
    plt.close()



