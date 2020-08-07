import os
import numpy as np
def read_source():
    pass
def generate_proj():
    pass
def generate_map():
    pass

def lidar2image(sample_name,data_path,out_path,type='proj'):
    """
    run one sample(one lidar and calibration frame) to generate one Lidar2image representation.
    """

    # read Lidar,calibration
    read_source()

    #generate different representation according to different type
    if type=='proj':
        generate_proj()
        pass
    elif type=='map':
        generate_map()
        pass

    pass

    return True


