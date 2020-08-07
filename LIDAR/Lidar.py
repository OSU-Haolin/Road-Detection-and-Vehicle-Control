import os
import time
import numpy as np
from Lidar.function import lidar2image
import argparse


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Lidar2image')
    argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--out_path', required=True, help='path of output')
    argparser.add_argument('--type', required=True, help='Type of Lidar representation.choose: proj, map, etc')
    args = argparser.parse_args()

    print('- Original Lidar Sources are from: %s' % args.data_path)
    print('- Lidar2image results will be saved at: %s' % args.out_path)

    names_sample = []
    for file in os.listdir(os.path.join(args.data_path,'training','velodyne')):
        if file[-4:] == '.bin':
            names_sample.append(file[:-4])
    # loop each sample
    for i, sample_name in enumerate(names_sample):
        try:
            print('==> running sample ' + sample_name + ', index=%d' % i)
            # type = 'proj' (lidar projection) / type = 'map' (lidar mapping)
            done = lidar2image(sample_name=sample_name,data_path=args.data_path,out_path=args.out_path,type=args.type)
            if done:
                print('==> finish sample ' + sample_name + ', index=%d' % i)
        except:
            print('all samples were processed')

if __name__ == "__main__":
    main()
