# Road Detection and Vehicle Control
#### Haolin ZHANG, Xiao LIU, Ruoyang FU, Simeng HUANG, Hongwei JIANG
#### Department of Electrical and Computer Engineering, The Ohio State University
##### Latest version:08/11/2020

#### Dataset Setup

Please setup dataset according to the following folder structure:
```
RD
 |---- ptsemseg
 |---- outputs
 |---- configs
 |---- dataset
 |    |---- training
 |    |    |---- image_2
 |    |    |---- gt_image_2
 |    |    |---- velodyne
 |    |    |---- calib
 |    |---- testing
 |    |    |---- image_2
 |    |    |---- velodyne
 |    |    |---- calib
```

#### Update Log

##### 06/06/2020
* Revised evaluation tools (in `devkit_road` (from http://www.cvlibs.net/datasets/kitti/eval_road.php))

##### 06/10/2020
* Upload segmentation models (in `RD`)

##### 06/12/2020
* Upload Lidar2image tools (in `LIDAR`)

##### 08/11/2020
* Finish Lidar2image tools (in `LIDAR`)

