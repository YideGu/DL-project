# DL-project
Course 598 deep learning project

# Reference:
The stable monocular depth esimation code (in `.\monodepth\`) is from [`https://github.com/mrharicot/monodepth.git`
](https://github.com/mrharicot/monodepth.git), which was finally used for bounding box prediction. Our OWN implementation for 
monocular depth esimation is in (in `.\monodepth2\`)  
The Mask-RCNN code (in `.\Mask_RCNN\`) is from [`https://github.com/matterport/Mask_RCNN.git`](https://github.com/matterport/Mask_RCNN.git), and fine-tuned on KITTI by us.

# Installation:
```
git clone https://github.com/WeilinXu/DL-project.git
cd DL-project
```
Add pre-trained model for monodepth:
```
sh ./monodepth/utils/get_model.sh model_kitti ../model/monodepth_model
```
Add pre-trained model for Mask-RCNN:
Download `mask_rcnn_coco.h5` to `../model/maskecnn_model/` from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

Add pre-trained model for disparityToDepth network:
Download `trained_model_cnn.pth` to `../model/d2z_model/` from [here](https://drive.google.com/open?id=1Q96jg1m1AYChdF6OBT1pm3HMIE8kl_vI)

# Test:
The input left image file (eg. `000056_10.png`) should be put in `DL-project/images/data/left_img`.
The corresponding calibration file (eg. `000056.txt`) should be put in `DL-project/images/data/calibration`.
Generate depth image estimation data (eg. `000056_10_disp.npy`) and depth image (eg. `000056_10_disp_pred.png`) in `DL-project/images/res` and plot 3D bounding box:
```
python src/demo2.py
```
Alternatively, generate depth image estimation data (eg. `000056_10_disp.npy`) and depth image (eg. `000056_10_disp_pred.png`) in `DL-project/images/res` (not plot 3D bounding box):
```
python ./monodepth/monodepth_simple.py --image_path images/data/left_img/000056_10.png --checkpoint_path ../model/monodepth_model/model_kitti
```

