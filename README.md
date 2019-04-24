# DL-project
Course 598 deep learning project

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

# Test:
The input left image file (eg. `000056_10.png`) should be put in `DL-project/images/data/left_img`.
The corresponding calibration file (eg. `000056.txt`) should be put in `DL-project/images/data/calibration`.
Generate depth image estimation data (eg. `000056_10_disp.npy`) and depth image (eg. `000056_10_disp_pred.png`) in `DL-project/images/res`:
```
python ./monodepth/monodepth_simple.py --image_path images/data/left_img/000056_10.png --checkpoint_path ../model/monodepth_model/model_kitti
```
Generate bounding box:
```
python src/demo2.py
```
