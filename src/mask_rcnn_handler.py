import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pickle

# TODO: ROOT_DIR to be changed
# Root directory of the project
ROOT_DIR =  "../"   #os.path.abspath( "../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# print(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  # To find local version
import coco
# # matplotlib inline 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file from KITTI fine-tuned result
COCO_MODEL_PATH = ROOT_DIR+"../model/maskrcnn_model/mask_rcnn_kitti.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskPredictClass:
    def __init__(self):
        # initialize mask RCNN
        config = InferenceConfig()
        # config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    def predict(self, filename = '../images/data/left_img/000056_10.png', is_savep = False, is_show = False, res_dir = '../images/res/'):
        print("predict mask for {}".format(filename))
        image = skimage.io.imread(filename)

        # Run detection
        results = self.model.detect([image], verbose=1)

        # Visualize results
        r = results[0]

        if(is_savep):
            image_idx = (filename.split('.')[-2]).split('/')[-1]
            p_name = res_dir + image_idx + '.p'
            print("Save .p file for image: "+image_idx)
            pickle.dump(r, open(p_name,"wb"))
        
        # image.savefig('demo.jpg')
        if(is_show):
            matplotlib.use('tkAgg')
            print(matplotlib.get_backend())

            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        self.class_names, r['scores'])
        
        mask_result = []
        for i in range(len(r['scores'])):
            temp_type = self.class_names[r['class_ids'][i]]
            if(temp_type == 'car' or temp_type == 'truck' or temp_type == 'bus'\
                or temp_type == 'train'):
                mask_result.append(r['masks'][:, :, i])
        return np.stack(mask_result, axis = 2)














