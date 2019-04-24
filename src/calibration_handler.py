import numpy as np

class CalibrationClass:
    def __init__(self, in_calib_dir = '../images/data/calibration/'):
        self.width_to_focal = dict()

        # camera parameters (from: data_scene_flow/data_sense_flow_calib/training)
        self.width_to_focal[1242] = 721.5377
        self.width_to_focal[1241] = 718.856
        self.width_to_focal[1224] = 707.0493
        self.width_to_focal[1238] = 718.3351
        self.P_rect = {}
        self.calib_dir = in_calib_dir
        
    def getP(self, img_idx):
        if img_idx not in self.P_rect:
            f_read = open(self.calib_dir+img_idx.split('_')[0]+".txt", "r")
            last_line = f_read.readlines()[-1]
            f_read.close()
            P_str = last_line.split(" ")[1:]
            P_in = [float(i) for i in P_str]
            #print(P_in)
            self.P_rect[img_idx] = np.array(P_in).reshape(3, 4)
        return self.P_rect[img_idx]