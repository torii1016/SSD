import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

import cv2

class BoundingBox(object):
    def __init__(self, label, rect_loc):
        self._label = label
        self._rect_loc = rect_loc

    def draw_rect(self, image, color=(0,0,255), thickness=1):
        center_x = image.shape[0]*self._rect_loc[0]-0.5
        center_y = image.shape[1]*self._rect_loc[1]-0.5
        height = image.shape[1]*self._rect_loc[2]
        width = image.shape[0]*self._rect_loc[3]

        point1_x = int(center_x-width/2) 
        point1_y = int(center_y-height/2)
        point2_x = int(center_x+width/2)
        point2_y = int(center_y+height/2) 

        image = cv2.rectangle(img=image,
                              pt1=(point1_x, point1_y), 
                              pt2=(point2_x, point2_y), 
                              color=color, 
                              thickness=thickness)
        return image