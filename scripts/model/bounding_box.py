import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

import cv2

class BoundingBox(object):
    def __init__(self, label, rect_loc):
        self._label = label
        self._rect_loc = rect_loc