import os 
import sys

from utils.voc2007_dataset_loader import VOC2007Dataset

class DatasetMaker(object):
    def __init__(self):
        self._dataset = {"VOC2007":VOC2007Dataset}
    
    def __call__(self, mode, config):
        return self._dataset[mode](config)