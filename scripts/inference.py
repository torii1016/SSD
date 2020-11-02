import argparse
import sys
import os
import numpy as np
import toml
from tqdm import tqdm
import pickle
from collections import OrderedDict
from datetime import datetime

import cv2
import tensorflow as tf

from model.ssd import SSD
from utils.voc2007_dataset_loader import VOC2007Dataset

class Inferencer(object):
    def __init__(self, config, data_loader=None):
        ssd_config = toml.load(open(config["network"]["ssd_config"]))

        self._use_gpu = config["train"]["use_gpu"]
        self._saved_model_path = config["train"]["save_model_path"]
        self._saved_model_name = config["train"]["save_model_name"]
    
        self._data_loader = data_loader
        self._label_name = self._data_loader.get_label_name()
        self._image_width, self._image_height, self._images_channel = self._data_loader.get_image_info()

        self._ssd = SSD(config["train"], ssd_config)
        self._ssd.set_model()

        if self._use_gpu:
            config = tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=0.8,
                    allow_growth=True
                )
            )
        else:
            config = tf.compat.v1.ConfigProto(
                device_count = {'GPU': 0}
            )

        self._sess = tf.compat.v1.Session(config=config)
        self._saver = tf.compat.v1.train.Saver()
        self._saver.restore(self._sess, self._saved_model_path + "/" + self._saved_model_name)


    def _save_result(self, image, locs, labels, num):
        image *= 255.
        image = np.clip(image, 0, 255).astype('uint8')
        image = cv2.resize(image, (self._image_width, self._image_height))
        if len(labels) and len(locs):
            for label, loc in zip(labels, locs):
                loc = np.array([loc[0], loc[1], loc[2], loc[3]])

                cv2.rectangle(image, 
                                (int(loc[0]*self._image_width), int(loc[1]*self._image_height)), 
                                (int(loc[2]*self._image_width), int(loc[3]*self._image_height)), 
                                (0, 0, 255), 1)

                cv2.putText(image, 
                            str(self._label_name[int(label)]), 
                            (int(loc[0]*self._image_width), int(loc[1]*self._image_height)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
        cv2.imwrite( "./result/test_{}.jpg".format(num), image )
        

    def inference(self):

        num = 0
        input_images, input_labels = self._data_loader.get_test_data()
        for image, label in zip(input_images, input_labels):
            pred_confs, pred_locs = self._ssd.inference(self._sess, image)
            locs, labels = self._ssd.detect_objects(pred_confs, pred_locs)

            self._save_result(image, locs, labels, num)
            num += 1
            return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/testing_param.toml", type=str, help="default: config/testing_param.toml")
    args = parser.parse_args()

    data_loader = VOC2007Dataset(toml.load(open(args.config)))

    inferencer = Inferencer(toml.load(open(args.config)), data_loader)
    inferencer.inference()