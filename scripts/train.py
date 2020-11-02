import argparse
import sys
import os
import numpy as np
import toml
from tqdm import tqdm
import pickle
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf

from model.ssd import SSD
from utils.voc2007_dataset_loader import VOC2007Dataset


class Trainer(object):
    def __init__(self, config, data_loader=None):
        ssd_config = toml.load(open(config["network"]["ssd_config"]))

        self._batch_size = config["train"]["batch_size"]
        self._epoch = config["train"]["epoch"]
        self._val_step = config["train"]["val_step"]
        self._use_gpu = config["train"]["use_gpu"]
        self._save_model_path = config["train"]["save_model_path"]
        self._save_model_name = config["train"]["save_model_name"]
    
        self._data_loader = data_loader

        self._ssd = SSD(config["train"], ssd_config, self._data_loader.get_image_info())
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
        init = tf.compat.v1.global_variables_initializer()
        self._sess.run(init)
        self._saver = tf.compat.v1.train.Saver()

        self._accuracy = 0.0
        self._loss = []

        self._tensorboard_path = "./logs/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


    def _save_model(self):
        os.makedirs(self._save_model_path, exist_ok=True)
        self._saver.save(self._sess, self._save_model_path+"/"+self._save_model_name)


    def _save_log(self):
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(range(len(self._loss)), self._loss, color="b")
        ax.set_xlabel("episode")
        ax.set_ylabel("loss")
        ax.set_ylim(0, 200)
        ax.grid()
        plt.savefig("logs/loss.png")


    def _save_tensorboard(self, loss):
        with tf.name_scope('log'):
            tf.compat.v1.summary.scalar('loss', loss)
            merged = tf.compat.v1.summary.merge_all()
            writer = tf.compat.v1.summary.FileWriter(self._tensorboard_path, self._sess.graph)


    def train(self):
        with tqdm(range(self._epoch)) as pbar:
            for i, ch in enumerate(pbar): #train
                input_images, input_labels = self._data_loader.get_train_data(self._batch_size)

                _, loss = self._ssd.train(self._sess, input_images, input_labels)
                pbar.set_postfix(OrderedDict(loss=loss, accuracy=self._accuracy))

                #self._save_tensorboard(loss)
                self._loss.append(loss)

                if i%self._val_step==0: #test
                    self._save_model()
                    self._save_log()


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/training_param.toml", type=str, help="default: config/training_param.toml")
    args = parser.parse_args()

    data_loader = VOC2007Dataset(toml.load(open(args.config)))

    trainer = Trainer(toml.load(open(args.config)), data_loader)
    trainer.train()