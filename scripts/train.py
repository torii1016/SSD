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


class VOC2007Dataset(object):
    def __init__(self, config):
        self._dataset_image_path = config["dataset"]["image"]
        self._dataset_label_path = config["dataset"]["label"]
        self._test_data_num = config["dataset"]["test_data_num"]

        self._batch_size = config["train"]["batch_size"]

        self._load_dataset()


    def _loading_image(self):

        images = []
        labels = []
        for img_name in self.keys:
            img = cv2.imread(self._dataset_image_path+"/"+img_name)
            h, w, c = img.shape
            img = cv2.resize(img, (300, 300))
            img = img[:, :, ::-1].astype("float32")
            img /= 255

            images.append(img)
            labels.append(self.label[img_name])
        
        return np.array(images), np.array(labels)


    def _loading_label(self):
        with open(self._dataset_label_path, "rb") as file:
            self.label = pickle.load(file)
            self.keys = sorted(self.label)
    

    def _load_dataset(self):
        self._loading_label()
        self._images, self._labels = self._loading_image()

        self._test_images = self._images[-self._test_data_num:]
        self._test_labels = self._labels[-self._test_data_num:]
        self._train_images = self._images[:-self._test_data_num]
        self._train_labels = self._labels[:-self._test_data_num]


    def get_train_data(self):
        choice_data = np.random.choice(range(len(self._train_images)), self._batch_size, replace=False)
        return self._train_images[choice_data], self._train_labels[choice_data]


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
        init = tf.compat.v1.global_variables_initializer()
        self._sess.run(init)
        self._saver = tf.train.Saver()

        self._accuracy = 0.0

        self._tensorboard_path = "./logs/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


    def _save_model(self):
        os.makedirs(self._save_model_path, exist_ok=True)
        self._saver.save(self._sess, self._save_model_path+"/"+self._save_model_name)


    def _save_tensorboard(self, loss):
        with tf.name_scope('log'):
            tf.compat.v1.summary.scalar('loss', loss)
            merged = tf.compat.v1.summary.merge_all()
            writer = tf.compat.v1.summary.FileWriter(self._tensorboard_path, self._sess.graph)


    def train(self):
        with tqdm(range(self._epoch)) as pbar:
            for i, ch in enumerate(pbar): #train
                input_images, input_labels = self._data_loader.get_train_data()

                _, loss = self._ssd.train(self._sess, input_images, input_labels)
                pbar.set_postfix(OrderedDict(loss=loss, accuracy=self._accuracy))
                print("\n")

                self._save_tensorboard(loss)

                if i%self._val_step==0: #test
                    self._save_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()


    data_loader = VOC2007Dataset(toml.load(open("config/training_param.toml")))

    #trainer = Trainer(toml.load(open("config/training_param.toml")), None)
    trainer = Trainer(toml.load(open("config/training_param.toml")), data_loader)
    trainer.train()