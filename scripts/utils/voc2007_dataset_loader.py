import argparse
import numpy as np
import pickle

import cv2

class VOC2007Dataset(object):
    def __init__(self, config):
        self._dataset_image_path = config["dataset"]["image"]
        self._dataset_label_path = config["dataset"]["label"]
        self._test_data_num = config["dataset"]["test_data_num"]
        self._label_name = config["dataset"]["label_name"]

        self._image_width = config["dataset"]["image_width"]
        self._image_height = config["dataset"]["image_height"]
        self._image_channel = config["dataset"]["image_channels"]

        self._load_dataset()


    def _loading_images(self):

        images = []
        labels = []
        for img_name in self.keys:
            img = self.loading_image(self._dataset_image_path+"/"+img_name)
            images.append(img)
            labels.append(self.label[img_name])
        
        return np.array(images), np.array(labels)


    def loading_image(self, path):
        img = cv2.imread(path)
        h, w, c = img.shape
        img = cv2.resize(img, (self._image_width, self._image_height))
        img = img[:, :, ::-1].astype("float32")
        img /= 255
        return img


    def loading_label(self, path):
        pass


    def _loading_pickle_label(self):
        with open(self._dataset_label_path, "rb") as file:
            self.label = pickle.load(file)
            self.keys = sorted(self.label)
    

    def _load_dataset(self):
        self._loading_pickle_label()
        self._images, self._labels = self._loading_images()

        self._test_images = self._images[-self._test_data_num:]
        self._test_labels = self._labels[-self._test_data_num:]
        self._train_images = self._images[:-self._test_data_num]
        self._train_labels = self._labels[:-self._test_data_num]


    def get_train_data(self, batch_size=50):
        choice_data = np.random.choice(range(len(self._train_images)), batch_size, replace=False)
        return self._train_images[choice_data], self._train_labels[choice_data]


    def get_test_data(self):
        return self._test_images, self._test_labels


    def get_label_name(self):
        return self._label_name
    
    def get_image_info(self):
        return self._image_width, self._image_height, self._image_channel