import argparse
import numpy as np
import pickle

import cv2
from utils.dataset_loader import Dataset

class VOC2007Dataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._dataset_image_path = config["image"]
        self._dataset_label_path = config["label"]

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