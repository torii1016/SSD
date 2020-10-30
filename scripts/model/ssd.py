# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, smooth_L1, SSDNetworkCreater, ExtraFeatureMapNetworkCreater
from .bbox_matcher import BBoxMatcher
from .default_box_generator import BoxGenerator

class _ssd_network(Layers):
    def __init__(self, name_scopes, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)

        self._ssd_config = config["SSD"]
        self._fmap_config = config["ExtraFmap"]
        self._ssd_network_creater = SSDNetworkCreater(config["SSD"], name_scopes[0]) 
        self._extra_feature_network_creater = ExtraFeatureMapNetworkCreater(config["ExtraFmap"], name_scopes[0]) 

    def set_model(self, inputs, is_training=True, reuse=False):
        _ = self._ssd_network_creater.create(inputs, self._ssd_config, is_training, reuse)
        extra_feature = self._ssd_network_creater.get_extra_feature()
        return self._extra_feature_network_creater.create(extra_feature, 
                                                          self._fmap_config, 
                                                          is_training, reuse)
    

class SSD(object):
    
    def __init__(self, param, config):

        self.batch_size = param["batch_size"]
        self.lr = param["lr"]
        self.output_dim = param["output_class"]
        self.image_width = param["image_width"]
        self.image_height = param["image_height"]
        self.image_channels = param["image_channels"]

        self.network = _ssd_network([config["SSD"]["network"]["name"]], config)
        
        self.box_generator = BoxGenerator(config["SSD"]["default_box"])

    def set_model(self):
        self.set_network()
        self.default_boxes = self.box_generator.generate_boxes(self.fmaps)
        self.set_loss()
        self.set_optimizer()

        self._matcher = BBoxMatcher(n_classes=self.output_dim, default_box_set=self.default_boxes)

    def set_network(self):
        
        # -- place holder ---
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channels])

        # -- set network ---
        self.fmaps, self.confs, self.locs = self.network.set_model(self.input, is_training=True, reuse=False)
        self.fmaps_wo, self.confs_wo, self.locs_wo = self.network.set_model(self.input, is_training=False, reuse=True)


    def set_loss(self):

        total_boxes = len(self.default_boxes)
        self.gt_labels_val = tf.compat.v1.placeholder(tf.int32, [None, total_boxes])
        self.gt_boxes_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes, 4])
        self.pos_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])
        self.neg_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])


        # Loss conf
        smoothL1_op = smooth_L1(self.gt_boxes_val-self.locs)
        loss_loc_op = tf.reduce_sum(smoothL1_op, reduction_indices=2)*self.pos_val
        loss_loc_op = tf.reduce_sum(loss_loc_op, reduction_indices=1)/(1e-5+tf.reduce_sum(self.pos_val, reduction_indices=1)) #average

        # Loss loc
        loss_conf_op = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                                            logits=self.confs, 
                                                            labels=self.gt_labels_val)
        loss_conf_op = loss_conf_op*(self.pos_val+self.neg_val)
        loss_conf_op = tf.reduce_sum(loss_conf_op, reduction_indices=1)/(1e-5+tf.reduce_sum((self.pos_val+self.neg_val), reduction_indices=1))

        # Loss
        self._loss_op = tf.reduce_sum(loss_conf_op+loss_loc_op)


    def set_optimizer(self):
        self._train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self._loss_op)


    def train(self, sess, input_images, input_labels):
    
        positives = []
        negatives = []
        ex_gt_labels = []
        ex_gt_boxes = []

        feed_dict = {self.input: input_images}
        _, confs, locs = sess.run([self.fmaps, self.confs, self.locs], feed_dict=feed_dict)

        for i in range(len(input_images)):
            actual_labels = []
            actual_loc_rects = []

            for obj in input_labels[i]:
                loc_rect = obj[:4]

                label = np.argmax(obj[4:])

                width = loc_rect[2]-loc_rect[0]
                height = loc_rect[3]-loc_rect[1]
                loc_rect = np.array([loc_rect[0], loc_rect[1], width, height])

                center_x = (2*loc_rect[0]+loc_rect[2])*0.5
                center_y = (2*loc_rect[1]+loc_rect[3])*0.5
                loc_rect = np.array([center_x, center_y, abs(loc_rect[2]), abs(loc_rect[3])])

                actual_loc_rects.append(loc_rect)
                actual_labels.append(label)

            pos_list, neg_list, expanded_gt_labels, expanded_gt_locs = self._matcher.match( 
                                                                                confs, 
                                                                                locs, 
                                                                                actual_labels, 
                                                                                actual_loc_rects)
            positives.append(pos_list)
            negatives.append(neg_list)
            ex_gt_labels.append(expanded_gt_labels)
            ex_gt_boxes.append(expanded_gt_locs)

        feed_dict = {self.input: input_images,
                     self.pos_val: positives,
                     self.neg_val: negatives,
                     self.gt_labels_val: ex_gt_labels,
                     self.gt_boxes_val: ex_gt_boxes}
        loss, _, = sess.run([ self._loss_op, self._train_op ], feed_dict=feed_dict)

        return _, loss

    def get_output(self, sess, input_data):
        feed_dict = {self.input: input_data}
        _ = sess.run([self.fmaps_wo], feed_dict=feed_dict)
        return _