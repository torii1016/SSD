# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, smooth_L1, SSDNetworkCreater, ExtraFeatureMapNetworkCreater
from .bbox_matcher import BBoxMatcher
from .default_box_generator import BoxGenerator
from .non_maximum_suppression import non_maximum_suppression

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
        
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channels])

        # ---------------------------------------------
        # confs = [None, default-bbox-numbers, class-numbers]
        # locs  = [None, default-bbox-numbers, bbox-info] : (xmin, ymin, xmax, ymax)
        # ---------------------------------------------
        self.fmaps, self.confs, self.locs = self.network.set_model(self.input, is_training=True, reuse=False) # train

        self.fmaps_wo, self.confs_wo, self.locs_wo = self.network.set_model(self.input, is_training=False, reuse=True) # inference
        self.confs_wo_softmax = tf.nn.softmax(self.confs_wo)


    def set_loss(self):

        total_boxes = len(self.default_boxes)
        self.gt_labels_val = tf.compat.v1.placeholder(tf.int32, [None, total_boxes])
        self.gt_boxes_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes, 4])
        self.pos_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])
        self.neg_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])

        # ---------------------------------------------
        # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
        # ---------------------------------------------
        smoothL1_op = smooth_L1(self.gt_boxes_val-self.locs)
        loss_loc_op = tf.reduce_sum(smoothL1_op, reduction_indices=2)*self.pos_val
        loss_loc_op = tf.reduce_sum(loss_loc_op, reduction_indices=1)/(1e-5+tf.reduce_sum(self.pos_val, reduction_indices=1)) #average

        # ---------------------------------------------
        # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = category / label
        # ---------------------------------------------
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
                actual_loc_rects.append(obj[:4])
                actual_labels.append(np.argmax(obj[4:]))

            pos_list, neg_list, expanded_gt_labels, expanded_gt_locs = self._matcher.match( 
                                                                                confs[i], 
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
        loss, _, = sess.run([self._loss_op, self._train_op], feed_dict=feed_dict)

        return _, loss


    def inference(self, sess, input_data):
        """
        this method returns inference results (pred-confs and locs)
        """

        feed_dict = {self.input: [input_data]}
        pred_confs, pred_locs = sess.run([self.confs_wo_softmax, self.locs_wo], feed_dict=feed_dict)
        return np.squeeze(pred_confs), np.squeeze(pred_locs)  # remove extra dimension


    def detect_objects(self, pred_confs, pred_locs, n_top_probs=200, prob_min=0.001, overlap_threshold=0.1):
        """
        this method returns detected objects list (means high confidences locs and its labels)
        """

        # ---------------------------------------------
        # extract maximum class possibility
        # ---------------------------------------------
        possibilities = [np.amax(conf) for conf in pred_confs]

        # ---------------------------------------------
        # extract the top 200 with the highest possibility value
        # ---------------------------------------------
        indicies = np.argpartition(possibilities, -n_top_probs)[-n_top_probs:]
        top200 = np.asarray(possibilities)[indicies]

        # ---------------------------------------------
        # exclude candidates with a possibility value below the threshold
        # ---------------------------------------------
        slicer = indicies[prob_min<top200]

        # ---------------------------------------------
        # exception process
        # ---------------------------------------------
        locations = pred_locs[slicer]
        locations = np.delete(locations, locations[:,2]==0, 0)  # exception process
        locations = np.delete(locations, locations[:,3]==0, 0)  # exception process

        labels = []
        for conf in pred_confs[slicer]:
            labels.append(np.argmax(conf))
        labels = np.asarray(labels).reshape(len(labels), 1)

        # ---------------------------------------------
        # non-maximum suppression
        # ---------------------------------------------
        filtered_locs, filtered_labels = non_maximum_suppression(boxes=locations, labels=labels, overlap_threshold=overlap_threshold)

        # ---------------------------------------------
        # exception process
        # ---------------------------------------------
        if len(filtered_locs)==0:
            filtered_locs = np.zeros((4, 4))
            filtered_labels = np.zeros((4, 1))

        return filtered_locs, filtered_labels