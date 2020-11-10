# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import pickle

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
    
    def __init__(self, param, config, image_info, label_name=None):
        self._lr = param["lr"]
        self._output_dim = param["output_class"]
        self._image_width, self._image_height, self._image_channels = image_info

        self._network = _ssd_network([config["SSD"]["network"]["name"]], config)
        self._box_generator = BoxGenerator(config["SSD"]["default_box"])
        self._label_name = label_name
        self._num = 0
        self._loss_old = None

    def set_model(self):
        self._set_network()
        self._default_boxes = self._box_generator.generate_boxes(self._fmaps)
        self._set_loss()
        self._set_optimizer()

        self._matcher = BBoxMatcher(n_classes=self._output_dim, 
                                    default_box_set=self._default_boxes,
                                    image_width = self._image_width,
                                    image_height = self._image_height)

        """
        image = np.zeros([300,300,3])
        for i in range(len(self._default_boxes)):
            image = self._default_boxes[i].draw_rect(image)
        cv2.imwrite("image.png", image)
        """

    def _set_network(self):
        
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self._image_height, self._image_width, self._image_channels])

        # ---------------------------------------------
        # confs = [None, default-bbox-numbers, class-numbers]
        # locs  = [None, default-bbox-numbers, bbox-info] : (center-x, center-y, width, height)
        # ---------------------------------------------
        self._fmaps, self._confs, self._locs = self._network.set_model(self.input, is_training=True, reuse=False) # train

        self._fmaps_wo, self._confs_wo, self._locs_wo = self._network.set_model(self.input, is_training=False, reuse=True) # inference
        self._confs_wo_softmax = tf.nn.softmax(self._confs_wo)


    def _set_loss(self):

        total_boxes = len(self._default_boxes)
        self.gt_labels_val = tf.compat.v1.placeholder(tf.int32, [None, total_boxes])
        self.gt_boxes_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes, 4])
        self.pos_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])
        self.neg_val = tf.compat.v1.placeholder(tf.float32, [None, total_boxes])

        # ---------------------------------------------
        # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
        # ---------------------------------------------
        smoothL1_op = smooth_L1(self.gt_boxes_val-self._locs)                           # loss = [batch-size, default-boxes, 4]
        loss_loc_op_ = tf.reduce_sum(smoothL1_op, reduction_indices=2)*self.pos_val     # loss = [batch-size, default-boxes]
        self._loss_loc_op = tf.reduce_sum(tf.reduce_sum(loss_loc_op_, reduction_indices=1)/(1e-5+tf.reduce_sum(self.pos_val, reduction_indices=1))) #average

        # ---------------------------------------------
        # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = category / label
        # ---------------------------------------------
        loss_conf_op_ = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                                            logits=self._confs, 
                                                            labels=self.gt_labels_val)
        loss_conf_op_ = loss_conf_op_*(self.pos_val+self.neg_val)
        self._loss_conf_op = tf.reduce_sum(tf.reduce_sum(loss_conf_op_, reduction_indices=1)/(1e-5+tf.reduce_sum((self.pos_val+self.neg_val), reduction_indices=1)))

        # Loss
        self._loss_op = tf.add(self._loss_conf_op, self._loss_loc_op)


    def _set_optimizer(self):
        #self._train_op = tf.compat.v1.train.AdamOptimizer(self._lr).minimize(self._loss_op, var_list = self._network.get_variables())
        self._train_op = tf.compat.v1.train.RMSPropOptimizer(self._lr).minimize(self._loss_op, var_list=self._network.get_variables())


    def train(self, sess, input_images, input_labels):
    
        positives = []
        negatives = []
        ex_gt_labels = []
        ex_gt_boxes = []

        feed_dict = {self.input: input_images}
        _, confs, locs = sess.run([self._fmaps, self._confs, self._locs], feed_dict=feed_dict)

        for i in range(len(input_images)):
            actual_labels = []
            actual_loc_rects = []
            actual_loc_rects_ = []

            image = input_images[i]*255
            
            for obj in input_labels[i]:
                loc_rect_type1 = [obj[0]*self._image_width, obj[1]*self._image_height, obj[2]*self._image_width, obj[3]*self._image_height]
                label = np.argmax(obj[4:])

                # ---------------------------------------------
                # convert location format
                # [xmin, ymin, xmax, ymax] → [center_x, center_y, width, height]
                # ---------------------------------------------
                width = loc_rect_type1[2]-loc_rect_type1[0]
                height = loc_rect_type1[3]-loc_rect_type1[1]

                # [xmin, ymin, xmax, ymax] → [xmin, ymin, width, height]
                loc_rect_type2 = np.array([loc_rect_type1[0], loc_rect_type1[1], width, height])

                center_x = loc_rect_type2[0]+loc_rect_type2[2]*0.5
                center_y = loc_rect_type2[1]+loc_rect_type2[3]*0.5
                loc_rect_type2 = np.array([center_x, center_y, abs(loc_rect_type2[2]), abs(loc_rect_type2[3])])
                        
                actual_loc_rects.append(loc_rect_type2) # [center_x, center_y, width, height]
                actual_loc_rects_.append(loc_rect_type1) # [xmin, ymin, xmax, ymax]
                actual_labels.append(label)


            pos_list, neg_list, expanded_gt_labels, expanded_gt_locs = self._matcher.match( 
                                                                                confs[i], 
                                                                                actual_labels, 
                                                                                actual_loc_rects,
                                                                                actual_loc_rects_)
            positives.append(pos_list)
            negatives.append(neg_list)
            ex_gt_labels.append(expanded_gt_labels)
            ex_gt_boxes.append(expanded_gt_locs)

            """
            for k in range(len(pos_list)):
                if pos_list[k]==1:
                    [xmin, ymin, xmax, ymax], _ = self._default_boxes[k].get_bbox_info(
                                                    self._image_width,
                                                    self._image_height,
                                                    center_x={"bbox":self._default_boxes[k]._center_x, "offset":expanded_gt_locs[k][0]},
                                                    center_y={"bbox":self._default_boxes[k]._center_y, "offset":expanded_gt_locs[k][1]},
                                                    width=   {"bbox":self._default_boxes[k]._width,    "offset":np.exp(expanded_gt_locs[k][2])},
                                                    height=  {"bbox":self._default_boxes[k]._height,   "offset":np.exp(expanded_gt_locs[k][3])})

                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0))
            cv2.imwrite("image_{}_gt.png".format(i), image)
            """


        feed_dict = {self.input: input_images,
                     self.pos_val: positives,
                     self.neg_val: negatives,
                     self.gt_labels_val: ex_gt_labels,
                     self.gt_boxes_val: ex_gt_boxes}
        loss, _, loss_conf, loss_loc, confs, locs = sess.run([self._loss_op, self._train_op, self._loss_conf_op, self._loss_loc_op, self._confs, self._locs], feed_dict=feed_dict)

        #w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="SSD/biasesfmap6")[0]
        #print("weight:{}".format(w))
        #print("weight:{}".format(w.eval(session=sess)))
        #print("weight:{}".format(self._network.get_variables()))

        self._loss_old = loss


        return _, loss, loss_conf, loss_loc


    def inference(self, sess, input_data):
        """
        this method returns inference results (pred-confs and locs)
        """

        feed_dict = {self.input: [input_data]}
        pred_confs, pred_locs = sess.run([self._confs_wo_softmax, self._locs_wo], feed_dict=feed_dict)
        return np.squeeze(pred_confs), np.squeeze(pred_locs)  # remove extra dimension


    def detect_objects(self, pred_confs, pred_locs, n_top_probs=200, prob_min=0.001, overlap_threshold=0.7):
        """
        this method returns detected objects list (means high confidences locs and its labels)
        """

        # ---------------------------------------------
        # extract maximum class possibility, bbox width/height
        # ---------------------------------------------
        possibilities = [np.amax(conf) for conf in pred_confs]

        # ---------------------------------------------
        # extract the top 200 with the highest possibility value
        # ---------------------------------------------
        indicies = np.argpartition(possibilities, -n_top_probs)[-n_top_probs:] # index
        top200 = np.asarray(possibilities)[indicies] # value
        print("top200:{}".format(top200))

        # ---------------------------------------------
        # exclude candidates with a possibility value below the threshold
        # ---------------------------------------------
        slicer = indicies[prob_min<top200] # index

        # ---------------------------------------------
        # generate detected bbox (default-box + offset)
        # ---------------------------------------------
        def generate_bbox(dboxs, offsets):
            rects = []
            for dbox, offset in zip(dboxs, offsets):
                [xmin, ymin, xmax, ymax], _ = dbox.get_bbox_info(self._image_width,
                                                                 self._image_height,
                                                                 center_x={"bbox":dbox._center_x, "offset":offset[0]},
                                                                 center_y={"bbox":dbox._center_y, "offset":offset[1]},
                                                                 width=   {"bbox":dbox._width,    "offset":np.exp(offset[2])},
                                                                 height=  {"bbox":dbox._height,   "offset":np.exp(offset[3])})
                rects.append([xmin, ymin, xmax, ymax])
            return np.array(rects)

        default_box = [self._default_boxes[i] for i in slicer]
        locations = generate_bbox(default_box, pred_locs[slicer])


        labels = [np.argmax(conf) for conf in pred_confs[slicer]]
        labels = np.asarray(labels).reshape(len(labels), 1)

        # ---------------------------------------------
        # non-maximum suppression
        # ---------------------------------------------
        index = non_maximum_suppression(boxes=locations, labels=labels, overlap_threshold=overlap_threshold)

        filtered_locs = locations[index]
        filtered_labels = labels[index]


        # ---------------------------------------------
        # exception process
        # ---------------------------------------------
        if len(filtered_locs)==0:
            filtered_locs = np.zeros((4, 4))
            filtered_labels = np.zeros((4, 1))
        
        return filtered_locs, filtered_labels
    

    def save(self, image, positive, negative, ex_gt_labels, ex_gt_boxes, loss, loss_conf, loss_loc, conf, loc):
        pickle.dump(loss, open("debug/loss_{}.pickle".format(self._num), "wb"))
        pickle.dump(loss_conf, open("debug/loss_conf_{}.pickle".format(self._num), "wb"))
        pickle.dump(loss_loc, open("debug/loss_loc_{}.pickle".format(self._num), "wb"))
        pickle.dump(positive, open("debug/positive_{}.pickle".format(self._num), "wb"))
        pickle.dump(negative, open("debug/negative_{}.pickle".format(self._num), "wb"))
        pickle.dump(ex_gt_boxes, open("debug/ex_gt_boxes_{}.pickle".format(self._num), "wb"))
        pickle.dump(ex_gt_labels, open("debug/ex_gt_labels_{}.pickle".format(self._num), "wb"))
        pickle.dump(conf, open("debug/conf_{}.pickle".format(self._num), "wb"))
        pickle.dump(loc, open("debug/loc_{}.pickle".format(self._num), "wb"))
        self._num += 1