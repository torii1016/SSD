import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

from .bounding_box import BoundingBox 


class BBoxMatcher(object):
    def __init__(self, n_classes, default_box_set):
        self._n_classes = n_classes
        self._default_boxes = default_box_set


    def calc_jaccard(self, rect1, rect2):

        def intersection(rect1, rect2):
            top = max(rect1[1], rect2[1])
            left = max(rect1[0], rect2[0])
            right = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
            bottom = min(rect1[1]+rect1[3], rect2[1]+rect2[3])

            if bottom>top and right>left:
                return (bottom-top)*(right-left)
            return 0

        rect1_ = [x if x>=0 else 0 for x in rect1]
        rect2_ = [x if x>=0 else 0 for x in rect2]
        s = rect1_[2]*rect1_[3]+rect2_[2]*rect2_[3]

        intersect = intersection(rect1_, rect2_)
        union = s-intersect

        return intersect/union


    def extract_highest_indicies(self, pred_confs, max_length):

        loss_confs = []
        for pred_conf in pred_confs:
            pred = np.exp(pred_conf)/(np.sum(np.exp(pred_conf))+1e-5)
            loss_confs.append(np.amax(pred))

        size = min(len(loss_confs), max_length)
        indicies = np.argpartition(loss_confs, -size)[-size:]

        return indicies


    def match(self, pred_confs, pred_locs, actual_labels, actual_locs):
        n_pos = 0 
        n_neg = 0
        pos_list = []
        neg_list = []
        expanded_gt_labels = []
        expanded_gt_locs = []
        bboxes_matched = []
        bboxes_label_matched = []

        for i in range(len(self._default_boxes)):
            bboxes_matched.append(None)

        for gt_label, gt_box in zip(actual_labels, actual_locs):

            for i in range(len(bboxes_matched)):
                dbox_rect = [self._default_boxes[i]._center_x, 
                             self._default_boxes[i]._center_y,
                             self._default_boxes[i]._height,
                             self._default_boxes[i]._width]

                jacc = self.calc_jaccard(gt_box, dbox_rect)
                
                if(jacc>=0.5):
                    bboxes_matched[i]=BoundingBox(label=gt_label, rect_loc=gt_box)
                    n_pos += 1
                    bboxes_label_matched.append(gt_label)

        
        neg_pos = 5
        indicies = self.extract_highest_indicies(pred_confs, n_pos*neg_pos)
        for i in indicies:
            if(n_neg>n_pos*neg_pos):
                    break

            if(bboxes_matched[i] is None and self._n_classes-1!=np.argmax(pred_confs[i])):
                bboxes_matched[i] = BoundingBox(label=self._n_classes-1, rect_loc=[])

                n_neg += 1

        for box in bboxes_matched:
            if box is None:
                pos_list.append(0)
                neg_list.append(0)
                expanded_gt_labels.append(self._n_classes-1)
                expanded_gt_locs.append([0]*4)

            elif 0==len(box._rect_loc):
                pos_list.append(0)
                neg_list.append(1)
                expanded_gt_labels.append(self._n_classes-1)
                expanded_gt_locs.append([0]*4)

            else:
                pos_list.append(1)
                neg_list.append(0)
                expanded_gt_labels.append(box._label)
                expanded_gt_locs.append(box._rect_loc)


        return pos_list, neg_list, expanded_gt_labels, expanded_gt_locs



