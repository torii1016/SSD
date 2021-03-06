import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

from .bounding_box import BoundingBox 


class BBoxMatcher(object):
    def __init__(self, n_classes, default_box_set, image_width, image_height):
        self._n_classes = n_classes
        self._default_boxes = default_box_set
        self._image_width = image_width
        self._image_height = image_height


    def calc_jaccard(self, rect1, rect2):
        """
        calculate the jaccard overlap for default box rect and gt box rect
        """

        #print("rect1:{}".format(rect1))
        #print("rect2:{}".format(rect2))

        #rect is defined as [xmin, ymin, xmax, ymax]
        def intersection(rect1, rect2):
            top = max(rect1[1], rect2[1])
            left = max(rect1[0], rect2[0])
            right = min(rect1[2], rect2[2])
            bottom = min(rect1[3], rect2[3])

            if bottom>top and right>left:
                return (bottom-top)*(right-left)
            return 0
        
        s = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])+(rect2[2]-rect2[0])*(rect2[3]-rect2[1])

        intersect = intersection(rect1, rect2)
        union = s-intersect

        return intersect/union


    def extract_highest_indicies(self, pred_confs, max_length):
        """
        extract specific indicies, that is, have most high loss_confs
        """

        loss_confs = []
        for pred_conf in pred_confs:
            pred = np.exp(pred_conf)/(np.sum(np.exp(pred_conf))+1e-5)
            loss_confs.append(np.amax(pred))

        size = min(len(loss_confs), max_length)
        indicies = np.argpartition(loss_confs, -size)[-size:]

        return indicies


    def _generate_gt(self, gt_box, dbox):
        gt_cx = (gt_box[0]-dbox[0])/dbox[2]
        gt_cy = (gt_box[1]-dbox[1])/dbox[3]
        gt_w = np.log(gt_box[2]/dbox[2])
        gt_h = np.log(gt_box[3]/dbox[3])
        return [gt_cx, gt_cy, gt_w, gt_h]


    def match(self, pred_confs, actual_labels, actual_locs, actual_locs_2):
        """
        matching strategy

        [Input]
            pred_confs    : [default-box-number][class-number]
            actual_labels : [gt-box-number][label-id]
            actual_locs   : [gt-box-number][center_x, center_y, width, height]
            actual_locs_2 : [gt-box-number][xmin, ymin, xmax, ymax]
        """

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

        # ---------------------------------------------
        # generate correct bounding box
        # ---------------------------------------------
        for gt_label, gt_box, gt_box_2 in zip(actual_labels, actual_locs, actual_locs_2):

            for i in range(len(bboxes_matched)):
                [xmin, ymin, xmax, ymax], [center_x, center_y, width, height] = self._default_boxes[i].get_bbox_info(
                                                                              self._image_width,
                                                                              self._image_height,
                                                                              center_x={"bbox":self._default_boxes[i]._center_x, "offset":0.0},
                                                                              center_y={"bbox":self._default_boxes[i]._center_y, "offset":0.0},
                                                                              width=   {"bbox":self._default_boxes[i]._width,    "offset":1.0},
                                                                              height=  {"bbox":self._default_boxes[i]._height,   "offset":1.0})
                

                jacc = self.calc_jaccard(gt_box_2, [xmin, ymin, xmax, ymax])

                
                if(jacc>=0.5):
                    dbox_rect = [center_x, center_y, width, height]
                    gt = self._generate_gt(gt_box, dbox_rect)

                    bboxes_matched[i]=BoundingBox(label=gt_label, rect_loc=gt)
                    n_pos += 1
                    bboxes_label_matched.append(gt_label)

        
        # ---------------------------------------------
        # generate bounding box of non-applicable bbox
        # ---------------------------------------------
        neg_pos = 3
        indicies = self.extract_highest_indicies(pred_confs, n_pos*neg_pos)
        for i in indicies:
            if(n_neg>n_pos*neg_pos):
                    break

            if(bboxes_matched[i] is None and self._n_classes-1!=np.argmax(pred_confs[i])):
                bboxes_matched[i] = BoundingBox(label=self._n_classes-1, rect_loc=[])
                n_neg += 1


        # ---------------------------------------------
        # positive/negative judgment for the generated bbox
        # ---------------------------------------------
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