# -*- coding:utf-8 -*-

import sys
import tensorflow as tf

from .network_creater import NetworkCreater

class ExtraFeatureMapNetworkCreater(NetworkCreater):
    def __init__(self, config, name_scope):
        super().__init__(config, name_scope)
        self._creater = {"conv2d":self._conv2d_creater,
                        "fc":self._fc_creater,
                        "reshape":self._reshape_creater,
                        "transform":self._transform_creater,
                        "maxpool":self._maxpool_creater}
        self._n_classes = config["network"]["output_class"]
        self._n_boxes = config["network"]["box_number"]
        

    def create(self, inputs, config, is_training=True, reuse=False):
        # Extra feature maps
        self._fmaps = []
        for i, layer in enumerate(list(config.keys())[self._model_start_key:]):
            self._fmaps.append(self._creater[config[layer]["type"]](inputs=inputs[i]["feature"],
                                                    data=config[layer],
                                                    is_training=is_training,
                                                    reuse=reuse))

        # calculate class id and score
        fmaps_reshaped = []
        for i, fmap in zip(range(len(self._fmaps)), self._fmaps):
            output_shape = fmap.get_shape().as_list() # [batch_size=None, image_height, image_width, n_channles]
            
            fmap_height = output_shape[1]
            fmap_width = output_shape[2]
            
            # [batch_size=None, image_height, image_width, n_channles] → [batch_size=None, xxx, self.n_classes + 4 ] に　reshape
            fmap_reshaped = tf.reshape(fmap, [-1, fmap_width * fmap_height * self._n_boxes[i], self._n_classes + 4])

            fmaps_reshaped.append(fmap_reshaped)

        fmap_concatenated = tf.concat(fmaps_reshaped, axis=1)

        self.pred_confs = fmap_concatenated[:, :, :self._n_classes] # shape = [None, None, 21] | 21: class num
        self.pred_locs = fmap_concatenated[:, :, self._n_classes:] # shape = [None, None, 4]  | 4 : (xmin, ymin, xmax, ymax) の bounding box info

        return self._fmaps, self.pred_confs, self.pred_locs