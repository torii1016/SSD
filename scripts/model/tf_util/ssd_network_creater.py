# -*- coding:utf-8 -*-

import sys
import tensorflow as tf

from .network_creater import NetworkCreater
from .network import conv2d

class SSDNetworkCreater(NetworkCreater):
    def __init__(self, config, name_scope):
        super().__init__(config, name_scope)
        self._creater = {"conv2d":self._conv2d_creater,
                        "fc":self._fc_creater,
                        "reshape":self._reshape_creater,
                        "transform":self._transform_creater,
                        "maxpool":self._maxpool_creater}
        self._extra_feature = []

    def _conv2d_creater(self, inputs, data, is_training=False, reuse=True):
        h = conv2d(inputs=inputs,
                   scope=self._name_scope,
                   name=data["name"], 
                   output_channels=data["output_channel"],
                   filter_size=data["fileter_size"],
                   stride=data["stride"],
                   padding=data["padding"],
                   bn=data["bn"],
                   activation_fn=self._active_function_list[data["activation_fn"]],
                   is_training=is_training,
                   reuse=reuse)
        
        if data["extra_feature"]:
            self._extra_feature.append({"feature":h, "layer":data["name"]})
        return h
    
    def get_extra_feature(self):
        return self._extra_feature