import numpy as np
import tensorflow as tf

import cv2

class DefaultBox(object):
    def __init__( 
            self,
            group_id=None,
            id=None,
            center_x=0.0,
            center_y=0.0,
            width=1,
            height=1,
            scale=1,
            aspect=1):

        self._group_id = group_id        
        self._id = id

        self._center_x = center_x
        self._center_y = center_y

        self._width = width
        self._height = height

        self._scale = scale
        self._aspect = aspect


    def draw_rect(self, image, color=(0,0,255), thickness=1):
        center_x = image.shape[0] * self._center_x - 0.5
        center_y = image.shape[1] * self._center_y - 0.5
        width = image.shape[0] * self._width * self._scale * (1 / np.sqrt( self._aspect ) )
        height = image.shape[1] * self._height * self._scale * np.sqrt( self._aspect )

        point1_x = int( center_x - width/2 )   # 長方形の左上 x 座標
        point1_y = int( center_y - height/2 )  # 長方形の左上 y 座標
        point2_x = int( center_x + width/2 )   # 長方形の右下 x 座標
        point2_y = int( center_y + height/2 )  # 長方形の右下 y 座標

        image = cv2.rectangle(
                    img = image,
                    pt1 = ( point1_x, point1_y ),  # 長方形の左上座標
                    pt2 = ( point2_x, point2_y ),  # 長方形の右下座標
                    color = color,                 # BGR
                    thickness = thickness          # 線の太さ（-1 の場合、color で設定した色で塗りつぶし）
                )
        return image


class BoxGenerator(object):
    def __init__(self, config):
        self._scale_min = config["scale_min"]
        self._scale_max = config["scale_max"]
        self._aspects = config["aspect_set"]
        self._n_fmaps = len(self._aspects)
        self._default_boxes = []
    

    def _calc_scale(self, k):
        return self._scale_min+(self._scale_max-self._scale_min)*(k-1.0)/(self._n_fmaps-1.0)
    

    def _calc_fmap_shape(self, fmaps):
        return [fmap.get_shape().as_list() for fmap in fmaps]


    def generate_boxes(self, fmaps):
        self._fmap_shapes = self._calc_fmap_shape(fmaps)

        id = 0
        for k, map_shape in enumerate(self._fmap_shapes):
            s_k = self._calc_scale(k+1)

            for i, aspect in enumerate(self._aspects[k]):
                fmap_width  = self._fmap_shapes[k][1]
                fmap_height = self._fmap_shapes[k][2]

                for y in range(fmap_height):
                    center_y = (y+0.5)/float(fmap_height)

                    for x in range(fmap_width):
                        center_x = (x+0.5)/float(fmap_width)

                        box_width = s_k*np.sqrt(aspect)
                        box_height = s_k/np.sqrt(aspect)

                        id += 1
                        default_box = DefaultBox(
                                        group_id = k+1,
                                        id = id,
                                        center_x = center_x, center_y = center_y,
                                        width = box_width, height = box_height, 
                                        scale = s_k,
                                        aspect = aspect)

                        self._default_boxes.append(default_box)

        return self._default_boxes