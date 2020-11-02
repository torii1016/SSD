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

        self._xmin, self._ymin, self._xmax, self._ymax = self._convert_format([self._center_x, self._center_y, self._width, self._height])


    def _convert_format(self, inputs):
        """
        convert format from [center_x, center_y, width, height] to [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """

        rect = [inputs[0]-inputs[2]/2.0, inputs[1]-inputs[3]/2.0, inputs[0]+inputs[2]/2.0, inputs[1]+inputs[3]/2.0] #[top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        rect = [x if 0<=x else 0 for x in rect]     # exception process
        rect = [x if x<=1.0 else 1.0 for x in rect] # exception process
        return [rect[0], rect[1], rect[2], rect[3]]



class BoxGenerator(object):
    def __init__(self, config):
        self._scale_min = config["scale_min"]
        self._scale_max = config["scale_max"]
        self._aspects = config["aspect_set"]
        self._n_fmaps = len(self._aspects)
        self._default_boxes = []
    

    def _calc_scale(self, k):
        return self._scale_min+(self._scale_max-self._scale_min)*(k-1.0)/(self._n_fmaps-1.0)


    def generate_boxes(self, fmaps):
        """
        generate default boxes based on defined number
        """

        self._fmap_shapes = [fmap.get_shape().as_list() for fmap in fmaps]

        id = 0
        for k, map_shape in enumerate(self._fmap_shapes):
            s_k = self._calc_scale(k+1)

            for i, aspect in enumerate(self._aspects[k]):
                fmap_width  = self._fmap_shapes[k][1]
                fmap_height = self._fmap_shapes[k][2]

                # ---------------------------------------------
                # generate bbox for each cell grid in the feature map
                # ---------------------------------------------
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