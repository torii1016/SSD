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


    def print(self, image_width, image_height):
        print("------------------------------")
        print("center_x:{}, center_y:{}".format(self._center_x, self._center_y))
        print("width:{}, height:{}".format(self._width, self._height))
        print("scale:{}, aspect:{}".format(self._scale, self._aspect))

        print("center_x:{}, center_y:{}".format(int(image_width*self._center_x-0.5), int(image_height*self._center_y-0.5)))
        #print("width:{}, height:{}".format(image_width*self._width*self._scale*(1/np.sqrt(self._aspect)), image_height*self._height*self._scale*np.sqrt(self._aspect)))
        print("width:{}, height:{}".format(int(image_width*self._width), int(image_height*self._height)))
        print("------------------------------")

    def _convert_format(self, inputs):
        """
        convert format from [center_x, center_y, width, height] to [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """

        rect = [inputs[0]-inputs[2]/2.0, inputs[1]-inputs[3]/2.0, inputs[0]+inputs[2]/2.0, inputs[1]+inputs[3]/2.0] #[top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        rect = [x if 0<=x else 0 for x in rect]     # exception process
        rect = [x if x<=1.0 else 1.0 for x in rect] # exception process
        return [rect[0], rect[1], rect[2], rect[3]]


    def draw_rect(self, image, color=(0,255,0), thickness=1):
        """
        draw the default bounding box
        """

        [point1_x, point1_y, point2_x, point2_y], _ = self.get_bbox_info(image_width=image.shape[0],
                                                                    image_height=image.shape[1],
                                                                    center_x={"bbox":self._center_x, "offset":0.0},
                                                                    center_y={"bbox":self._center_y, "offset":0.0},
                                                                    width=   {"bbox":self._width,    "offset":1.0},
                                                                    height=  {"bbox":self._height,   "offset":1.0})
        image = cv2.rectangle(img=image,
                              pt1=(point1_x, point1_y),
                              pt2=(point2_x, point2_y),
                              color=color,
                              thickness=thickness)

        return image
    

    def get_bbox_info(self, image_width, image_height, center_x, center_y, width, height):
        """
        [Input]
            image_width  : input image width
            image_hieght : input image height
            center_x     : bounding box center-x {"bbox":--, "offset":--}
            center_y     : bounding box center-y {"bbox":--, "offset":--}
            width        : bounding box width    {"bbox":--, "offset":--}
            height       : bounding box height   {"bbox":--, "offset":--}
        """

        box_center_x = image_width*center_x["bbox"]-0.5
        box_center_y = image_height*center_y["bbox"]-0.5
        box_width = image_width*width["bbox"]
        #box_width = image_width*width["bbox"]*self._scale*(1/np.sqrt(self._aspect))
        box_height = image_height*height["bbox"]
        #box_height = image_height*height["bbox"]*self._scale*np.sqrt(self._aspect)

        real_center_x = box_center_x+box_width*center_x["offset"]
        real_center_y = box_center_y+box_height*center_y["offset"]
        real_width = box_width*width["offset"]
        real_height = box_height*height["offset"]

        xmin = int(real_center_x-real_width/2) 
        ymin = int(real_center_y-real_height/2)
        xmax = int(real_center_x+real_width/2)
        ymax = int(real_center_y+real_height/2)


        def exception_process(x, min_x, max_x):
            if x<min_x:
                return min_x
            elif max_x<x:
                return max_x
            else:
                return x
        
        xmin = exception_process(xmin, min_x=0, max_x=image_width-1)
        xmax = exception_process(xmax, min_x=0, max_x=image_width-1)
        ymin = exception_process(ymin, min_x=0, max_x=image_height-1)
        ymax = exception_process(ymax, min_x=0, max_x=image_height-1)

        return [xmin, ymin, xmax, ymax], [real_center_x, real_center_y, real_width, real_height]



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