import numpy as np

def non_maximum_suppression(boxes, labels, overlap_threshold=0.7):
    """
    Non-Maximum Suppression algorithm
    this is nms(non maximum_suppression) which filters predicted objects.

    [Input]
        boxes  : [good-boxes-numbet][box-info-dim] (box-info-dim: [top-left x-coordinate, top-left y-coordinate, bottom-right x-coordinate, bottom-right y-coordinate])
        labels : [good-boxes-numbet]
    """
    
    if len(boxes)==0:
        return [], []

    picked = []

    # ---------------------------------------------
    # devide boxes info: [top-left x-coordinate, top-left y-coordinate, bottom-right x-coordinate, bottom-right y-coordinate]
    # ---------------------------------------------
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # ---------------------------------------------
    # compute the area of the bounding boxes and sort the bounding box
    # ---------------------------------------------
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            # ---------------------------------------------
            # extract smallest and largest bouding boxes
            # ---------------------------------------------
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            # ---------------------------------------------
            # overlap of current box and those in area list
            # ---------------------------------------------
            overlap = float(w*h)/area[j]
            # ---------------------------------------------
            # suppress current box
            # ---------------------------------------------
            if overlap>overlap_threshold:
                suppress.append(pos)

        # ---------------------------------------------
        # delete suppressed indexes
        # ---------------------------------------------
        idxs = np.delete(idxs, suppress)

    return picked