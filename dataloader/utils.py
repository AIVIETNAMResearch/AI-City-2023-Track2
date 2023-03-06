import random
import numpy as np
import cv2
import os

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    #print(intervals)
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        
    #print(ranges)
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_motion_img(data_dir, img_path_list, boxes_list, num_frames):
    first = cv2.imread(os.path.join(data_dir, img_path_list[0][2:]))
    w, h, c = first.shape
    motion_img = first
    line_motion_img = np.zeros((w, h, c), dtype=np.int16)
    
    center_points = []
    
    prev_boxes = None
    box_indices = []
    
    for idx, boxes in enumerate(boxes_list):
        x, y, w, h = boxes

        if prev_boxes is None:
            prev_boxes = [x, y, x+w, y+h]
            box_indices.append(idx)
        else:
            curr_boxes = [x, y, x+w, y+h]
            
            if bb_intersection_over_union(prev_boxes, curr_boxes) < 0.05:
                prev_boxes = curr_boxes
                box_indices.append(idx)
                
    # print("BOXLEN: ", len(box_indices))
    for idx in box_indices:
        img_path = img_path_list[idx]
        img = cv2.imread(os.path.join(data_dir, img_path[2:]))
        
        x, y, w, h = boxes_list[idx]
        context_img = img[y:y+h, x:x+w, :]

        motion_img[y:y+h, x:x+w, :] = context_img
        center_points.append((int(x+ w/2),int(y + h/2)))
        
    center_points = np.array(center_points)

    for point1, point2 in zip(center_points, center_points[1:]):
        cv2.line(line_motion_img, point1, point2, [255, 255, 255], 80)    
        
    frame_indexes = sample_frames(num_frames, len(img_path_list), sample='uniform')
    
    context_images = []
    for idx in frame_indexes:
        img = cv2.imread(os.path.join(data_dir, img_path_list[idx][2:]))
        x, y, w, h = boxes_list[idx]
        context_img = img[y:y+h, x:x+w, :]
        context_images.append(context_img)
    
    return context_images, line_motion_img, motion_img