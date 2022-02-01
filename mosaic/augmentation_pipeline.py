import os
import cv2
import random
import numpy as np
import math
import tensorflow as tf

class Mosic_Augmentation():
    def __init__(self, input_size, mosaic_border, yc, xc):    
        self.input_size = input_size
        self.mosaic_border = mosaic_border
        self.yc = yc
        self.xc = xc

    def box_candidates(self,box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

    def random_perspective(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
                
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

    def label_reorder(self,labels, label_first = True):
        reordered_label = np.zeros(labels.shape,dtype=np.float32)        
        
        # (x1,y1,x2,y2,label) -> (label,x1,y1,x2,y2)
        if label_first:
            reordered_label[:,0] = labels[:,4]
            reordered_label[:,1:] = labels[:,:4]

        # (label,x1,y1,x2,y2,label) -> (x1,y1,x2,y2,label)
        else:
            reordered_label[:,4] = labels[:,0]
            reordered_label[:,:4] = labels[:,1:]
        
        return reordered_label
    
    def create_mosaic_image(self, images, labels, mosaic_border, yc, xc):        
        mosic_image = np.full((self.input_size * 2, self.input_size * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        mosic_label = []
        
        for idx, (image, label) in enumerate(zip(images,labels)):
            h,w,_ = image.shape

            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.input_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.input_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.input_size * 2), min(self.input_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  
            padw = x1a - x1b
            padh = y1a - y1b

            # Normalized xywh to pixel xyxy format            
            label = self.label_reorder(label)
            labels_x = label.copy()
            if label.size > 0:  
                labels_x[:, 1] = (labels_x[:, 1] * w) + padw
                labels_x[:, 2] = (labels_x[:, 2] * h) + padh
                labels_x[:, 3] = (labels_x[:, 3] * w) + padw
                labels_x[:, 4] = (labels_x[:, 4] * h) + padh
                
            mosic_label.append(labels_x)
        
        # Concat/clip labels
        if len(mosic_label):
            mosic_label = np.concatenate(mosic_label, 0)
            np.clip(mosic_label[:, 1:], 0, 2 * self.input_size, out=mosic_label[:, 1:])  # use with random_perspective
            
        return mosic_image, mosic_label
    
    def __call__(self, images, labels):
                    
            mosic_image, mosic_label = self.create_mosaic_image(images,
                                                                labels,
                                                                self.mosaic_border, 
                                                                self.yc, 
                                                                self.xc)

            # Augment
            mosic_image, mosic_label = self.random_perspective(mosic_image,
                                                               mosic_label,
                                                               degrees=0.0,
                                                               translate=0.1,
                                                               scale=0.5,
                                                               shear=0.0,
                                                               perspective=0.0,
                                                               border=self.mosaic_border)  
            
            mosic_label = self.label_reorder(mosic_label,label_first=False)
            mosic_label[:,:-1]/=self.input_size
            
            return tf.constant(mosic_image,dtype='float32'), tf.constant(mosic_label,dtype='float32')