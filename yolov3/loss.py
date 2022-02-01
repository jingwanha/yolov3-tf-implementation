import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy,sparse_categorical_crossentropy

from .utils import broadcast_iou
from .models import yolo_boxes

def YoloLoss(anchors, input_size, classes, ignore_thresh=0.5, smoothing_factor=0):    
    
    def yolo_loss(y_true, y_pred):
            
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, input_size, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations        
        grid_size = tf.shape(y_true)[1]        
        
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
                             (pred_box, true_box, obj_mask),tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        
        # xy_loss
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        
        # wh_loss
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        
        # object loss
        obj_loss = binary_crossentropy(true_obj, pred_obj, label_smoothing=smoothing_factor)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        # focal loss 적용
        alpha = 1.0
        gamma = 2.0        
        focal_mask = alpha * tf.pow(tf.abs(obj_mask - tf.squeeze(pred_obj)), gamma)
        obj_loss *= focal_mask
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    
        
        # class loss        
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        
        return xy_loss + wh_loss + obj_loss + class_loss
    
    return yolo_loss