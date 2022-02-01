from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import math

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    
    layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2, iou_type=None):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    
    union = (box_1_area + box_2_area - int_area)
    iou = int_area/union # base iou
    
    # CIOU
    eps = 1e-7
    
    b1_x1, b1_x2, b1_y1, b1_y2 = box_1[..., 0], box_1[..., 2], box_1[..., 1], box_1[..., 3]
    b2_x1, b2_x2, b2_y1, b2_y2 = box_2[..., 0], box_2[..., 2], box_2[..., 1], box_2[..., 3]
    
    cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
    ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
    
    c2 = cw ** 2 + ch ** 2 + eps  
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    v = (4 / math.pi ** 2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
    alpha = v / (v - iou + (1 + eps))
    
    return iou - (rho2 / c2 + v * alpha)

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

'''
Draw utils are below
'''

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_result_with_gt(img, gt_result, pred_result, classname_path):
    
    classname_dict = {}
    with open(classname_path,'r') as f: 
        for idx, l in enumerate(f.readlines()):
            classname_dict[idx] = l.replace('\n','')
            classname_dict[l.replace('\n','')] = l.replace('\n','')
            
    h,w = img.shape[0:2]
    
    draw_img = np.zeros((h,w*2,3),np.uint8)
    draw_gt = img.copy()
    draw_pred = img.copy()
    
    for gt in gt_result:
        x1,y1,width,height = gt['bbox']
        category_id = gt['category_id']
        
        draw_gt = cv2.rectangle(draw_gt,(x1,y1),(x1+width,y1+height), (255,0,0),2)
        draw_gt = cv2.putText(draw_gt, '{}'.format(classname_dict[category_id]),(x1,y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    for _,x1,y1,width,height,score,category_id in pred_result:
        draw_pred = cv2.rectangle(draw_pred,(x1,y1),(x1+width,y1+height), (255,255,0),2)
        draw_pred = cv2.putText(draw_pred, 
                                '{} {:.2}'.format(classname_dict[category_id], score),
                                (x1,y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    
    draw_img[:,0:w,:] = draw_pred
    draw_img[:,w:w*2,:] = draw_gt
    
    return draw_img
            
def vis(history,name) :
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    
    plt.text(np.argmin(value)+1,
             min(value),
             str(round(min(value),2)),
             color='b',
             horizontalalignment='center',
             verticalalignment='bottom')
    
    if val_value is not None :
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
        text_color='r'
        
        plt.text(np.argmin(val_value)+1,
             min(val_value),
             str(round(min(val_value),2)),
             color='r',
             horizontalalignment='center',
             verticalalignment='top')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2) , fontsize=10 , ncol=1)
    
def plot_history(history, save_path, keys=['loss']) :
    key_value = list(set([i.split("val_")[-1] for i in list(keys)]))
    plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.savefig(save_path)