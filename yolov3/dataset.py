import os
import random
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .augmentation_pipeline import basic_augmentation, Mosic_Augmentation
from .anchors import anchor_masks

def data_augmentation(dataset, n_time):
    merge_dataset = dataset
    
    for idx, (image, boxes) in tqdm(enumerate(dataset.as_numpy_iterator())):
        for _ in range(n_time):
            aug_i, aug_box = basic_augmentation(image,boxes)
            merge_dataset = merge_dataset.concatenate(tf.data.Dataset.from_tensors((aug_i,aug_box)))
    
    return merge_dataset

def data_mosaic(dataset, input_size, n_time):
    mosaic_border = [-input_size // 2, - input_size // 2]
    yc, xc = [int(random.uniform(-x, 2 * input_size + x)) for x in mosaic_border]  # mosaic center x, y
    
    mosic_aug = Mosic_Augmentation(input_size, mosaic_border, yc ,xc)

    # 이미지를 한 장씩 로드
    for loaded_image, loaded_labels in tqdm(dataset.as_numpy_iterator()):
        candidate_images = [loaded_image]
        candidate_labels = [loaded_labels]

        # mosic 할 임의의 3장 이미지&라벨 로드
        dataset = dataset.shuffle(1024)
        for picked_image, picked_labels in dataset.take(3):
            candidate_images.append(picked_image)
            candidate_labels.append(picked_labels)
            
        for _ in range(n_time):
            mosaic_i, mosaic_box = mosic_aug(candidate_images,candidate_labels)
            
            try : merge_dataset = merge_dataset.concatenate(tf.data.Dataset.from_tensors((mosaic_i,mosaic_box)))
            except : merge_dataset = tf.data.Dataset.from_tensors((mosaic_i,mosaic_box))
    
    return merge_dataset

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, size):

    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),(1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


IMAGE_FEATURE_MAP = {
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size):
    
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(x['image/object/class/text'], default_value='')
    
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)
    
    return x_train, y_train

def padding_(max_boxes, x_train, y_train):
    paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return x_train, y_train

def load_tfrecord_dataset(file_pattern, class_file, size=416):    
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))    
    
def _trfrecord_factorize(example): return tf.io.parse_single_example(example, IMAGE_FEATURE_MAP)

def create_gt_from_tfrecord(data_path, classname_path, input_size):
    
    # tfrecord를 읽어서 coco 형태의 gt를 만듬
    dataset = tf.data.TFRecordDataset(data_path).map(_trfrecord_factorize)
    class_table = class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(classname_path, 
                                                                                        tf.string, 
                                                                                        0, tf.int64, -1, delimiter="\n"), -1)
    gt_dict = {
    'images' : [],
    'categories' : [],
    'annotations' : []
    }

    anno_id = 1
    label_idx = {}
    with open(classname_path, 'r') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines): label_idx[l.replace('\n','')]=idx

    for label_names in label_idx.keys():
        single_category = {'id': label_idx[label_names], 'name': label_names, 'supercategory': label_names}
        gt_dict['categories'].append(single_category)

    
    for idx, data in enumerate(dataset):
    
        img_info = {}
        img_info['id'] = idx+1
        img_info['width'] = 800
        img_info['height'] = 800

        img = tf.image.decode_jpeg(data['image/encoded'], channels=3)
        img_info['img_raw'] = img
        
        gt_dict['images'].append(img_info)

        class_text = tf.sparse.to_dense(data['image/object/class/text'], default_value='')
        labels = tf.cast(class_table.lookup(class_text), tf.float32)

        bbox = tf.stack([tf.sparse.to_dense(data['image/object/bbox/xmin']),
                         tf.sparse.to_dense(data['image/object/bbox/ymin']),
                         tf.sparse.to_dense(data['image/object/bbox/xmax']),
                         tf.sparse.to_dense(data['image/object/bbox/ymax']),
                         labels], axis=1)

        for x1,y1,x2,y2,label in bbox.numpy():
            x1,y1,x2,y2 = list(map(int,np.array([x1,y1,x2,y2])*input_size))
            width, height = x2-x1, y2-y1

            annotations = {}
            annotations['category_id'] = int(label)
            annotations['bbox'] = [x1, y1, width, height]
            annotations['area'] = width*height

            annotations['image_id'] = idx+1

            annotations['iscrowd'] = 0
            annotations['ignore'] = 0
            annotations['id'] = anno_id
            anno_id +=1

            gt_dict['annotations'].append(annotations)
    
    return gt_dict