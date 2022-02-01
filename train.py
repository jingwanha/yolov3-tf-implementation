import os 
import time
import json
import argparse

import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

from yolov3.models import YoloV3
from yolov3.loss import YoloLoss
from yolov3.utils import freeze_all,plot_history
from yolov3.dataset import load_tfrecord_dataset, transform_images,transform_targets, data_augmentation, padding_
from yolov3.callbacks.clr_callback import CyclicLR
from yolov3.callbacks.map_callback import AP_Calculation
from yolov3.anchors import default_anchor, anchor_optimization, anchor_masks

from eval import evaluation_model_by_mAP

def build_model(config, anchors):
    model = YoloV3(config, anchors, training='train', classes=config['data']['n_classes'])
    
    if config['model']['transfer_weight']:
        model_pretrained = YoloV3(config, anchors, training='train', classes=config['model']['transfer_classes'])
        model_pretrained.load_weights(config['model']['transfer_weight'])
        
        model.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())

        if config['model']['backbone_freeze'] : 
            print("BACKBONE (darknet) Freeze")
            freeze_all(model.get_layer('yolo_darknet'))
            
    
    if config['training']['optimizer']=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'],
                                             beta_1=0.9)
        
    elif config['training']['optimizer']=='SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['training']['learning_rate'],
                                            momentum=0.98,
                                            nesterov=True)
    
    loss = [YoloLoss(anchors[mask], 
                     input_size = config['data']['input_size'],
                     classes=config['data']['n_classes'],
                     smoothing_factor = config['training']['smoothing_factor'],
                     ignore_thresh=config['training']['ignore_iou_threshold']) for mask in anchor_masks]
    
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

if __name__=='__main__': 
    
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',help='config file path (json file format)')
    
    args = parser.parse_args()
    config_path = args.config_path
    
    # config load
    with open(config_path,'r') as f: config = json.load(f)
    
    # GPU Setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:print(e)
    
    # Train dataset
    train_data_path = os.path.join(config['data']['root_path'],config['data']['train_tfrecord'])
    classname_path = os.path.join(config['data']['root_path'],config['data']['classes_names'])
    input_size = config['data']['input_size']

    train_dataset = load_tfrecord_dataset(train_data_path, classname_path, input_size)  
    
    # Anchor Optimization
    if config['data']['anchor_optimization']:
        if os.path.isfile(os.path.join(config['data']['root_path'],'optimized_anchor.npy')):
            anchors = np.load(os.path.join(config['data']['root_path'],'optimized_anchor.npy'))
            
        else:
            anchors = anchor_optimization(train_dataset)
            np.save(os.path.join(config['data']['root_path'],'optimized_anchor.npy'),anchors)

    else : anchors = default_anchor
            
    
    # Image Augmentation 
    if config['data']['augmentation']:
        print("DATA AUGMENTATION...")
        train_dataset = data_augmentation(train_dataset, config['data']['n_augment'])
    
    
    train_dataset = train_dataset.map(lambda x,y : padding_(config['nms']['max_boxes'],x,y))
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(config['training']['batch_size'])
    
    # 모자이크 적용할 경우 아래의 map 함수를 변경
    train_dataset = train_dataset.map(lambda x, y: (transform_images(x, input_size),transform_targets(y, anchors, input_size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # valid dataset load
    val_dataset = None
    if config['data']['valid_tfrecord'] : 
        valid_data_path = os.path.join(config['data']['root_path'],config['data']['valid_tfrecord'])
        val_dataset = load_tfrecord_dataset(valid_data_path, classname_path, input_size)
        val_dataset = val_dataset.map(lambda x,y : padding_(config['nms']['max_boxes'],x,y))
        val_dataset = val_dataset.batch(config['training']['batch_size'])
        val_dataset = val_dataset.map(lambda x, y: (transform_images(x, input_size), transform_targets(y, anchors, input_size)))
        
        print("Validation Dataset are Loaded")
            
    # Model Load & Training
    model = build_model(config,anchors)
    
    monitor_factor = 'val_loss' if val_dataset else 'loss'   
    
    # callbacks setup
    callbacks = [EarlyStopping(monitor='loss', patience=10, verbose=1),
                 ModelCheckpoint(os.path.join(config['training']['save_path'], 
                                              config['model_name'],'valloss','trained_valloss_best.tf'),
                                 monitor = 'val_loss', mode = 'min',verbose=1, save_weights_only=True,save_best_only= True),
                 
                 ModelCheckpoint(os.path.join(config['training']['save_path'],
                                              config['model_name'],'loss','trained_loss_best.tf'), 
                                 monitor = 'loss', mode = 'min',verbose=1, save_weights_only=True,save_best_only= True),
                
                AP_Calculation(config,
                              os.path.join(config['training']['save_path'],config['model_name'],'trained_mAP_best.tf'),
                              warmup_epoch=5)]
    
    if config['training']['lr_policy'] == 'clr' :  
        clr = CyclicLR(mode='exp_range', # triangular, triangular2, exp_range
                       base_lr=0.00001,
                       max_lr=0.01,
                       step_size=200)
        callbacks.append(clr)
        
        
    elif config['training']['lr_policy'] == 'plateau':
        plateau = ReduceLROnPlateau(monitor=monitor_factor,
                                    verbose=1, 
                                    min_lr = 0.0, 
                                    factor=0.1, 
                                    min_delta = 0.0001, 
                                    patience=10)
        callbacks.append(plateau)

    history = model.fit(train_dataset,
                        epochs=config['training']['epoch'],
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        verbose=1)
    
    # history draw
    history_path =  os.path.join(config['training']['save_path'],config['model_name'],'logs')
    try : os.mkdir(history_path)
    except: pass
    history_save_path = os.path.join(history_path,'loss.png')
    plot_history(history, history_save_path)

    # Evaluation    
    eval_model = YoloV3(config,anchors, classes=config['data']['n_classes'])
    export_model = YoloV3(config,anchors, classes=config['data']['n_classes'],training='export')
    
    weight_path = os.path.join(config['training']['save_path'], config['model_name'],'trained_mAP_best.tf')
    eval_model.load_weights(weight_path).expect_partial()
    export_model.load_weights(weight_path).expect_partial()
    
    # prediction result save 
    img_save_path = os.path.join(config['training']['save_path'],config['model_name'],'logs',config['data']['save_test_image_path'])
    try:os.mkdir(img_save_path)
    except:pass
    result = evaluation_model_by_mAP(eval_model, config, save_path=img_save_path)
    
    # Model Export
    print("{} model export".format(weight_path))
    if config['training']['export_path'] is not None:
        exported_path = os.path.join(config['training']['export_path'],config['model_name'])
        export_model.save(exported_path,include_optimizer=True)