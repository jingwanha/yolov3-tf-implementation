from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K

import numpy as np
import os

import sys
sys.path.append('../../')
sys.path.append('../')

from yolov3.models import YoloV3
from yolov3.dataset import transform_images, create_gt_from_tfrecord
from yolov3.anchors import default_anchor, anchor_masks

from eval import ap_per_classes, predict_for_eval,evaluation_model_by_mAP
    
class AP_Calculation(Callback):

    def __init__(self, config, save_path, warmup_epoch = 5):        
        self.eval_model = None
        self.train_gt_dict = None
        self.eval_gt_dict = None
        
        self.best_mAP = -np.Inf
        self.best_loss = np.Inf
        
        self.config=config
        
        self.save_path = save_path
        self.warmup_epoch = warmup_epoch
        self.monitor = 'loss'
        
        self.eval_data_path = os.path.join(config['data']['root_path'],config['data']['test_tfrecord'])
        self.train_data_path = os.path.join(config['data']['root_path'],config['data']['train_tfrecord'])
        
        self.classname_path = os.path.join(config['data']['root_path'],config['data']['classes_names'])
        self.input_size = config['data']['input_size']
        

    def print_mAP(self,result, epoch, file_name, monitor_factor):
        
        classname_dict = {}
        classname_dict[self.config['data']['n_classes']] = 'ALL'
        with open(self.classname_path,'r') as f: 
            for idx, l in enumerate(f.readlines()):
                classname_dict[idx] = l.replace('\n','')

        print("{}\t{}\t{}\n".format('CLASS','mAP0.5','mAP0.5:0.95'))
        with open(os.path.join(self.config['training']['save_path'],self.config['model_name'],file_name),'a') as f:
            f.write("{}\t{}\t{}\t\tepoch:{}\n".format('CLASS','mAP0.5','mAP0.5:0.95',epoch))
            
            for category_id, mAP05, mAP in result:
                f.write("{}\t{}\t{}\n".format(classname_dict[category_id],round(mAP05,3),round(mAP,3)))
                print("{}\t{}\t{}".format(classname_dict[category_id],round(mAP05,3),round(mAP,3)))
            
            if monitor_factor == 'mAP':
                print("Current Best mAP : {}\n".format(self.best_mAP))
                f.write("Current Best mAP : {}".format(self.best_mAP))
                f.write("\n\n")
                
            elif monitor_factor == 'loss':
                print("Loss : {},  mAP : {}\n".format(self.best_loss, result[self.config['data']['n_classes']][1]))
                f.write("Loss : {},  mAP : {}\n".format(self.best_loss, result[self.config['data']['n_classes']][1]))
                f.write("\n\n")
    
    def on_train_begin(self, logs=None):
        
        if self.config['data']['anchor_optimization']:
            anchor_path = os.path.join(self.config['data']['root_path'],'optimized_anchor.npy')
            anchors = np.load(anchor_path)
                
        else : anchors = default_anchor
        
        self.eval_gt_dict = create_gt_from_tfrecord(self.eval_data_path, self.classname_path ,self.input_size)        
        self.eval_model = YoloV3(self.config, anchors, classes=self.config['data']['n_classes'])
    
    def on_epoch_end(self, epoch, logs=None):
            
        if epoch >= self.warmup_epoch:
            # weight 초기화
            self.eval_model.set_weights(self.model.get_weights())

            # prediction result 저장 & Evaluation
            pred_anno = predict_for_eval(self.eval_model, self.eval_gt_dict, self.input_size, self.classname_path)        
            result = ap_per_classes(self.eval_gt_dict, pred_anno, self.config['data']['n_classes'])

            # 모델 Save
            curr_mAP = result[self.config['data']['n_classes']][1]
            if self.best_mAP < curr_mAP:
                print ("mAP is improved from {} to {}".format(self.best_mAP,curr_mAP))
                self.model.save_weights(self.save_path,overwrite=True)
                self.best_mAP = curr_mAP

            print("TESTSET mAP")
            self.print_mAP(result, epoch, file_name='mAP_history.txt',monitor_factor='mAP')