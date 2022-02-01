import numpy as np
import tensorflow as tf
import cv2
import os 

from tqdm import tqdm

from yolov3.metrics.coco import COCO 
from yolov3.metrics.cocoeval import COCOeval
from yolov3.dataset import transform_images, create_gt_from_tfrecord
from yolov3.utils import draw_result_with_gt
        
def ap_per_classes(gt_dict, pred_anno, n_classes):
    mAP_list = []
    mAP_0595_list = []
    
    
    result = []
    
    # 아무런 객체도 검출되지 않는 경우    
    if pred_anno.shape[0]==0: 
        dummy_result = []
        for i in range(n_classes+1): dummy_result.append([i,0,0])
            
        return dummy_result
        
    for i in range(n_classes):
        pred_class = pred_anno[pred_anno[:,6]==i]

        gt_class = gt_dict.copy()
        gt_class['annotations'] = [anno for anno in gt_dict['annotations'] if anno['category_id']==i]

        if len(pred_class) != 0:
            gt = COCO(gt_class)
            pred = gt.loadRes(pred_class)
            coco_eval = COCOeval(gt, pred)    
            mAP_0595, mAP = coco_eval.stats[0:2]
            

        # 해당 클래스에 검출된 객체가 없는 경우
        else:
            mAP_0595, mAP = 0, 0

        mAP_list.append(mAP)
        mAP_0595_list.append(mAP_0595)

        result.append([i, mAP, mAP_0595])
        
    result.append([i+1, round(np.mean(mAP_list),3), round(np.mean(mAP_0595_list),3)])    
    
    return result

def predict_for_eval(model, gt_dict, input_size, classname_path, save_path=None):
    pred_results = []
    
    print("{} evaluation images are predicting".format(len(gt_dict['images'])))
    
    for gt in tqdm(gt_dict['images']):
        file_name = gt['file_name']
        img_id = gt['id']

        # img_raw = tf.image.decode_image(open(file_name, 'rb').read(), channels=3)
        img_raw = gt['img_raw']

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, input_size)

        # predict
        boxes, scores, classes, nums = model(img)
        
        pred_draw = []
        for i in range(nums[0]):         
            det_box = tf.clip_by_value(boxes[0][i], clip_value_min=0.0, clip_value_max=1.0)
            x1,y1,x2,y2 = list(map(int,det_box*input_size))

            width, height = x2-x1, y2-y1

            score = np.array(scores[0][i])
            category_id = np.array(classes[0][i])

            pred_results.append([img_id,x1,y1,width,height,score,int(category_id)])
            pred_draw.append([img_id,x1,y1,width,height,score,int(category_id)])
            
        if save_path : 
            gt_draw = [gt_anno for gt_anno in gt_dict['annotations'] if gt_anno['image_id']==img_id]
            draw_img = cv2.cvtColor(img_raw.numpy(),cv2.COLOR_BGR2RGB)
            
            draw_img = draw_result_with_gt(draw_img,gt_draw, pred_draw, classname_path)
            cv2.imwrite(os.path.join(save_path,file_name.split('/')[-1]),draw_img)
            
    return np.array(pred_results)


def evaluation_model_by_mAP(model, config, data_path=None, weight_path=None, input_size=None, save_path = None):
        
    # dataset & classnmae file 로드
    if data_path is None:
        data_path = os.path.join(config['data']['root_path'],config['data']['test_tfrecord'])
        
    classname_path = os.path.join(config['data']['root_path'],config['data']['classes_names'])
    
    if input_size is None:
        input_size = config['data']['input_size']
        
    # gt 생성
    gt_dict = create_gt_from_tfrecord(data_path, classname_path ,input_size)
    
    # gt 정보를 기반으로 이미지 prediction
    pred_anno = predict_for_eval(model, gt_dict, input_size, classname_path, save_path=save_path)

    
    # class 별 AP 계산
    mAP_result = ap_per_classes(gt_dict, pred_anno, config['data']['n_classes'])

    
    return mAP_result