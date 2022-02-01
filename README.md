# YOLOv3-tf

## 구현내용
- YOLOV3 Detector
- Mosaic Augmentation
- Focal Loss

<br><br>

## 디렉토리 구조
- configs : config 파일이 저장되는 경로

<br>

- ipynb
  - **coco_eval.ipynb** : coco evaluation 코드
  - **data_utils.ipynb** : raw 데이터를 학습 가능한 형태의 데이터 format으로 변환 및 유효성 검증 코드
  - **model_export.ipynb** : 모델 export를 위한 코드
  - **train_debug.ipynb** : 즉시 실행모드로 모델을 학습하는 코드로 학습과정의 디버깅을 위한 코드
  - **vis_tfrecord.ipynb** : tfrecord 데이터 시각화를 위한 코드
  - **anchor_optimization.ipynb** : anchor optimization을 위한 코드

  <br>

- weights : 학습 weight 저장 경로
<br>

- yolov3
  - **anchor.py** : anchor 관련 소스코드
  - **dataset.py** : 데이터셋 로드 관련 코드
  - **augmentation_pipeline.py** : augmentation pipeline 소스코드
  - **loss.py** :  loss관련 소스코드
  - **metrics.py** : 모델 평가 metric 코드
  - **models.py** : 모델의 구조가 정의된 코드
  - **utils.py** : 기타 uitilty 함수 모음

<br>

- configuration.ipynb : 모델 학습을 위한 config 정의
- eval.py : evaluation 소스코드
- train.py : 모델 학습을 위한 소스코드 
  

<br><br>

  ## 소스코드 실행방법
  - configuration.ipynb 파일을 실행하여 모델 학습을 위한 config 파일(.json) 생성
  - 아래의 실행 명령어를 통해 생성된 config 파일을 인자로 전달
    - CUDA_VISIBLE_DEVICES=0 python train.py --config_path=configs/default.json
    
  - 데이터 저장경로(하이큐브 경로)
    - /home/jovyan/DATA/googlemap/tfrecord


<br><br>

## 모델 실험 파라미터

|**config name**|default|
|:----:|:----:|
|**Summary**|초기 weight 없이 학습|
|**batch size**|8|
|**epoch**|1000|
|**anchor optimization**|False|
|**augmentation**|False|
|**augmentation 실행횟수**|None|
|**transfer learning**|Darknet backbone trained from Pascal|
|**backbone freezing**|True|
|**lr policy**|plateau|
|**initial learning rate**|0.001|
|**training ignore iou**|0.5|
|**nms_max_box**|100|
|**nms_iou_threshold**|0.5|
|**nms_score_threshold**|0.5|
|**optimizer**|adam|

<br><br>

## References
- https://github.com/zzh8829/yolov3-tf2
- https://github.com/ultralytics/yolov3
