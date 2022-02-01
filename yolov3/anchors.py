import numpy as np
import json

default_anchor = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],np.float32)
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def anchor_optimization(trrecord_dataset):
    data = []
    label = []
    anchor_first_result = []
    
    for x,y in trrecord_dataset.as_numpy_iterator():
        for label in y:
            if np.sum(label) ==0 : break
            label = list(map(int,label[:-1]*800))
            anchor_first_result.append([label[2]-label[0], label[3]-label[1]])
            
    anchors, avg_iou = get_kmeans(np.array(anchor_first_result), 9)
    print("AVG IOU : {}".format(avg_iou))
    
    return np.array(anchors,np.float32)

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    return iou_

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    
    # 전체 배열에 동일한 행이 포함되어 있으면 Forgy 메서드가 실패합니다.
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou