import os

import cv2
import numpy as np

import torch
print(torch.cuda.is_available())

from yolov8_seg_interfaces.msg import Box, Mask, Objects

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/docker_semseg/colcon_ws/src/semseg_ros2/weights/yolov8n-seg.pt')
    img = '/home/docker_semseg/colcon_ws/src/semseg_ros2/media/frame000441.jpg'
    predictions = model(img, device='cpu', retina_masks=True)[0]

    conf = predictions.boxes.conf.cpu().numpy().astype(np.float32).tolist()

    classes = predictions.boxes.cls.cpu().numpy().astype(np.uint8).tolist()

    boxes = predictions.boxes.xywh.cpu().numpy() # x_c, y_c, w, h 
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes = boxes.astype(np.uint32).tolist()
    obj_boxes = []
    for box in boxes:
        obj_box = Box()
        obj_box.x, obj_box.y, obj_box.w, obj_box.h = box
        obj_boxes.append(obj_box)

    masks = predictions.masks.xy
    obj_masks = []
    for mask in masks:
        obj_mask = Mask()
        obj_mask.mask_poly = mask.astype(np.uint32).ravel().tolist()
        obj_masks.append(obj_mask)

    objects = Objects()
    objects.scores = conf
    objects.classes_ids = classes
    objects.boxes = obj_boxes
    objects.masks = obj_masks

    # print(classes)
    # print(obj_boxes)
    # print(type(masks))

    # annotated_img = predictions.plot()

    # cv2.imwrite('/home/docker_semseg/colcon_ws/src/semseg_ros2/media/frame000441_pred.jpg', annotated_img)
