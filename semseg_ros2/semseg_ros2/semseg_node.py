import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ultralytics import YOLO

# from semseg.semseg import SemanticSegmentator
from semseg_ros2.inference_speed_meter import InferenceSpeedMeter

from yolov8_seg_interfaces.msg import Box, Mask, Objects


class SemSegNode(Node):

    def __init__(self) -> None:
        super().__init__('semseg_node')

        self.declare_parameter('weights',
                               '/home/docker_semseg/colcon_ws/src/semseg_ros2/weights/roboseg_L_5_cats.pt')
        self.weights = self.get_parameter('weights').get_parameter_value().string_value

        self.declare_parameter('confidence', 0.25)
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value

        self.declare_parameter('treshold', 0.5)
        self.treshold = self.get_parameter('treshold').get_parameter_value().double_value

        self.segmentator = YOLO(self.weights)

        self.br = CvBridge()

        self.sub_image = self.create_subscription(Image, '/sensum/left/image_raw', self.on_image, 10)
        self.pub_segmentation = self.create_publisher(Objects, 'segmentation', 10)

        self.speed_meter = InferenceSpeedMeter()


    def on_image(self, image_msg : Image):
        # image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        segmentation_msg = self.process_img(image)
        segmentation_msg.header = image_msg.header

        self.pub_segmentation.publish(segmentation_msg)

    def process_img(self, image:np.ndarray) -> Objects:
        self.speed_meter.start()

        predictions = self.segmentator(image, conf=self.confidence, iou=self.treshold)[0]

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

        masks = predictions.masks
        if masks is None:
            masks = np.array([])
        else:
            masks = masks.xy
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

        self.speed_meter.stop()
        
        return objects


def main(args=None):
    rclpy.init(args=args)

    node = SemSegNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
