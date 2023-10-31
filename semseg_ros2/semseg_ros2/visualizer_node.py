import rclpy
import cv2
import numpy as np
import message_filters

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from yolov8_seg_interfaces.msg import Box, Mask, Objects


class VisualizerNode(Node):

    def __init__(self):
        super().__init__('visualizer_node')

        self.colors_palette = {
            1: {'bounds': (7,   7,   132), 'inner': (36,  77,  201)}, # firehose
            2: {'bounds': (158, 18,  6  ), 'inner': (196, 48,  35 )}, # hose
            3: {'bounds': (96,  12,  107), 'inner': (214, 80,  229)}, # waste
            4: {'bounds': (112, 82,  0  ), 'inner': (255, 208, 79 )}, # puddle
            5: {'bounds': (163, 0,   68 ), 'inner': (244, 88,  153)}, # breakroad
            }
        
        # self.counter = 0

        image_sub = message_filters.Subscriber(self, Image, '/sensum/left/image_raw')
        segmentation_sub = message_filters.Subscriber(self, Objects, '/segmentation')

        self.ts = message_filters.TimeSynchronizer([image_sub, segmentation_sub], 10)
        self.ts.registerCallback(self.on_image_segmentation)

        self.pub_segmentation_color = self.create_publisher(Image, 'segmentation_color', 10)

        self.br = CvBridge()


    def on_image_segmentation(self, image_msg : Image, segm_msg : Objects):
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        # image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

        segmentation_color = self.draw_masks(image, segm_msg.masks, segm_msg.classes_ids)
        # cv2.imwrite(
        #     f'/home/docker_semseg/colcon_ws/src/semseg_ros2/media/test2/{self.counter}.jpg',
        #     segmentation_color)
        # self.counter += 1

        segm_color_msg = self.br.cv2_to_imgmsg(segmentation_color, 'rgb8')
        segm_color_msg.header = segm_msg.header

        self.pub_segmentation_color.publish(segm_color_msg)

    def draw_masks(self, image, masks, classes):
        shape = image.shape
        line_thickness = max(1, int((shape[0] + shape[1]) / 2) // 400)

        for cls, mask in zip(classes, masks):
            mask = mask.mask_poly
            if not mask:
                continue
            mask = np.array(mask).reshape(len(mask) // 2, 2).tolist()

            mask_pattern = np.zeros(image.shape)
            cv2.fillPoly(
                mask_pattern,
                np.array([mask]),
                color=self.colors_palette[cls]['inner']
                )
            image[mask_pattern > 0] = 0.5 * image[mask_pattern > 0] + 0.5 * mask_pattern[mask_pattern > 0]

            cv2.polylines(image, np.array([mask]), isClosed=True,
                        color=self.colors_palette[cls]['bounds'],
                        thickness=line_thickness)

        return image


def main(args=None):
    rclpy.init(args=args)

    node = VisualizerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
