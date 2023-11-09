#!/bin/bash
docker exec --user "docker_yolov8_seg" -it yolov8_seg \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; source /opt/ros/foxy/setup.bash; cd /home/docker_yolov8_seg; /bin/bash"