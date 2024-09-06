#!/bin/bash

docker exec --user "docker_semseg" -it yolov8_seg \
        /bin/bash -c "source /opt/ros/humble/setup.bash; cd /home/docker_semseg; /bin/bash"
