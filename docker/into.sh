#!/bin/bash

docker exec --user "docker_semseg" -it vlsat_yolov8_seg \
        /bin/bash -c "source /opt/ros/humble/setup.bash; cd /home/docker_semseg_vlsat; /bin/bash"
