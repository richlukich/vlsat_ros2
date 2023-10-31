#!/bin/bash
docker exec --user "docker_semseg" -it semseg \
    /bin/bash -c "cd /home/docker_semseg; /bin/bash"