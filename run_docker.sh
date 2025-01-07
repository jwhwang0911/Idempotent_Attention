docker build --build-arg USER_NAME="jwhwang0911" . -t dl_container
DIR=`pwd`
nvidia-docker run \
    --rm \
    -v ${DIR}/Data:/Data \
    -v ${DIR}/Code:/Code \
    -v ${DIR}/Result:/Result \
    --shm-size=8G \
    -it dl_container /bin/bash;