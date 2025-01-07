docker build --build-arg USER_NAME="jwhwang0911" . -t dl_container

DIR=$(pwd)
docker run \
    --rm \
    --gpus all \
    -v ${DIR}/Data:/workspace/Data \
    -v ${DIR}/Code:/workspace/Code \
    -v ${DIR}/Result:/workspace/Result \
    --shm-size=8G \
    -it dl_container /bin/bash