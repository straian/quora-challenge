#!/bin/bash

CONTAINER_NAME=test-container

echo $CONTAINER_NAME

docker pull straian/quora-challenge

rm -fr logs
rm -fr outputs
rm -fr charts
rm -fr checkpoints
mkdir logs
mkdir outputs
mkdir charts
mkdir checkpoints
docker rm -f $CONTAINER_NAME
docker run --runtime=nvidia -dit --name=$CONTAINER_NAME \
    -v `pwd`/dataset:/dataset \
    -v `pwd`/model:/model \
    -v `pwd`/npydata:/npydata \
    -v `pwd`/outputs:/outputs \
    -v `pwd`/charts:/charts \
    -v `pwd`/checkpoints:/checkpoints \
    straian/quora-challenge bash

docker exec -t $CONTAINER_NAME python python/train.py
#docker exec -t $CONTAINER_NAME python python/predict.py

# Delete all but last
cd checkpoints; rm -f `ls|sort -r|awk 'NR>1'`; cd -

docker rm -f $CONTAINER_NAME

