#!/bin/bash

# How to generate key: https://stackoverflow.com/a/53524211/5258187
SSH_KEY=~/.ssh/gcloud-test-ml

# instance-1
#HOST_ADDR=34.83.191.160

# gpu-preemptible-1
#HOST_ADDR=35.197.30.148

# gpu-preemptible-2
#HOST_ADDR=34.83.72.35

# gpu-preemptible-3
#HOST_ADDR=35.212.178.217

# gpu-preemptible-4
#HOST_ADDR=35.212.176.78

echo $HOST_ADDR
$HOST_ADDR && exit 1

./build.sh

# Serialized numpy inputs
#scp -i $SSH_KEY npydata straian@$HOST_ADDR:.

rm -fr logs/logs-$HOST_ADDR-old
rm -fr outputs/outputs-$HOST_ADDR-old
rm -fr charts/charts-$HOST_ADDR-old
rm -fr checkpoints/checkpoints-$HOST_ADDR-old
cp -r logs/logs-$HOST_ADDR logs/logs-$HOST_ADDR-old
cp -r outputs/outputs-$HOST_ADDR outputs/outputs-$HOST_ADDR-old
cp -r charts/charts-$HOST_ADDR charts/charts-$HOST_ADDR-old
cp -r checkpoints/checkpoints-$HOST_ADDR checkpoints/checkpoints-$HOST_ADDR-old
rm -fr logs/logs-$HOST_ADDR
rm -fr outputs/outputs-$HOST_ADDR
rm -fr charts/charts-$HOST_ADDR
rm -fr checkpoints/checkpoints-$HOST_ADDR

scp -i $SSH_KEY docker-run.sh straian@$HOST_ADDR:.
ssh -i $SSH_KEY straian@$HOST_ADDR bash docker-run.sh
scp -r -i $SSH_KEY straian@$HOST_ADDR:logs logs/logs-$HOST_ADDR
scp -r -i $SSH_KEY straian@$HOST_ADDR:outputs outputs/outputs-$HOST_ADDR
scp -r -i $SSH_KEY straian@$HOST_ADDR:charts charts/charts-$HOST_ADDR
scp -r -i $SSH_KEY straian@$HOST_ADDR:checkpoints checkpoints/checkpoints-$HOST_ADDR
open charts-$HOST_ADDR/*.png

