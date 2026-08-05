#!/bin/bash

# note:
# 1. model: the model folder should be mounted
# 2. shm-size: it needs larger than default shm size
#
docker run --rm --gpus all \
  -v /images/models:/images/models \
  --shm-size=8g \
  -w /opt/kvcc \
  kvcc-custom-runtime:dev \
  python3 -m pytest \
    tests/v1/kv_offload/kvcc-tests/kvcc-e2e/validate_kvcc_e2e.py \
    --model /images/models/Qwen/Qwen3-0.6B-FP8 \
    --gpus 0,1 \
    --eager-ctrl-connect true \
    --partial-submit-load 0.6 \
    -s -vv
