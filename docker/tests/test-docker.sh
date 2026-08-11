#!/bin/bash

# note:
# 1. model: the model folder should be mounted
# 2. shm-size: it needs larger than default shm size

# unit test
docker run --rm --gpus all \
  -w /opt/kvcc \
  kvcc-custom-runtime:dev \
  python3 -m pytest \
    tests/v1/kv_offload/tiering/test_kvcc_tier.py

# e2e test
docker run --rm --gpus all \
  -v /images/models:/images/models \
  --shm-size=8g \
  -w /opt/kvcc \
  kvcc-custom-runtime:dev \
  python3 -m pytest \
    tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_regression_transfer.py \
    --model /images/models/Qwen/Qwen3-0.6B-FP8 \
    --gpus 0,1 \
    --eager-ctrl-connect true \
    --partial-submit-load 0.6 \
    -s -vv
