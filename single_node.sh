#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    --master_port  $RANDOM \
    ss_baselines/savi/run.py \
    --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
    --model-dir data/models/savi_dummy
