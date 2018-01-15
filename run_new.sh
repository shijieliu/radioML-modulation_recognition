#!/usr/bin/env bash

python main_new.py \
    --param "./param2" \
    --lr 0.01 \
    --weights './weights/new_accuracy65' \
    --confident_penalty True \
    --use_gpu "1"
