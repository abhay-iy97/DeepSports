#!/bin/bash
cd "/home1/$USER/DeepSports/training/TimeSformer"
python3 train.py --loglevel INFO \
    --root_dir "/scratch1/$USER/whole_videos_frames" \
    --evaluate True \
    --output "/home1/$USER/DeepSports/losses.png"
    --pretrained_model ./TimeSformer_divST_8x32_224_K400.pyth \
    --batch_size 4 \
    --epochs 5 \
    --learning_rate 0.00001 \
    --weight_decay 0.00001 \
    --optimizer AdamW \
    --dropout 0.5 \
    --frame_method spaced_varied \
    --topology 512 256