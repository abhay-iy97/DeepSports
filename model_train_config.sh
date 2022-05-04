#!/bin/bash
cd "/home1/$USER/DeepSports/training/TimeSformer"
python3 train.py --loglevel INFO \
    --root_dir "/scratch1/$USER/DeepSports_dataset/whole_videos_frames" \
    --output "/home1/$USER/DeepSports/losses.png" \
    --evaluate True \
    --pretrained_model "./TimeSformer_divST_96x32_224_HowTo100M.pyth" \
    --freeze False \
    --train_val_split_ratio 0.8 \
    --batch_size 8 \
    --epochs 200 \
    --spatial_size 224 \
    --normalize True \
    --data_aug True \
    --frame_method spaced_varied_new \
    --frame_num 48 \
    --optimizer AdamW \
    --amsgrad True \
    --learning_rate 0.00001 \
    --weight_decay 0.00001 \
    --momentum 0.9 \
    --activation LeakyReLU \
    --loss_mse_weight 1 \
    --loss_spcoef_weight 0 \
    --use_decoder False \
    --dropout 0.2 0.2 \
    --topology 512 512