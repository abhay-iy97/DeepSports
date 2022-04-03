#!/bin/bash
for i in 01 02 03 04 05 06 07 09 10 13 14 17 18 22 26;
do
mkdir -p /scratch1/$USER/DeepSports_dataset/whole_videos_frames/$i
ffmpeg -i /scratch1/$USER/DeepSports_dataset/whole_videos/$i.mp4 /scratch1/$USER/DeepSports_dataset/whole_videos_frames/$i/%06d.jpg
done