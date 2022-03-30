#!/bin/bash
mkdir -p ~/test/whole_videos
echo '~/test/whole_videos CREATED'


cp ./download_videos.sh -t ~/test/whole_videos
echo './download_videos.sh COPIED to ~/test/whole_videos'


cd ~/test/whole_videos
sh download_videos.sh
echo './download_videos.sh executed'

cd ~/DeepSports/filter_frames
echo 'Back in ~/DeepSports/filter_frames'

mkdir -p /scratch1/<username>/deepsports_dataset/whole_videos_frames
echo '/scratch1/<username>/deepsports_dataset/whole_videos_frames CREATED'

cp ./my.job -t /scratch1/<username>/deepsports_dataset/
echo './my.job COPIED to /scratch1/<username>/deepsports_dataset/'

cp ./frame_extractor.sh -t /scratch1/<username>/deepsports_dataset/
echo './frame_extractor.sh COPIED to /scratch1/<username>/deepsports_dataset/'

cp filter_frames.py frames* -t /scratch1/<username>/deepsports_dataset/
echo 'filter_frames.py and frames* COPIED to /scratch1/<username>/deepsports_dataset/'

cp final_annotations_dict.pkl -t /scratch1/<username>/deepsports_dataset/
echo 'final_annotations_dict.pkl COPIED to /scratch1/<username>/deepsports_dataset/'


cd /scratch1/<username>/deepsports_dataset/
touch job_id.txt
sbatch my.job > job_id.txt
echo 'Batch SCHEDULED'

