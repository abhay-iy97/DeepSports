for i in 01 02 03 04 07 09 10 13 14 17 18 22 26;
do
mkdir whole_videos_frames/$i
ffmpeg -i ~/test/whole_videos/$i.mp4 whole_videos_frames/$i/%06d.jpg
done
