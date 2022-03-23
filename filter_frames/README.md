# How to Run
1. Perform frame extraction on the videos using MTL-AQA's code [here](https://github.com/ParitoshParmar/MTL-AQA).
2. Store the frames to **whole_videos_frames/XX** where XX is the video number (01, 02, .. 26)
3. Copy the contents of this directory to be adjacent to whole_videos_frames, as shown below:

- filter_frames.py
- framesXX.txt
- whole_videos_frames/
 
4. Run filter_frames.py from the command line using:

```
python3 filter_frames.py
```