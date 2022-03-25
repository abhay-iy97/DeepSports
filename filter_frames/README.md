# How to Run

Make changes in dataset_generator.sh
1. Replace all "\<username\>" with your credentials. Example: adiyer
2. Execute
```
sh dataset_generator.sh
```

### Result
1. It will perform frame extraction on the videos using MTL-AQA's code [here](https://github.com/ParitoshParmar/MTL-AQA).
2. Store the frames to ~/scratch1/\<username\>/deepsports_dataset/**whole_videos_frames/XX** where XX is the video number (01, 02, .. 26)
3. Perform filtering of frames.