# Requires framesXX.txt files in the same directory as this file.
# Requires the extracted frames in the following directory local to this file: whole_videos_frames/XX

import os

# reads in the framesXX.txt files which has the start-end frames for each video, then uses this to build a set of all used frames
def get_used_frames(filename, verbose=True):
    used_frames = set([])
    with open(filename, "r") as f:
        for line_num, line in enumerate(f.readlines()):
            split = line.split(",")
            if len(split) != 2:
                if verbose:
                    print(f"[Info] Skipping line {line_num} as it had more than two columns: {split}")
                    continue
            try:
                start_frame = int(split[0])
                end_frame = int(split[1])
                used_frames.update(range(start_frame, end_frame+1))
            except:
                if verbose:
                    print(f"[Info] Skipping line {line_num} as it failed frame values parsing: {split}")
    return used_frames

# given a directory (i.e. whole_videos_frames/26) and the used_frames set from get_used_frames(), it will remove any frame not found in the used set
def remove_unused_frames(directory, used_frames):
    print(f"Processing {directory}...")
    total_files = 0
    removed_files = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            frame = int(filename.replace(".jpg", ""))
            total_files += 1
            if frame not in used_frames:
                removed_files += 1
                os.remove(f)
            if total_files % 10000 == 0:
                print(f"\tProcessed {total_files} files so far...")
    print(f"\n\tDone! {directory} had {total_files} frames, removed {100.0 * removed_files / total_files:.2f}% unused frames.\n")
    
def main():
    # To run them all:
    for i in ["01", "02", "03", "04", "07", "09", "10", "13", "14", "17", "18", "22", "26"]:
        used_frames = get_used_frames(f"frames{i}.txt")
        remove_unused_frames(f"whole_videos_frames/{i}", used_frames)
        
if __name__ == "__main__":
    main()