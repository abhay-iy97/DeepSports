# For help type 'python filter_frames.py -h' in the commandline.
# Example: python3 filter_frames.py -log INFO -root path\to\whole_videos_frames -ann path\to\final_annotations_dict.pkl -vid all

import pickle
import os
from pathlib import Path
import argparse
import logging
import sys
import math
from typing import Dict, List, Set, Tuple, Union


# Annotation Alias, matches the format of the contents in the annotations .pkl file
#   key = tuple of video name and clip number
#   value = json data corresponding to key
Annotation = Dict[Tuple[int, int], Dict[str, float]]


# UsedFrames Alias, the processed annotations
#   key = video name
#   value = set of all used frames
UsedFrames = Dict[str, Set[int]]


# Converts the size in bytes to an appropriate representation (KB, MB, GB, etc..)
def convert_size(size_bytes: float) -> str:
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


# Loads and returns the final annotations dictionary pickle file from the specified filepath.
def load_annotations(filepath: str) -> Annotation:
    logging.info(f"Loading annotations from {filepath}")
    
    if filepath == None:
        logging.error(f"Annotations file path is invalid: {filepath}")
        sys.exit(1)
    
    if not os.path.isfile(filepath):
        logging.error(f"Annotations file does not exist at {filepath}")
        sys.exit(2)
    
    if not filepath.lower().endswith(".pkl"):
        logging.error(f"Annotations file is expected to have a .pkl extension, but was given {filepath}")
        sys.exit(3)
    
    with open(filepath, "rb") as f:
        try:
            data = pickle.load(f)
            return data
        except Exception as ex:
            logging.error(f"Failed to load annotations file at {filepath} with reason {ex}")
            sys.exit(4)


# Returns a dictionary of unused frames, where the key is the video name, and the value is a set of all used frames.
def parse_used_frames_from_annotations(annotations: Annotation) -> UsedFrames:
    logging.info("Parsing frames from annotations...")
    
    if annotations == None:
        logging.error(f"Failed to parse used frames from annotations as it was None")
        sys.exit(5)
    
    try:
        used_frames = {}
        for (key, value) in annotations.items():
            video_num = f"{key[0]:02d}"
            start_frame = value["start_frame"]
            end_frame = value["end_frame"]
            if video_num not in used_frames:
                used_frames[video_num] = set([])
            used_frames[video_num].update(range(start_frame, end_frame + 1))
        return used_frames
    except Exception as ex:
        logging.error(f"Failed to parse annotations with reason {ex}")
        sys.exit(6)


# Removes any unused frames in the given directories. (unused frame is one that is not present in the annotations)
# video_directories can be "all" to process all videos in the root_directory, or a list of video directories
def remove_unused_frames(root_directory: str, used_frames: UsedFrames, video_directories: Union[str, List[str]]) -> None:
    logging.info(f"Removing unused frames from {root_directory}...")
    
    if root_directory == None:
        logging.error(f"Root directory is invalid: {root_directory}")
        sys.exit(7)
        
    if not os.path.isdir(root_directory):
        logging.error(f"Root directory does not exist at {root_directory}")
        sys.exit(8)
        
    if used_frames == None:
        logging.error(f"The parsed used frames is None")
        sys.exit(9)
        
    if video_directories[0] == "all":
        video_directories: List[str] = ["01", "02", "03", "04", "05", "06", "07", "09", "10", "13", "14", "17", "18", "22", "26"]
    
    total_stats = {"total_files": 0, "removed_files": 0, "total_size": 0, "removed_size": 0}
    for video_dir in video_directories:
        video_dir_path = os.path.join(root_directory, video_dir)
        logging.info(f"Removing unused frames in {video_dir_path}")
        video_stats = {"total_files": 0, "removed_files": 0, "total_size": 0, "removed_size": 0}
        
        # Process each .jpg file in directory
        try:
            for filename in os.listdir(video_dir_path):
                f = os.path.join(video_dir_path, filename)
                if os.path.isfile(f) and f.lower().endswith('.jpg'):
                    frame_num = int(filename.replace(".jpg", ""))
                    file_size = os.path.getsize(f)
                    video_stats["total_files"] += 1
                    video_stats["total_size"] += file_size
                    if frame_num not in used_frames[video_dir]:
                        os.remove(f)
                        video_stats["removed_files"] += 1
                        video_stats["removed_size"] += file_size
                if video_stats["total_files"] % 10000 == 0:
                    logging.info(f"\tProcessed {video_stats['total_files']} JPG files")
        except Exception as ex:
            logging.error(f"Failed to remove unused frames at {video_dir_path} with reason {ex}")
            sys.exit(10)
            
        # Accumulate statistics
        total_stats["total_files"] += video_stats["total_files"]
        total_stats["removed_files"] += video_stats["removed_files"]
        total_stats["total_size"] += video_stats["total_size"]
        total_stats["removed_size"] += video_stats["removed_size"]
        
        # Display current video's statistics
        removed_files_pct = 100 * video_stats["removed_files"] / max(1, video_stats["total_files"])
        removed_size_pct = 100 * video_stats["removed_size"] / max(1, video_stats["total_size"])
        logging.info(f"\tPrevious: {video_stats['total_files']} ({convert_size(video_stats['total_size'])})"\
                     f"  Removed: {video_stats['removed_files']} ({convert_size(video_stats['removed_size'])})"\
                     f"  Removed%: {removed_files_pct:.1f}% ({removed_size_pct:.1f}% bytes)"\
                     f"  Current: {video_stats['total_files'] - video_stats['removed_files']}"\
                     f" ({convert_size(video_stats['total_size'] - video_stats['removed_size'])})"
                     )
        
    # Display overall statistics
    removed_files_pct = 100 * total_stats["removed_files"] / max(1, total_stats["total_files"])
    removed_size_pct = 100 * total_stats["removed_size"] / max(1, total_stats["total_size"])
    logging.info(f"Overall: "\
                 f"  Previous: {total_stats['total_files']} ({convert_size(total_stats['total_size'])})"\
                 f"  Removed: {total_stats['removed_files']} ({convert_size(total_stats['removed_size'])})"\
                 f"  Removed%: {removed_files_pct:.1f}% ({removed_size_pct:.1f}% bytes)"\
                 f"  Current: {total_stats['total_files'] - total_stats['removed_files']}"\
                 f" ({convert_size(total_stats['total_size'] - total_stats['removed_size'])})"
                 )


# Program's main entry point
def main():
    parser = argparse.ArgumentParser(description="Remove unused diving frames in MTL-AQA dataset.")
    
    parser.add_argument(
        "-log",
        "--loglevel",
        type=str,
        help="The level of logging in the application. Default value is 'INFO'. Possible values: [DEBUG, INFO, WARNING, ERROR, CRITICAL].",
        default="INFO",
        required=False
    )
    
    parser.add_argument(
        "-root",
        "--root_dir",
        type=str,
        help="The path to the root directory containing the frame images. Default value is './', the same directory of this python script.",
        default="./",
        required=False
    )
    
    parser.add_argument(
        "-ann",
        "--annotation_path",
        type=str,
        help="The path to the final annotations dict pickle file. Default value is './final_annotations_dict.pkl', in the same directory of this python script.",
        default="./final_annotations_dict.pkl",
        required=False
    )
    
    parser.add_argument(
        "-vid",
        "--videos",
        type=str,
        nargs="+",
        help="The videos directory name to filter the frames for. Default value is 'all', which will filter the frames from all videos present. To filter select directories, only list their directory names, i.e. '-vid 01 02 03'",
        default=["all"],
        required=False
    )
    
    # Parse arguments
    args = vars(parser.parse_args())
    arg_loglevel = args["loglevel"]
    arg_root_dir = args["root_dir"]
    arg_annotation_path = args["annotation_path"]
    arg_videos = [x.lower() for x in args["videos"]]
    
    # Get directory of this script
    directory_path = Path(__file__).parent.absolute()
    
    # Set the defaults of the commandline arguments to be relative to current directory
    if arg_annotation_path == "./final_annotations_dict.pkl":
        arg_annotation_path = os.path.join(directory_path, "final_annotations_dict.pkl")
    if arg_root_dir == "./":
        arg_root_dir = directory_path
    
    # Prepend a timestamp to each logging
    numeric_loglevel = getattr(logging, arg_loglevel.upper(), None)
    if not isinstance(numeric_loglevel, int):
        raise ValueError('Invalid log level: %s' % numeric_loglevel)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=numeric_loglevel,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Program logic
    annotations = load_annotations(arg_annotation_path)
    used_frames = parse_used_frames_from_annotations(annotations)
    remove_unused_frames(arg_root_dir, used_frames, arg_videos)
    logging.info("Filter framing finished executing")
    

if __name__ == "__main__":
    main()