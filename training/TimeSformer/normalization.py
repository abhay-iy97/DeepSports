import logging
import os
import pickle
import sys
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level="INFO",
        datefmt='%Y-%m-%d %H:%M:%S')


# Annotation Alias, matches the format of the contents in the annotations .pkl file
#   key = tuple of video name and clip number
#   value = json data corresponding to key
Annotation = Dict[Tuple[int, int], Dict[str, float]]


################################## Data Preparation ##################################


class VideoClip:
    """
    A data class to hold a single video clip for the MTL-AQA dataset.
    """
    
    # Convertor between PIL images to PyTorch Tensor
    CONVERT_TENSOR = transforms.ToTensor()
    
    # The root directory path to the frames on disk (should be the one that includes the 01/ 02/ ... directories)
    FRAME_ROOT_DIR = "D:/MTL-AQA-Frames/"
    
    # Spatial dimensions that each clip will be resized to
    SPATIAL_SIZE = 224
    
    def __init__(self, video_num: int, start_frame: int, end_frame: int) -> None:
        """
        Initializes a video clip from the given metadata.

        Args:
            video_num (int): the video this clip is from (i.e. 1 for 01.mp4)
            start_frame (int): the first frame in the clip
            end_frame (int): the last frame in the clip
        """
        
        if start_frame > end_frame:
            logging.warning(f"Invalid clip for {video_num:02d}, with start_frame ({start_frame}) > end_frame {end_frame}")
        
        self.video_num = video_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.clip_num_frames = end_frame - start_frame + 1
        
    def load(self) -> torch.Tensor:
        """
        Loads the clip from disk and returns the input as a pytorch tensor:
            inputs- the video clip in the shape (channels, frames, height, width)

        Returns:
            torch.Tensor: the input video as a tensor
        """
        
        # Creates the video clip in torch with (channels, num_frames, height, width)
        clip = torch.empty(size=(3, self.clip_num_frames, self.SPATIAL_SIZE, self.SPATIAL_SIZE), dtype=torch.float64)
        
        # Load the images from disk and preprocess them into a tensor
        for idx, frame_num in enumerate(range(self.start_frame, self.end_frame + 1)):
            img_path = os.path.join(*[self.FRAME_ROOT_DIR, f"{self.video_num:02d}", f"{frame_num:06d}.jpg"])
            if not os.path.exists(img_path):
                logging.error(f"File does not exist: {img_path}. start={self.start_frame}, end={self.end_frame}")
                sys.exit()
            img = Image.open(img_path).resize((self.SPATIAL_SIZE, self.SPATIAL_SIZE))
            tensor = self.CONVERT_TENSOR(img).type(dtype=torch.float64) # (channels, height, width)
            clip[:, idx, :, :] = tensor
        
        return clip


class DivingDataset(Dataset):
    """
    Dataset class to hold the video clips from MTL-AQA and provide on-demand loading of instances from disk.
    """
    
    def __init__(self, clips_dataset: List[VideoClip]) -> None:
        """
        Initializes the dataset from the given list of video clips.

        Args:
            clips_dataset (List[VideoClip]): list of video clips in the dataset
        """
        self.clips_dataset = clips_dataset
    
    def __len__(self) -> int:
        """
        Returns:
            int: the number of clips in the dataset
        """
        return len(self.clips_dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the clip from disk at the given index and returns the inputs and the targets as a tuple of pytorch's tensor:
            inputs- the video clip in the shape (channels, frames, height, width)
            targets- the normalized score and the difficulty in the shape (2,)

        Args:
            index (int): clip index in the dataset

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the inputs and the targets as a tuple
        """
        return self.clips_dataset[index].load()


def load_annotations(filepath: str) -> Annotation:
    """
    Loads and returns the final annotations dictionary pickle file from the specified filepath.
    Annotation Alias, matches the format of the contents in the annotations .pkl file
        -key = tuple of video name and clip number
        -value = json data corresponding to key

    Args:
        filepath (str): the annotation pickle filepath

    Returns:
        Annotation: a dictionary of annotations
    """
    logging.info(f"Loading annotations from {filepath}")
    
    if filepath == None:
        logging.error(f"Annotations file path is invalid: {filepath}")
        sys.exit()
    
    if not os.path.isfile(filepath):
        logging.error(f"Annotations file does not exist at {filepath}")
        sys.exit()
    
    if not filepath.lower().endswith(".pkl"):
        logging.error(f"Annotations file is expected to have a .pkl extension, but was given {filepath}")
        sys.exit()
    
    with open(filepath, "rb") as f:
        try:
            data = pickle.load(f)
            return data
        except Exception as ex:
            logging.error(f"Failed to load annotations file at {filepath} with reason {ex}")
            sys.exit()


def load_dataset(annotations: Annotation, train_split_filepath: str) -> Dataset:
    """
    Loads the training dataset using the train split given from the official MTL-AQA authors.

    Args:
        annotations (Annotation): The parsed dictonary representation of the annotations from MTL-AQA authors.
        train_split_filepath (str): The path to the official train split from MTL-AQA authors (pickle file).

    Returns:
        Dataset: The training data set of clips.
    """
    
    logging.info("Loading dataset...")
    
    # Load the train/val clips metadata
    train_clips = []
    with open(train_split_filepath, "rb") as f:
        try:
            data = pickle.load(f)
            # format of data tuples of (video_num, clip_idx)
            for (video_num, clip_idx) in data:
                # Look up the clip data from the given annotations
                annotation = annotations[(video_num, clip_idx)]
                start_frame = annotation["start_frame"]
                end_frame = annotation["end_frame"]
                train_clips.append(VideoClip(video_num, start_frame, end_frame))
        except Exception as ex:
            logging.error(f"Failed to load training split file at {train_split_filepath} with reason {ex}")
            sys.exit()
            
    # Create dataset from the clips
    train_dataset = DivingDataset(train_clips)
    logging.info(f"Loaded training dataset with {len(train_dataset)} clips")
    
    return train_dataset


################################## Main Entry Point ##################################


# Program's main entry point
def main():
    logging.info("--- Start of Program ---")
    
    # Load annotations
    annotations_filepath = "D:/Programming/Github/DeepSports/training/TimeSformer/final_annotations_dict.pkl"
    annotations = load_annotations(annotations_filepath)
    
    # Load training dataset
    train_split_filepath = "D:/Programming/Github/DeepSports/training/TimeSformer/train_split_0.pkl"
    train_dataset = load_dataset(annotations, train_split_filepath)
    
    # Compute online mean and std across each channel
    overall_mean = torch.zeros(3, dtype=torch.float64)
    overall_std = torch.zeros(3, dtype=torch.float64)
    for idx, clip in enumerate(train_dataset):
        # channels, num_frames, height, width
        n = idx + 1
        std, mean = torch.std_mean(clip, dim=(1,2,3))
        overall_mean = (1.0 / n) * (mean + (n-1) * overall_mean)
        overall_std = (1.0 / n) * (std + (n-1) * overall_std)
        if idx % 10 == 0:
            logging.info(f"Clip {idx}/{len(train_dataset)}\tmean: {overall_mean}\tstd: {overall_std}")
    
    logging.info(f"Overall Mean:\t{overall_mean}")
    logging.info(f"Overall Std:\t{overall_std}")
    # mean=(0.2719, 0.4617, 0.5961),
    # std=(0.1870, 0.1881, 0.2604)
    
    logging.info("--- End of Program ---")


if __name__ == "__main__":
    main()
