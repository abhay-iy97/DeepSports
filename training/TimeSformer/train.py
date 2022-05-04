# For help type 'python train.py -h' in the command-line.
# Example:
# python train.py -log INFO --root_dir path\to\whole_videos_frames --pretrained_model path\to\model.pyth --batch_size 4 --epochs 5 --learning_rate 0.00001 --weight_decay 0.00001 --optimizer AdamW --dropout 0.5 0.5 --topology 512 256 --frame_method spaced_varied


"""
    Command-Line Argument       Default                             Values
    ----------------------------------------------------------------------------------------------------------------------
    loglevel				    INFO                                [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    gpu						    True                                [False, True]
    root_dir				    ./                                  directory path
    train_path				    ./train_split_0.pkl                 .pkl file path
    test_path				    ./test_split_0.pkl                  .pkl file path
    batch_size				    4                                   [1, inf]
    epochs					    20                                  [1, inf]
    learning_rate			    0.00001                             [0.0, inf]
    weight_decay			    0.00001                             [0.0, inf]
    momentum			        0.9                                 [0.0, inf]
    amsgrad                     False                               [False, True]
    train_val_split_ratio	    0.8                                 [0.0, 1.0]
    frame_num				    8                                   [1, inf]
    frame_method			    space_fixed		                    [random, spaced_fixed, spaced_varied, spaced_fixed_new, spaced_varied_new]
    spatial_size			    224                                 [1, inf]
    freeze					    False                               [False, True]
    dropout					    [0.5, 0.5]                          list of drop prob, [0.0, 1.0] each. dropout is before the linear
    topology				    [512, 256]                          list of ints [1, inf] each
    output					    ./losses.png                        .png file path
    annotation_path			    ./final_annotations_dict.pkl        .pkl file path
    attention_type			    divided_space_time                  [divided_space_time, space_only, joint_space_time]
    optimizer				    AdamW	                            [Adam, AdamW, SGD, RMSProp]
    activation				    None	                            [None, ReLU, LeakyReLU, ELU, GELU]
    patch_size				    16                                  [1, inf]
    embed_dim				    768                                 [1, inf]
    pretrained_model		    none                                path/to/model.pyth or none
    evaluate                    False                               [False, True]
    normalize                   False                               [False, True]
    data_aug                    False                               [False, True]
    videos                      all                                 all or list of directories (i.e. 01 02)
    loss_mse_weight             1                                   [-inf, inf]
    loss_spcoef_weight          0                                   [-inf, inf]
    use_decoder                 False                               [False, True]
"""


import argparse
import logging
import math
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import functional as T
import torchsort
from PIL import Image
import numpy as np
from timesformer.models.vit import TimeSformer
import matplotlib.pyplot as plt
import random
import gc


# Annotation Alias, matches the format of the contents in the annotations .pkl file
#   key = tuple of video name and clip number
#   value = json data corresponding to key
Annotation = Dict[Tuple[int, int], Dict[str, float]]


################################## Miscellaneous ##################################

    
def get_cmdline_arguments() -> Dict[str, Any]:
    """
    Parses the command line arguments and validates them.
    Then returns them as a dictionary.

    Returns:
        Dict[str, Any]: the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune TimeSformer on MTL-AQA dataset.")
    
    parser.add_argument(
        "-log",
        "--loglevel",
        type=str,
        help="The level of logging in the application. Default value is 'INFO'. Possible values: [DEBUG, INFO, WARNING, ERROR, CRITICAL].",
        default="INFO",
        required=False
    )
    
    parser.add_argument(
        "-vid",
        "--videos",
        type=str,
        nargs="+",
        help="The videos directory name to use. Default value is 'all'. To use select directories, only list their directory names, i.e. '-vid 01 02 03'",
        default=["all"],
        required=False
    )
    
    parser.add_argument(
        "-gpu",
        "--gpu",
        type=str,
        help="Whether to use GPU for training. Defaults to True.",
        default="True",
        required=False
    )
    
    parser.add_argument(
        "--use_decoder",
        type=str,
        help="Whether to use a Transformer Decoder or revert to using MLP. Defaults to False.",
        default="False",
        required=False
    )
    
    parser.add_argument(
        "-eval",
        "--evaluate",
        type=str,
        help="Whether to use evaluate on testing dataset. Defaults to False.",
        default="False",
        required=False
    )
    
    parser.add_argument(
        "--loss_spcoef_weight",
        type=float,
        help="The loss weight for the spearman coefficient. Defaults to 0 (No spearman in loss).",
        default=0,
        required=False
    )
    
    parser.add_argument(
        "--loss_mse_weight",
        type=float,
        help="The loss weight for the MSE. Defaults to 1.",
        default=1,
        required=False
    )
    
    parser.add_argument(
        "-norm",
        "--normalize",
        type=str,
        help="Whether to use normalize the RGB channels in the video clips as a preprocessing step. Defaults to False.",
        default="False",
        required=False
    )
    
    parser.add_argument(
        "-da",
        "--data_aug",
        type=str,
        help="Whether to use randomly resize and crop the video clips as a preprocessing step. Defaults to False.",
        default="False",
        required=False
    )
    
    parser.add_argument(
        "-ams",
        "--amsgrad",
        type=str,
        help="Whether to use amsgrad for Adam/AdamW optimizer. Defaults to False.",
        default="False",
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
        "-tr",
        "--train_path",
        type=str,
        help="The filepath to the train split pickle file. Default value is './train_split_0.pkl', the same directory of this python script.",
        default="./train_split_0.pkl",
        required=False
    )
    
    parser.add_argument(
        "-te",
        "--test_path",
        type=str,
        help="The filepath to the test split pickle file. Default value is './test_split_0.pkl', the same directory of this python script.",
        default="./test_split_0.pkl",
        required=False
    )
    
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="The batch size used in training. Default value is 4.",
        default=4,
        required=False
    )
    
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        help="The number of epochs used in training. Default value is 5.",
        default=5,
        required=False
    )
    
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="The learning rate used in training. Default value is 0.00001 (1e-5).",
        default=1e-5,
        required=False
    )
    
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        help="The weight decay used in training. Default value is 0.00001 (1e-5).",
        default=1e-5,
        required=False
    )
    
    parser.add_argument(
        "-mom",
        "--momentum",
        type=float,
        help="The momentum used in SGD/RMSProp optimizers for training. Default value is 0.9.",
        default=0.9,
        required=False
    )
    
    parser.add_argument(
        "-at",
        "--attention_type",
        type=str,
        help="The type of attention used in the transformer. Default value is 'divided_space_time'. Possible values are: ['divided_space_time', 'space_only', 'joint_space_time']",
        default="divided_space_time",
        required=False
    )
    
    parser.add_argument(
        "-act",
        "--activation",
        type=str,
        help="The activation function used in the MLP network. Default value is 'None'. Possible values are: ['None', 'ReLU', 'LeakyReLU', 'ELU', 'GELU']",
        default="None",
        required=False
    )
    
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        help="The optimizer used in training. Default value is 'AdamW'. Possible values are: ['Adam', 'AdamW', 'SGD', 'RMSProp'] (not case-sensitive)",
        default="AdamW",
        required=False
    )
    
    parser.add_argument(
        "-ps",
        "--patch_size",
        type=int,
        help="The patch size used in the transformer. Default value is 16.",
        default=16,
        required=False
    )
    
    parser.add_argument(
        "-ed",
        "--embed_dim",
        type=int,
        help="The embed dimensions output from the transformer. Default value is 768.",
        default=768,
        required=False
    )
    
    parser.add_argument(
        "-pm",
        "--pretrained_model",
        type=str,
        help="The filepath to the pretrained .pyth model for the TimeSformer. Defaults to 'none'. If set to 'none', it will train a TimeSformer from scratch.",
        default="none",
        required=False
    )
    
    parser.add_argument(
        "-sr",
        "--train_val_split_ratio",
        type=float,
        help="The ratio in which the training and validation datasets are split. Default value is 0.8.",
        default=0.8,
        required=False
    )
    
    parser.add_argument(
        "-fn",
        "--frame_num",
        type=int,
        help="The number of frames to use in each clip. Default value is 8.",
        default=8,
        required=False
    )
    
    parser.add_argument(
        "-fm",
        "--frame_method",
        type=str,
        help="The algorithm to use to sample frames from a long clip. Default is 'spaced_fixed'. Possible values are ['random', 'spaced_fixed', 'spaced_varied', 'spaced_fixed_new', 'spaced_varied_new'].",
        default="spaced_fixed",
        required=False
    )
    
    parser.add_argument(
        "-ss",
        "--spatial_size",
        type=int,
        help="The image size in pixels that they will be resized to. Default value is 224.",
        default=224,
        required=False
    )
    
    parser.add_argument(
        "-f",
        "--freeze",
        type=str,
        help="Whether to freeze the gradients in the TimeSformer model. Defaults to False.",
        default="False",
        required=False
    )
    
    parser.add_argument(
        "-dr",
        "--dropout",
        type=float,
        help="The dropout value per layer in MLP. Default value is [0.5, 0.5].",
        default=[0.5, 0.5],
        nargs="+",
        required=False
    )
    
    parser.add_argument(
        "-top",
        "--topology",
        type=int,
        help="The hidden neurons topology between the TimeSformer model and the final output layer. Defaults to [512, 256].",
        default=[512, 256],
        nargs="+",
        required=False
    )
    
    parser.add_argument(
        "-out",
        "--output",
        type=str,
        help="The output filepath for the losses figure. Defaults to './losses.png', in the same directory.",
        default="./losses.png",
        required=False
    )
    
    # Parse arguments
    args = vars(parser.parse_args())
    
    # Prepend a timestamp to each logging
    numeric_loglevel = getattr(logging, args["loglevel"].upper(), None)
    if not isinstance(numeric_loglevel, int):
        raise ValueError('Invalid log level: %s' % numeric_loglevel)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=numeric_loglevel,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Convert booleans to the correct type (workaround of argparse)
    args["gpu"] = bool(args["gpu"].lower() == "true")
    args["evaluate"] = bool(args["evaluate"].lower() == "true")
    args["normalize"] = bool(args["normalize"].lower() == "true")
    args["data_aug"] = bool(args["data_aug"].lower() == "true")
    args["amsgrad"] = bool(args["amsgrad"].lower() == "true")
    args["freeze"] = bool(args["freeze"].lower() == "true")
    args["use_decoder"] = bool(args["use_decoder"].lower() == "true")
    
    # Convert any relative paths to absolute paths
    args["root_dir"] = os.path.abspath(args["root_dir"])
    args["annotation_path"] = os.path.abspath(args["annotation_path"])
    args["train_path"] = os.path.abspath(args["train_path"])
    args["test_path"] = os.path.abspath(args["test_path"])
    args["test_path"] = os.path.abspath(args["test_path"])
    args["output"] = os.path.abspath(args["output"])
    
    # Special case for pretrained model's path. If none, we dont use a path; otherwise, resolve path to absolute
    if args["pretrained_model"].lower() == "none":
        args["pretrained_model"] = args["pretrained_model"].lower()
    else:
        args["pretrained_model"] = os.path.abspath(args["pretrained_model"])
    
    # Lowercase strings to allow case-insensitive input
    args["optimizer"] = args["optimizer"].lower()
    args["frame_method"] = args["frame_method"].lower()
    args["attention_type"] = args["attention_type"].lower()
    args["activation"] = args["activation"].lower()
    args["videos"] = [x.lower() for x in args["videos"]]
        
    # Ensure validation of arguments
    if args["frame_method"] not in ["random", "spaced_fixed", "spaced_varied", "spaced_fixed_new", "spaced_varied_new"]:
        logging.warning(f"The parameter frame_method uses an invalid value ({args['frame_method']}). Will use 'spaced_fixed' instead.")
        args["frame_method"] = "spaced_fixed"
    if args["attention_type"] not in ["divided_space_time", "space_only","joint_space_time"]:
        logging.warning(f"The parameter attention_type uses an invalid value ({args['attention_type']}). Will use 'divided_space_time' instead.")
        args["attention_type"] = "divided_space_time"
    if len(args["topology"]) != len(args["dropout"]):
        logging.warning(f"Dropouts and Topology do not match. Topology: {args['topology']}. Dropout: {args['dropout']}. Using dropout of 0.5 for each layer instead.")
        args["dropout"] = [0.5] * len(args["topology"])
    if args["optimizer"] not in ["adam", "adamw", "sgd", "rmsprop"]:
        logging.warning(f"The parameter optimizer uses an invalid value ({args['optimizer']}). Will use 'AdamW' instead.")
        args["optimizer"] = "adamw"
    if args["activation"] not in ["none", "relu", "leakyrelu", "elu", "gelu"]:
        logging.warning(f"The parameter activation uses an invalid value ({args['activation']}). Will use 'None' instead.")
        args["activation"] = "none"
        
    if args["videos"][0] == "all":
        args["videos"] = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 17, 18, 22, 26]
    else:
        args["videos"] = list(map(int, args["videos"]))
        
    return args


def convert_size(size_bytes: float) -> str:
    """
    Converts the size in bytes to an appropriate representation (KB, MB, GB, etc..)  

    Args:
        size_bytes (float): the size in bytes to convert

    Returns:
        str: the human readable string of the byte size
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the rank of the given tensor.
    Credit: https://discuss.pytorch.org/t/spearmans-correlation/91931/5
    """
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x), device=x.device)
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the spearman correlation between two 1D tensors of shape (batch_size, ).
    Credit: https://discuss.pytorch.org/t/spearmans-correlation/91931/5
    
    Args:
        x: Shape (batch_size, )
        y: Shape (batch_size, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def differentiable_spearman_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    A differentiable spearman correlation function using torchsort from:
        https://github.com/teddykoker/torchsort

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor

    Returns:
        torch.Tensor: spearman correlation value
    """
    pred = torchsort.soft_rank(pred.float().cpu())
    target = torchsort.soft_rank(target.float().cpu())
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


def custom_loss(outputs: torch.Tensor, targets: torch.Tensor, mse_weight: float, spcoeff_weight: float, device: torch.device) -> torch.Tensor:
    """A custom loss that combines MSE loss with differentiable spearman correlation: alpha * MSE - beta * spcoeff.

    Args:
        outputs (torch.Tensor): the predictions from the model
        targets (torch.Tensor): the ground truth labels
        mse_weight (float): the weight given for MSE loss
        spcoeff_weight (float): the weight given for the spearman correlation loss
        device (torch.device): the device to cast the loss to

    Returns:
        torch.Tensor: the combined loss
    """
    loss = F.mse_loss(outputs, targets) * mse_weight
    if spcoeff_weight != 0:
        spcoef = differentiable_spearman_correlation(outputs, targets) * spcoeff_weight
        spcoef = spcoef.to(device)
        loss -= spcoef
    return loss


################################## Data Preparation ##################################


class VideoClip:
    """
    A data class to hold a single video clip for the MTL-AQA dataset.
    """
    
    # Normalizes the RGB channels on the video clip
    VIDEO_NORMALIZE = None
    
    # Randomly crop and flip the video clip
    VIDEO_CROP_FLIP = None
    
    # The root directory path to the frames on disk (should be the one that includes the 01/ 02/ ... directories)
    FRAME_ROOT_DIR = "./"
    
    # The number of frames each clip will have
    NUM_FRAMES = 8
    
    # The method in which the frames are selected. Values: ['random', 'spaced_fixed', 'spaced_varied']
    # If the image had 80 frames, and we need to sample only 8 frames then:
    #   random- will sample 8 indices from [0, 79] and use the sorted values as the clip values
    #   spaced_fixed- will yield [0+k, 10+k, 20+k, ..., 60+k, 70+k] where k=random[0,9] since the step_size=80/8
    #   spaced_varied- its similar to spaced_fixed, except k is different for each subclip, i.e. k0, k1, ..., kn are all random[0,9]
    FRAME_SELECTION_METHOD = "spaced_fixed"
    
    # Spatial dimensions that each clip will be resized to
    SPATIAL_SIZE = 224
    
    def __init__(self, video_num: int, start_frame: int, end_frame: int, difficulty: float, final_score: float, args: Dict[str, Any]) -> None:
        """
        Initializes a video clip from the given metadata.

        Args:
            video_num (int): the video this clip is from (i.e. 1 for 01.mp4)
            start_frame (int): the first frame in the clip
            end_frame (int): the last frame in the clip
            difficulty (float): the difficulty assigned to the clip
            final_score (float): the final score of the clip
            args (Dict[str, Any]): the commandline arguments.
        """
        
        if start_frame > end_frame:
            logging.warning(f"Invalid clip for {video_num:02d}, with start_frame ({start_frame}) > end_frame {end_frame}")
        
        self.video_num = video_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.clip_num_frames = end_frame - start_frame + 1
        self.difficulty = difficulty
        self.final_score = final_score
        # calculate a normalized score of [0, 1], diving by 30 because 3 judges voting [0, 10] each
        self.normalized_score = (final_score / difficulty) / 30
        
        # Update state using args
        self.FRAME_ROOT_DIR = args["root_dir"]
        self.FRAME_SELECTION_METHOD = args["frame_method"].lower()
        self.NUM_FRAMES = args["frame_num"]
        self.SPATIAL_SIZE = args["spatial_size"]
        
        if args["normalize"] and self.VIDEO_NORMALIZE is None:
            self.VIDEO_NORMALIZE = transforms.Compose([
                transforms.Normalize(
                    mean=(0.2719, 0.4617, 0.5961),
                    std=(0.1870, 0.1881, 0.2604)
                ),
            ])
            
        if args["data_aug"] and self.VIDEO_CROP_FLIP is None:
            self.VIDEO_CROP_FLIP = transforms.Compose([
                transforms.RandomCrop(size=(args["spatial_size"], args["spatial_size"])),
                transforms.RandomHorizontalFlip(),
            ])
        
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the clip from disk and returns the inputs and the targets as a tuple of pytorch's tensor:
            inputs- the video clip in the shape (channels, frames, height, width)
            targets- the normalized score and the difficulty in the shape (2,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the inputs and the targets as a tuple
        """
        
        # Since we are constrained to a set number of frames, sample (somehow) the frames from the entire clip
        frame_indices = None
        
        if self.FRAME_SELECTION_METHOD == "random":
            frame_range = range(self.start_frame, self.end_frame + 1)
            frame_indices = sorted(random.sample(frame_range, self.NUM_FRAMES))
        
        else: # space_fixed (default) or space_varied (and their newer implementation variants)
            step_size = self.clip_num_frames / self.NUM_FRAMES
            offset = self.start_frame # offset to the first frame
            
            if self.FRAME_SELECTION_METHOD == "spaced_fixed":
                frame_indices = [int(i * step_size + offset) for i in range(self.NUM_FRAMES)]
            elif self.FRAME_SELECTION_METHOD == "spaced_varied":
                offset += random.random() * step_size
                frame_indices = [int(i * step_size + offset) for i in range(self.NUM_FRAMES)]
            elif self.FRAME_SELECTION_METHOD == "spaced_fixed_new":
                subclip_offset = random.random() * step_size
                frame_indices = [int(i * step_size + offset + subclip_offset) for i in range(self.NUM_FRAMES)]
            elif self.FRAME_SELECTION_METHOD == "spaced_varied_new":
                frame_indices = [int(i * step_size + offset + random.random() * step_size) for i in range(self.NUM_FRAMES)]
        
        # The intermediate sizes to downsample video resolution
        downsample_height = 256
        downsample_width = int(round(downsample_height * 16.0 / 9.0))
        
        # Contains the video clip in torch with (channels, num_frames, height, width)
        clip = torch.empty(size=(3, self.NUM_FRAMES, downsample_height, downsample_width), dtype=torch.float32)
        
        # Load the images from disk and preprocess them into a tensor
        for idx, frame_num in enumerate(frame_indices):
            img_path = os.path.join(*[self.FRAME_ROOT_DIR, f"{self.video_num:02d}", f"{frame_num:06d}.jpg"])
            if not os.path.exists(img_path):
                logging.error(f"File does not exist: {img_path}. frame_indices={frame_indices}, step_size={step_size}, offset={offset}, start={self.start_frame}, end={self.end_frame}")
                sys.exit()
            
            img = Image.open(img_path)
            tensor = T.to_tensor(img).type(dtype=torch.float32) # (channels, height, width)
            tensor = T.resize(tensor, size=(downsample_height, downsample_width))
            
            # Normalize the RGB values in the video clip
            if self.VIDEO_NORMALIZE is not None:
                tensor = self.VIDEO_NORMALIZE(tensor)
            
            clip[:, idx, :, :] = tensor
            del tensor
        
        # Randomly crop and flip the video
        if self.VIDEO_CROP_FLIP is not None:
            clip = self.VIDEO_CROP_FLIP(clip)
        # Otherwise downsample to target size directly
        else:
            clip = T.resize(clip, size=(self.SPATIAL_SIZE, self.SPATIAL_SIZE))
        
        # Create the outputs/targets for this clip
        target = torch.tensor([self.normalized_score, self.difficulty], dtype=torch.float32)
        return clip, target


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


def load_dataset(annotations: Annotation, args: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the data using the train/test splits given from the official MTL-AQA authors.
    Also, splits the training dataset into a train/validation splits, as specified by 'train_val_split_ratio'.

    Args:
        annotations (Annotation): The parsed dictonary representation of the annotations from MTL-AQA authors.
        args (Dict[str, Any]): The commandline arguments.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The train/validation/test split data loaders
    """
    
    train_split_filepath = args["train_path"] # The path to the official train split from MTL-AQA authors (pickle file).
    test_split_filepath = args["test_path"] # The path to the official test split from MTL-AQA authors (pickle file).
    batch_size = args["batch_size"] # The size of batch size used for the data loaders.
    train_val_split_ratio = args["train_val_split_ratio"] # The train/validation split ratio, which represents the percentage of instances used for training.
    available_videos = args["videos"] # The videos which we want to use in our dataset.   
    
    logging.info("Loading datasets...")
    logging.info(f"Using only these videos: {available_videos}")
    
    # Load the train/val clips metadata
    train_val_clips = []
    with open(train_split_filepath, "rb") as f:
        try:
            data = pickle.load(f)
            # format of data tuples of (video_num, clip_idx)
            for (video_num, clip_idx) in data:
                # Skip if not in our available videos (i.e. we want only specific to train on specific vids)
                if video_num not in available_videos:
                    continue
                
                # Look up the clip data from the given annotations
                annotation = annotations[(video_num, clip_idx)]
                start_frame = annotation["start_frame"]
                end_frame = annotation["end_frame"]
                difficulty = annotation["difficulty"]
                final_score = annotation["final_score"]
                train_val_clips.append(VideoClip(video_num, start_frame, end_frame, difficulty, final_score, args))
        except Exception as ex:
            logging.error(f"Failed to load training split file at {train_split_filepath} with reason {ex}")
            sys.exit()
            
    logging.info(f"Found {len(train_val_clips)} clips in {train_split_filepath}")
    
    # Load the test clips metadata
    test_clips = []
    with open(test_split_filepath, "rb") as f:
        try:
            data = pickle.load(f)
            # format of data tuples of (video_num, clip_idx)
            for (video_num, clip_idx) in data:
                # Skip if not in our available videos (i.e. we want only specific to train on specific vids)
                if video_num not in available_videos:
                    continue
                
                # Look up the clip data from the given annotations
                annotation = annotations[(video_num, clip_idx)]
                start_frame = annotation["start_frame"]
                end_frame = annotation["end_frame"]
                difficulty = annotation["difficulty"]
                final_score = annotation["final_score"]
                test_clips.append(VideoClip(video_num, start_frame, end_frame, difficulty, final_score, args))
        except Exception as ex:
            logging.error(f"Failed to load training split file at {test_split_filepath} with reason {ex}")
            sys.exit()
            
    logging.info(f"Found {len(test_clips)} clips in {test_split_filepath}")
            
    # Split the training clips into a train & validation split
    num_clips = len(train_val_clips)
    num_train = int(num_clips * train_val_split_ratio)
    num_val = num_clips - num_train
    logging.info(f"Splitting training ({num_clips}) with ratio {train_val_split_ratio} into {num_train} train and {num_val} validation...")
    train_clips, val_clips = random_split(train_val_clips, (num_train, num_val))
    
    # Create dataset from the clips
    train_dataset = DivingDataset(train_clips)
    val_dataset = DivingDataset(val_clips)
    test_dataset = DivingDataset(test_clips)
    
    logging.info(f"Loaded training dataset with {len(train_dataset)} clips")
    logging.info(f"Loaded validation dataset with {len(val_dataset)} clips")
    logging.info(f"Loaded testing dataset with {len(test_dataset)} clips")
    
    # Create data loaders for each dataset
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    
    logging.info("Successfully created data loaders for the datasets")
    
    return train_data_loader, val_data_loader, test_data_loader


################################## Model Creation ##################################


class DivingViT(nn.Module):
    def __init__(self, timesformer: TimeSformer, mlp_topology: List[int], dropout: List[float], freeze:bool=False, activation:str="none", use_decoder:bool=False) -> None:
        """
        Builds upon the TimeSformer model with additional MLP layers that are user-defined.

        Args:
            timesformer (TimeSformer): A pretrained instance of TimeSformer of 8 frames and 224x224 (or from scratch).
            mlp_topology (List[int]): The number of hidden neurons in the MLP layer after the pretrained model (excluding the 768 from the model and the 2 at the output).
            dropout (List[float]): Drop probability for dropout for each MLP layer after the TimeSformer model.
            freeze (bool, optional): Whether to freeze the pretrained model weights (except its head) or also add them to the gradient updates. Defaults to False.
            activation (str, optional): The activation function used in the MLP layers. Defaults to "none". Only supports ["none", "relu", "leakyrelu", "elu", "gelu"].
            use_decoder (bool, optional): Whether to use a Transformer Decoder or revert to using MLP. Defaults to False (use MLP).
        """
        super().__init__()
        
        self.timesformer = timesformer
        if freeze:
            # Freeze all layers
            for param in timesformer.parameters():
                param.requires_grad = False
            # Except last layer
            for param in timesformer.model.head.parameters():
                param.requires_grad = True
        
        self.stacked_mlp = None
        self.transformer_decoder = None
        
        # Use the Transformer Decoder
        if use_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=4)
            self.stacked_mlp = nn.Linear(768, 2)
        # Use the MLP as a decoder
        else:
            # Build the MLP linear net, starts with 768, ... (mlp_topology) ..., 2
            net = []
            last_num_features = self.timesformer.model.embed_dim # 768
            for num_features, drop_prob in zip(mlp_topology, dropout):
                net.append(nn.Dropout(p=drop_prob))
                net.append(nn.Linear(in_features=last_num_features, out_features=num_features))
                if activation == "relu":
                    net.append(nn.ReLU())
                elif activation == "leakyrelu":
                    net.append(nn.LeakyReLU())
                elif activation == "elu":
                    net.append(nn.ELU())
                elif activation == "gelu":
                    net.append(nn.GELU())
                last_num_features = num_features
            # Add last layer
            net.append(nn.Dropout(p=dropout[-1]))
            net.append(nn.Linear(in_features=last_num_features, out_features=2))
            self.stacked_mlp = nn.Sequential(*net)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward propagation pass of the network.
        First, passes the input 'x' through the pretrained model and then through the series of MLP network.

        Args:
            x (torch.Tensor): the video clips to the network of shape (batch_size, channels, num_frames, height, width)

        Returns:
            torch.Tensor: the normalized score and the difficulty in the shape (2,)
        """
        out = self.timesformer(x)   # (batch, 768)
        if self.transformer_decoder is not None:
            out = self.transformer_decoder(out, out) # (batch, 768)
        out = self.stacked_mlp(out) # (batch, 2)
        return out


def create_model(device: torch.device, args: Dict[str, Any]) -> DivingViT:
    """
    Creates a model given the specified parameters.

    Args:
        args (Dict[str, Any]): command line arguments

    Returns:
        DivingViT: The created model
    """
    logging.info("Creating model...")
    
    # Create TimeSformer (either pretrained or from scratch)
    if args["pretrained_model"] == "none":
        timesformer = TimeSformer(
            img_size=args["spatial_size"],
            num_classes=args["embed_dim"],
            num_frames=args["frame_num"],
            attention_type=args["attention_type"]
        )
    else:
        timesformer = TimeSformer(
            img_size=args["spatial_size"],
            num_classes=args["embed_dim"],
            num_frames=args["frame_num"],
            attention_type=args["attention_type"],
            pretrained_model=args["pretrained_model"]
        )
    
    # Create our diving model
    model = DivingViT(
        timesformer=timesformer,
        dropout=args["dropout"],
        freeze=args["freeze"],
        mlp_topology=args["topology"],
        activation=args["activation"],
        use_decoder=args["use_decoder"]
    )
    
    # To allow for multiple GPUs
    if args["gpu"]:
        model = nn.DataParallel(model)
    
    # Move to the desired device
    model.to(device)
    return model


################################## Training ##################################


def train(model: DivingViT, device: torch.device, train_data_loader: DataLoader, val_data_loader: DataLoader, args: Dict[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    """
    Trains the model on the specified device on the training dataset. Then evaluates on the validation dataset.
    Returns the training and validation losses per epoch.

    Args:
        model (DivingViT): model to train
        device (torch.device): device (gpu/cpu) to train on
        train_data_loader (DataLoader): training dataset
        val_data_loader (DataLoader): validation dataset
        args (Dict[str, Any]): commandline arguments

    Returns:
        Tuple[List[float], List[float], List[float]]: a tuple of training losses, validation losses, and validation spearman correlation per epoch
    """
    logging.info("Training model...")
   
    # Optimizer
    if args["optimizer"].lower() == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"], amsgrad=args["amsgrad"])
    elif args["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"], amsgrad=args["amsgrad"])
    elif args["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"], momentum=args["momentum"])
    elif args["optimizer"].lower() == "rmsprop":
        optimizer = optim.RMSprop(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"], momentum=args["momentum"])
    
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses = []
    val_losses = []
    val_spearman_correlations = []
    epochs = args["epochs"]
    
    for epoch in range(epochs):
        # Training
        train_loss = []
        model.train()
        for (inputs, targets) in train_data_loader:
            # Zero out parameters efficiently
            for param in model.parameters():
                param.grad = None
            
            # Move data to correct device (gpu/cpu)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with autocast, which automatically casts to lower floating precision if needed
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Custom loss that is alpha*MSE - beta*spearman
                loss = custom_loss(outputs, targets, args["loss_mse_weight"], args["loss_spcoef_weight"], device)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(loss.item())
            
            # Free memory
            del inputs, targets, outputs, loss
            gc.collect()
            if args["gpu"]:
                torch.cuda.empty_cache()
        train_losses.append(np.mean(train_loss))
        
        # Validation
        val_loss = []
        spearman_outputs = []
        spearman_targets = []
        model.eval()
        with torch.no_grad():
            for (inputs, targets) in val_data_loader:
                # Move data to correct device (gpu/cpu)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass with autocast, which automatically casts to lower floating precision if needed
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # Custom loss that is alpha*MSE - beta*spearman
                    loss = custom_loss(outputs, targets, args["loss_mse_weight"], args["loss_spcoef_weight"], device)
                
                val_loss.append(loss.item())
                spearman_outputs.append(outputs)
                spearman_targets.append(targets)
        
        val_losses.append(np.mean(val_loss))
        
        # Compute spearman coefficient
        spearman_output = torch.cat(spearman_outputs, dim=0) # concat across batches
        spearman_target = torch.cat(spearman_targets, dim=0) # concat across batches
        spearman_output = spearman_output[:, 0] * spearman_output[:, 1] * 30 # get final score
        spearman_target = spearman_target[:, 0] * spearman_target[:, 1] * 30 # get final score
        val_spearman_correlations.append(spearman_correlation(spearman_target, spearman_output).item())
    
        # Log losses and spearman correlation
        logging.info(f"Epoch {epoch+1}/{epochs} \t train_loss: {np.mean(train_loss):.4f} \t val_loss: {np.mean(val_loss):.4f} \t val_spcoeff: {spearman_correlation(spearman_target, spearman_output):.4f}")
    
    return train_losses, val_losses, val_spearman_correlations
    

################################## Evaluation ##################################


def evaluate(model: DivingViT, device: torch.device, test_data_loader: DataLoader, args: Dict[str, Any]) -> None:
    """
    Evaluates the model on the testing dataset by computing the testing loss and spearman correlation.

    Args:
        model (DivingViT): model to evaluate
        device (torch.device): the device (cpu/gpu) to run on
        test_data_loader (DataLoader): testing dataset
        args (Dict[str, Any]): commandline arguments
    """
    logging.info("Evaluating model...")
    
    test_loss = []
    spearman_outputs = []
    spearman_targets = []
    
    with torch.no_grad():
        model.eval()
        for (inputs, targets) in test_data_loader:
            # Move data to correct device (gpu/cpu)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with autocast, which automatically casts to lower floating precision if needed
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Custom loss that is alpha*MSE - beta*spearman
                loss = custom_loss(outputs, targets, args["loss_mse_weight"], args["loss_spcoef_weight"], device)

            test_loss.append(loss.item())
            spearman_outputs.append(outputs)
            spearman_targets.append(targets)
        
    # Compute spearman coefficient
    spearman_output = torch.cat(spearman_outputs, dim=0) # concat across batches
    spearman_target = torch.cat(spearman_targets, dim=0) # concat across batches
    spearman_output = spearman_output[:, 0] * spearman_output[:, 1] * 30 # get final score
    spearman_target = spearman_target[:, 0] * spearman_target[:, 1] * 30 # get final score
    
    # Log loss and spearman correlation
    logging.info(f"test_loss: {np.mean(test_loss):.4f} \t test_spcoeff: {spearman_correlation(spearman_target, spearman_output):.4f}")


# Save the plotted training and validation losses to the specified filepath
def plot_losses(fig_filepath: str, train_losses: List[float], val_losses: List[float], val_spearman_correlations: List[float]) -> None:
    """
    Save the plotted training and validation losses to the specified filepath
    
    Args:
        fig_filepath (str): filepath to save the plot to
        train_losses (List[float]): list of training losses per epoch
        val_losses (List[float]): list of validation losses per epoch
        val_spearman_correlations (List[float]): list of validation spearman correlations per epoch
    """
    logging.info("Plotting losses...")
    
    num_epochs = len(train_losses)
    
    fig = plt.figure(figsize = (10, 5))
    xvalues = list(range(1, num_epochs+1))
    
    ax1 = plt.subplot(121)
    ax1.plot(xvalues, train_losses, label="train_loss")
    ax1.plot(xvalues, val_losses, label="val_loss")
    ax1.title.set_text("Loss per Epoch")
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    
    ax2 = plt.subplot(122)
    ax2.plot(xvalues, val_spearman_correlations, label="Spearman Correlation")
    ax2.title.set_text("Spearman Correlation per Epoch")
    ax2.legend()
    ax2.set_ylabel("Spearman Correlation")
    ax2.set_xlabel("Epoch")
    
    plt.savefig(fig_filepath)
    
    logging.info(f"Saved figure to {fig_filepath}")


################################## Main Entry Point ##################################


# Program's main entry point
def main():
    # Parse commandline arguments
    args = get_cmdline_arguments()
    
    logging.info("--- Start of Program ---")
    logging.info(f"Command-Line Args: {args}")
    
    # Prepare dataset
    annotations = load_annotations(args["annotation_path"])
    train_data, val_data, test_data = load_dataset(annotations, args)
    
    # Get the appropriate device
    device = torch.device('cuda' if args["gpu"] and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if args["gpu"] and torch.cuda.is_available():
        logging.info(f"GPU Total VRAM: {convert_size(torch.cuda.get_device_properties(device).total_memory)}")
        logging.info(f"GPU Reserved VRAM: {convert_size(torch.cuda.memory_reserved(device))}")
        logging.info(f"GPU Allocated VRAM: {convert_size(torch.cuda.memory_allocated(device))}")
    
    # Create the model
    model = create_model(device, args)
    
    # Train the model
    train_losses, val_losses, val_spearman_correlations = train(model, device, train_data, val_data, args)
    
    # Evaluate the model
    if args["evaluate"]:
        evaluate(model, device, test_data, args)
        plot_losses(args["output"], train_losses, val_losses, val_spearman_correlations)
    
    logging.info("--- End of Program ---")


if __name__ == "__main__":
    main()
