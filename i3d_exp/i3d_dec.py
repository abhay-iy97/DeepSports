#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
#from timesformer.models.vit import TimeSformer
import matplotlib.pyplot as plt
import random
import gc
from mxnet import gluon, nd, image, npx


# In[3]:


Annotation = Dict[Tuple[int, int], Dict[str, float]]


# In[4]:


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


# In[5]:


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the rank of the given tensor.
    Credit: https://discuss.pytorch.org/t/spearmans-correlation/91931/5
    """
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x), device=x.device)
    return ranks


# In[6]:


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


# In[7]:


class VideoClip:
    """
    A data class to hold a single video clip for the MTL-AQA dataset.
    """

    # Convertor between PIL images to PyTorch Tensor
    CONVERT_TENSOR = transforms.ToTensor()

    # The root directory path to the frames on disk (should be the one that includes the 01/ 02/ ... directories)
    FRAME_ROOT_DIR = "./"

    # The number of frames each clip will have
    NUM_FRAMES = 128

    # The method in which the frames are selected. Values: ['random', 'spaced_fixed', 'spaced_varied']
    # If the image had 80 frames, and we need to sample only 8 frames then:
    #   random- will sample 8 indices from [0, 79] and use the sorted values as the clip values
    #   spaced_fixed- will yield [0, 10, 20, ..., 60, 70] since the step_size=80/8
    #   spaced_varied- its similar to spaced_fixed, except the entire list is shifted by a random value [0, step_size-1]
    FRAME_SELECTION_METHOD = "spaced_varied"

    # Spatial dimensions that each clip will be resized to
    SPATIAL_SIZE = 224

    def __init__(self, video_num: int, start_frame: int, end_frame: int, difficulty: float, final_score: float) -> None:
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
        self.FRAME_ROOT_DIR = "/scratch1/bodala/diving_data/whole_videos_frames"
        self.FRAME_SELECTION_METHOD = "spaced_varied"
        self.NUM_FRAMES = 128
        self.SPATIAL_SIZE = 224

        global fsr, nf, ss
        fsr = self.FRAME_SELECTION_METHOD
        nf = self.NUM_FRAMES
        ss = self.SPATIAL_SIZE

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the clip from disk and returns the inputs and the targets as a tuple of pytorch's tensor:
            inputs- the video clip in the shape (channels, frames, height, width)
            targets- the normalized score and the difficulty in the shape (2,)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the inputs and the targets as a tuple
        """

        # Creates the video clip in torch with (channels, num_frames, height, width)
        clip = torch.empty(size=(3, self.NUM_FRAMES, self.SPATIAL_SIZE, self.SPATIAL_SIZE), dtype=torch.float32)

        # Since we are constrained to a set number of frames, sample (somehow) the frames from the entire clip
        frame_indices = None

        if self.FRAME_SELECTION_METHOD == "random":
            frame_range = range(self.start_frame, self.end_frame + 1)
            frame_indices = sorted(random.sample(frame_range, self.NUM_FRAMES))
        else: # spaced_fixed (default) or spaced_varied
            step_size = self.clip_num_frames / self.NUM_FRAMES
            offset = self.start_frame
            if self.FRAME_SELECTION_METHOD == "spaced_varied":
                offset += random.random() * step_size
            frame_indices = [int(i * step_size + offset) for i in range(self.NUM_FRAMES)]

        # Load the images from disk and preprocess them into a tensor
        for idx, frame_num in enumerate(frame_indices):
            img_path = os.path.join(*[self.FRAME_ROOT_DIR, f"{self.video_num:02d}", f"{frame_num:06d}.jpg"])
            if not os.path.exists(img_path):
                logging.error(f"File does not exist: {img_path}. frame_indices={frame_indices}, step_size={step_size}, offset={offset}, start={self.start_frame}, end={self.end_frame}")
                sys.exit()
            img = Image.open(img_path).resize((self.SPATIAL_SIZE, self.SPATIAL_SIZE))
            tensor = self.CONVERT_TENSOR(img).type(dtype=torch.float32) # (channels, height, width)
            clip[:, idx, :, :] = tensor

        # Create the outputs/targets for this clip
        target = torch.tensor([self.normalized_score, self.difficulty], dtype=torch.float32)
        return clip, target


# In[8]:


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


# In[9]:


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


# In[10]:


def load_dataset(annotations: Annotation) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the data using the train/test splits given from the official MTL-AQA authors.
    Also, splits the training dataset into a train/validation splits, as specified by 'train_val_split_ratio'.
    Args:
        annotations (Annotation): The parsed dictonary representation of the annotations from MTL-AQA authors.
        args (Dict[str, Any]): The commandline arguments.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The train/validation/test split data loaders
    """

    train_split_filepath = "/scratch1/bodala/diving_data/train_split_0.pkl" # The path to the official train split from MTL-AQA authors (pickle file).
    test_split_filepath = "/scratch1/bodala/diving_data/test_split_0.pkl" # The path to the official test split from MTL-AQA authors (pickle file).
    batch_size = 16 # The size of batch size used for the data loaders.
    train_val_split_ratio = 0.9 # The train/validation split ratio, which represents the percentage of instances used for training.
    available_videos = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 17, 18, 22, 26] # The videos which we want to use in our dataset.
    #available_videos = [13]

    global b_s, tvr
    b_s = batch_size
    tvr = train_val_split_ratio

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
                train_val_clips.append(VideoClip(video_num, start_frame, end_frame, difficulty, final_score))
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
                test_clips.append(VideoClip(video_num, start_frame, end_frame, difficulty, final_score))
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
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, persistent_workers=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=20, persistent_workers=True, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, persistent_workers=True, pin_memory=True)

    print("Successfully created data loaders for the datasets")

    return train_data_loader, val_data_loader, test_data_loader


# In[11]:


class DivingViT(nn.Module):
    def __init__(self):
        super(DivingViT, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Loadning the Pre-trained I3D model
        self.model = torch.load("/home1/bodala/i3d_compl.pth")

        #Modifying the I3D models head
        classi = nn.Linear(1024, 1024, bias=True)
        self.model.output = classi
        self.model = self.model.to(device)

        #freezing the pre-trained I3D
        # for param in self.model.parameters():
        #     param.requires_grad = False

        #adding transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.transformer_decoder = self.transformer_decoder.to(device)

        #MLP block
        self.classifier_layer = nn.Sequential(
            nn.Linear(1024,512, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(512,512, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(512,256, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(256,2)
        )
        self.classifier_layer = self.classifier_layer.to(device)

    def forward(self, inputs):

        x = self.model(inputs)
        x = self.transformer_decoder(x, x)
        x = self.classifier_layer(x)
        return x


# In[22]:


def create_model(device: torch.device) -> DivingViT:
    """
    Creates a model given the specified parameters.
    Args:
        args (Dict[str, Any]): command line arguments
    Returns:
        DivingViT: The created model
    """
    logging.info("Creating model...")




    # Create model
    model = DivingViT()
    global m
    m = model
    # To allow for multiple GPUs
    #if args["gpu"]:
    model = nn.DataParallel(model)

    # Move to the desired device
    model.to(device)
    return model


# In[21]:


def train(model: DivingViT, device: torch.device, train_data_loader: DataLoader, val_data_loader: DataLoader) -> Tuple[List[float], List[float], List[float]]:
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
    lr = 0.00005
    weight_decay = 0.01
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []
    val_spearman_correlations = []
    epochs = 250

    global hlr, hwtdc, ep
    hlr = lr
    hwtdc = weight_decay
    ep = epochs

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
                loss = F.mse_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())

            # Free memory
            del inputs, targets, outputs, loss
            gc.collect()
            #if args["gpu"]:
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
                    loss = F.mse_loss(outputs, targets)

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
        print(f"Epoch {epoch+1}/{epochs} \t train_loss: {np.mean(train_loss):.4f} \t val_loss: {np.mean(val_loss):.4f} \t val_spcoeff: {spearman_correlation(spearman_target, spearman_output):.4f}")

    return train_losses, val_losses, val_spearman_correlations


# In[18]:


def evaluate(model: DivingViT, device: torch.device, test_data_loader: DataLoader) -> None:
    """
    Evaluates the model on the testing dataset by computing the testing loss and spearman correlation.
    Args:
        model (DivingViT): model to evaluate
        device (torch.device): the device (cpu/gpu) to run on
        test_data_loader (DataLoader): testing dataset
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
                loss = F.mse_loss(outputs, targets)

            test_loss.append(loss.item())
            spearman_outputs.append(outputs)
            spearman_targets.append(targets)

    # Compute spearman coefficient
    spearman_output = torch.cat(spearman_outputs, dim=0) # concat across batches
    spearman_target = torch.cat(spearman_targets, dim=0) # concat across batches
    spearman_output = spearman_output[:, 0] * spearman_output[:, 1] * 30 # get final score
    spearman_target = spearman_target[:, 0] * spearman_target[:, 1] * 30 # get final score

    # Log loss and spearman correlation
    print(f"test_loss: {np.mean(test_loss):.4f} \t test_spcoeff: {spearman_correlation(spearman_target, spearman_output):.4f}")


# In[36]:


def plot_losses(train_losses: List[float], val_losses: List[float], val_spearman_correlations: List[float]) -> None:
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
    plt.show()
    plt.savefig("/scratch1/bodala/deco24.png")

    #print(f"Saved figure to {fig_filepath}")


# In[37]:


def main():
    # Parse commandline arguments
    #args = get_cmdline_arguments()

    logging.info("--- Start of Program ---")
    #logging.info(f"Command-Line Args: {args}")

    # Prepare dataset
    annotations = load_annotations("/scratch1/bodala/diving_data/final_annotations_dict.pkl")
    train_data, val_data, test_data = load_dataset(annotations)

    # Get the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    logging.info(f"Using device: {device}")
    # if args["gpu"] and torch.cuda.is_available():
    #     logging.info(f"GPU Total VRAM: {convert_size(torch.cuda.get_device_properties(device).total_memory)}")
    #     logging.info(f"GPU Reserved VRAM: {convert_size(torch.cuda.memory_reserved(device))}")
    #     logging.info(f"GPU Allocated VRAM: {convert_size(torch.cuda.memory_allocated(device))}")

    # Create the model
    model = create_model(device)

    # Train the model
    train_losses, val_losses, val_spearman_correlations = train(model, device, train_data, val_data)

    # Evaluate the model
    #if args["evaluate"]:
    evaluate(model, device, test_data)

    print("deco24.txt")
    f = open("/scratch1/bodala/deco24.txt", "a")
    f.write("learning rate:")
    f.write(str(hlr))
    f.write("weight decay:")
    f.write(str(hwtdc))
    f.write("batch size:")
    f.write(str(b_s))
    f.write("train_val_split_ratio:")
    f.write(str(tvr))
    f.write("FRAME_SELECTION_METHOD:")
    f.write(str(fsr))
    f.write("NUM_FRAMES:")
    f.write(str(nf))
    f.write("SPATIAL_SIZE:")
    f.write(str(ss))
    #f.write(m.summary)
    f.close()
    torch.save(m, "/scratch1/bodala/deco24.pth")

    plot_losses(train_losses, val_losses, val_spearman_correlations)

    logging.info("--- End of Program ---")


if __name__ == "__main__":
    main()
