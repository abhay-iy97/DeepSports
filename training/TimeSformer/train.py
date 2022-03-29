import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from timesformer.models.vit import TimeSformer
import matplotlib.pyplot as plt
import random
from datetime import datetime

def getTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

class DivingViT(nn.Module):
    def __init__(self, pretrained_model, drop_prob=0.5, freeze=False, num_classes=1):
        """
        Builds upon the TimeSformer model with additional MLP layers of (512 > 256 > num_classes).

        Args:
            pretrained_model (TimeSformer): A pretrained instance of TimeSformer of 8 frames and 224x224.
            drop_prob (float, optional): Drop probability for dropout after the TimeSformer model. Defaults to 0.5.
            freeze (bool, optional): Whether to freeze the pretrained model weights or also add them to the gradient updates. Defaults to False.
            num_classes (int, optional): The number of output classes. Defaults to 1.
        """
        super(DivingViT, self).__init__()
        
        self.pretrained_model = pretrained_model
        if freeze:
            pretrained_model.requires_grad_ = False
            
        self.dropout = nn.Dropout(p=drop_prob)
        self.linears = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        """
        Computes the forward pass of the model. The input must be in the shape (batch, channels, frames, height, width).
        The output will have the shape of (batch, num_classes).

        Args:
            x (torch.Tensor): The input to the model. It is in the shape (batch, channels, frames, height, width).

        Returns:
            torch.Tensor: Returns the output of the model. It is in the shape (batch, num_classes).
        """
        out = self.pretrained_model(x) # (batch, 768)
        out = self.dropout(out) # (batch, 768)
        out = self.linears(out) # (batch, num_classes)
        return out


# A data class to hold a single video clip
class VideoClip:
    PATH_TO_FRAMES = "/scratch1/adiyer/deepsports_dataset/whole_videos_frames/"
    def __init__(self, video_num, start_frame, end_frame, difficulty, final_score):
        self.video_num = video_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.difficulty = difficulty
        self.final_score = final_score
    
    def load(self):
        # Computes the number of frames in the clip
        clip_num_frames = self.end_frame - self.start_frame + 1
        
        # Creates the video clip in torch with (channels x frames x height x width)
        clip = torch.empty(size=(3, 8, 224, 224), dtype=torch.float32)
        
        # Since we are constrained to 8 frames, pick 8 frames from the entire clip
        step_size = clip_num_frames // 8 + 1
        
        # Convertor between PIL to Pytorch Tensor
        convert_tensor = transforms.ToTensor()
        for idx, frame_num in enumerate(range(self.start_frame, self.end_frame + 1, step_size)):
            img_path = VideoClip.PATH_TO_FRAMES + f"{self.video_num:02d}/{frame_num:06d}.jpg"
            img = Image.open(img_path).resize((224, 224))
            tensor = convert_tensor(img).type(dtype=torch.float32) # (channels x height x width)
            clip[:, idx, :, :] = tensor
        
        # Generate the outputs/targets for this clip
        target = torch.tensor([self.final_score], dtype=torch.float32)
        return clip, target

# Loads all the video clips from the dataset textfile that Mohammad generates. They're in the format video_num,start_frame,end_frame,difficulty,final_score
# where video_num is the video number it came from (01.mp4 is 1, 02.mp4 is 2, etc.)
def load_dataset_file(filename):
    loaded_clips = []
    with open(filename, "r") as f:
        # Read all lines, but skip the header
        for line in f.readlines()[1:]:
            # format: video_num,start_frame,end_frame,difficulty,final_score
            split = line.split(",")
            
            # TODO: Add better checking and error handling
            video_num = int(split[0])
            start_frame = int(split[1])
            end_frame = int(split[2])
            difficulty = float(split[3])
            final_score = float(split[4])
            loaded_clips.append(VideoClip(video_num, start_frame, end_frame, difficulty, final_score))
    print(getTime(), f" Found {len(loaded_clips)} in {filename}.")
    return loaded_clips

# print(f"Loaded {len(clips_dataset)} clips in the dataset.")

class MyDataset(Dataset):
    def __init__(self, clips_dataset):
        self.clips_dataset = clips_dataset
        
    def __len__(self):
        return len(self.clips_dataset)

    def __getitem__(self, index):
        return self.clips_dataset[index].load()

def splitDataset(clips_dataset):
    num_clips = len(clips_dataset)
    num_train = int(num_clips * 0.7)
    num_val = int(num_clips * 0.15)
    num_test = num_clips - num_train - num_val
    train_split, val_split, test_split = random_split(clips_dataset, (num_train, num_val, num_test))

    print(getTime(), f"Split dataset into {num_train} training, {num_val} validation, and {num_test} testing samples.")
    train_dataset = MyDataset(train_split)
    val_dataset = MyDataset(val_split)
    test_dataset = MyDataset(test_split)
    return train_dataset, val_dataset, test_dataset

def train(batch_size, train_dataset, val_dataset):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    learning_rate = 1e-5

    path = "/scratch1/adiyer/timesformer-model/TimeSformer_divST_8x32_224_K400.pyth"
    pretrained_model = TimeSformer(img_size=224, num_classes=768, num_frames=8, attention_type='divided_space_time',  pretrained_model=path)
    # model = DivingViT(pretrained_model, drop_prob=0.5, freeze=False, num_classes=1).to(device)
    
    model = DivingViT(pretrained_model, drop_prob=0.5, freeze=False, num_classes=1)
    model = nn.DataParallel(model)
    model.to(device)

    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    plot_data = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        
        # Training
        train_loss = []
        model.train()
        for (inputs, targets) in train_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        plot_data["train_loss"].append(np.mean(train_loss))
        
        # Validation
        val_loss = []
        model.eval()
        with torch.no_grad():
            for (inputs, targets) in val_data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
        plot_data["val_loss"].append(np.mean(val_loss))
        
        # Print losses
        print(getTime(), f"Epoch {epoch+1}/{epochs} \t train_loss: {np.mean(train_loss):.4f} \t val_loss: {np.mean(val_loss):.4f}")
    return model, plot_data


def plotData(plot_data):
    # Plot the training and validation losses

    epochs_range = list(range(1, epochs+1))

    fig = plt.figure(figsize = (10, 5))

    ax1 = plt.subplot(121)
    ax1.plot(epochs_range, plot_data["train_loss"], label="train_loss")
    ax1.plot(epochs_range, plot_data["val_loss"], label="val_loss")
    ax1.title.set_text("Loss")
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    plt.savefig('loss.png')
    # plt.show()

def evaluate(batch_size, test_dataset, model):
    # Evaluate on the testing dataset
    test_loss = []
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        for (inputs, targets) in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
    print(getTime(), f" test_loss: {np.mean(test_loss):.4f}")

if __name__ == '__main__':
    clips_dataset = []
    print(getTime(), f" Loading dataset...")
    # Load all the datasets
    for i in ["01", "02", "03", "04", "07", "09", "10", "13", "14", "17", "18", "22", "26"]:
        clips_dataset.extend(load_dataset_file(f"dataset/{i}.csv"))

    print(getTime(), f"Loaded {len(clips_dataset)} clips in the dataset.")
    train_dataset, val_dataset, test_dataset = splitDataset(clips_dataset)

    # print(torch.cuda.is_available())
    batch_size = 8 # change this depending on the available VRAM
    model, plot_data = train(batch_size, train_dataset, val_dataset)
    evaluate(batch_size, test_dataset, model)
    print(getTime(), f" DONE")
