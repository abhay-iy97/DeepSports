# Login CARC

Connect to CARC Discovery Node with SSH (i.e. PuTTY). Replace <code>username</code> with your USC username (i.e. <code>adiyer</code> or <code>alalim</code>):
   - **Host Name (or IP):** <code>username@discovery.usc.edu</code>
   - **Port:** 22
   - **Note:** If possible, set seconds between keepalives to 60 to ping every 60s to maintain a long-lasting connection.

# Clone Repo

To simplify the entire process, clone the entire repo to your home directory (i.e. <code>adiyer/home1/</code>) by running:

```
cd "/home1/$USER"
git clone -b main https://github.com/abhay-iy97/DeepSports.git
```

- **Note:** When it asks for your GitHub password, you may have to setup a personal access token for GitHub and use that instead (click [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for quick guide).

# Dataset Preparation

In this step, we download the videos by running the following command. Note that it is not part of a job script because job's are not able to CURL from outside sources.

```
sh "/home1/$USER/DeepSports/data_preparation/download_videos.sh"
```

After that is done, to extract frames of the MTL-AQA videos, run:

```
sbatch "/home1/$USER/DeepSports/job_files/prepare_dataset.job"
```

To check on the job's status, you can run:
```
squeue --me
```

# Downloading a Pre-Trained Model

TimeSformer models pretrained on Kinetics-400 (K400), Kinetics-600 (K600), Something-Something-V2 (SSv2), and HowTo100M datasets are shown in the table below. First, copy the download URL of the model you wish to use below, then modify the command below to use the link you copied.
- **Note:** Modify the dropbox link you copied to end with **dl=1** not **dl=0**.

```
cd "/home1/$USER/DeepSports/training/TimeSformer"
curl https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=1 -L -o TimeSformer_divST_8x32_224_K400.pyth
```



| name           | dataset | # of frames | spatial crop | acc@1 | acc@5 | url                                                                                           |
| -------------- | ------- | ----------- | ------------ | ----- | ----- | --------------------------------------------------------------------------------------------- |
| TimeSformer    | K400    | 8           | 224          | 77.9  | 93.2  | [model](https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0)  |
| TimeSformer-HR | K400    | 16          | 448          | 79.6  | 94.0  | [model](https://www.dropbox.com/s/6f0x172lpqy3oxt/TimeSformer_divST_16x16_448_K400.pyth?dl=0) |
| TimeSformer-L  | K400    | 96          | 224          | 80.6  | 94.7  | [model](https://www.dropbox.com/s/r1iuxahif3sgimo/TimeSformer_divST_96x4_224_K400.pyth?dl=0)  |

| name           | dataset | # of frames | spatial crop | acc@1 | acc@5 | url                                                                                           |
| -------------- | ------- | ----------- | ------------ | ----- | ----- | --------------------------------------------------------------------------------------------- |
| TimeSformer    | K600    | 8           | 224          | 79.1  | 94.4  | [model](https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=0)  |
| TimeSformer-HR | K600    | 16          | 448          | 81.8  | 95.8  | [model](https://www.dropbox.com/s/ft1e92g2vhvxecv/TimeSformer_divST_16x16_448_K600.pyth?dl=0) |
| TimeSformer-L  | K600    | 96          | 224          | 82.2  | 95.6  | [model](https://www.dropbox.com/s/857rx6xeclxfhdg/TimeSformer_divST_96x4_224_K600.pyth?dl=0)  |

| name           | dataset | # of frames | spatial crop | acc@1 | acc@5 | url                                                                                        |
| -------------- | ------- | ----------- | ------------ | ----- | ----- | ------------------------------------------------------------------------------------------ |
| TimeSformer    | SSv2    | 8           | 224          | 59.1  | 85.6  | [model](https://www.dropbox.com/s/tybhuml57y24wpm/TimeSformer_divST_8_224_SSv2.pyth?dl=0)  |
| TimeSformer-HR | SSv2    | 16          | 448          | 61.8  | 86.9  | [model](https://www.dropbox.com/s/9t68uzk8w2fpfnv/TimeSformer_divST_16_448_SSv2.pyth?dl=0) |
| TimeSformer-L  | SSv2    | 64          | 224          | 62.0  | 87.5  | [model](https://www.dropbox.com/s/3f1rm2al8mhprwa/TimeSformer_divST_64_224_SSv2.pyth?dl=0) |

| name        | dataset   | # of frames | spatial crop | single clip coverage | acc@1 | url                                                                                                |
| ----------- | --------- | ----------- | ------------ | -------------------- | ----- | -------------------------------------------------------------------------------------------------- |
| TimeSformer | HowTo100M | 8           | 224          | 8.5s                 | 56.8  | [model](https://www.dropbox.com/s/9v8hcm88b9tc6ff/TimeSformer_divST_8x32_224_HowTo100M.pyth?dl=0)  |
| TimeSformer | HowTo100M | 32          | 224          | 34.1s                | 61.2  | [model](https://www.dropbox.com/s/4roflx4q1gscu85/TimeSformer_divST_32x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 64          | 448          | 68.3s                | 62.2  | [model](https://www.dropbox.com/s/15bvqltl1j5vyp3/TimeSformer_divST_64x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 96          | 224          | 102.4s               | 62.6  | [model](https://www.dropbox.com/s/t2mzgahnfhgakma/TimeSformer_divST_96x32_224_HowTo100M.pyth?dl=0) |

# Setting Up Environment for TimeSformer

Run the following, then restart your SSH connection with CARC to restart the shell so that the environment changes take effect.

```
module load python/3.9.2
module load anaconda3/2021.05
conda create -n timesformer python=3.7 -y
conda init
```

After restarting, run this to install all dependencies:

```
module load python/3.9.2
module load anaconda3/2021.05
conda activate timesformer
pip install torch torchvision fvcore simplejson einops timm
conda install av -c conda-forge
pip install psutil scikit-learn opencv-python tensorboard matplotlib
pip install torchsort
```

At this point, you should be set and ready to train models!

# Training a Diving ViT Model

First, modify the contents of <code>/home1/\<username\>/DeepSports/model_train_config.sh</code> to the parameters you want, here's a full argument list:

| Argument                  | Description                                                                                                                        | Default Value                             | Possible Values                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| **loglevel**              | The level of logging in the application.                                                                                           | <code>INFO</code>                         | <code>[DEBUG, INFO, WARNING, ERROR, CRITICAL]</code>                                    |
| **gpu**                   | Whether to use GPU for training.                                                                                                   | <code>True</code>                         | <code>[False, True]</code>                                                              |
| **root_dir**              | The path to the root directory containing the frame images.                                                                        | <code>./</code>                           | <code>directory path</code>                                                             |
| **train_path**            | The filepath to the train split pickle file.                                                                                       | <code>./train_split_0.pkl</code>          | <code>.pkl file path</code>                                                             |
| **test_path**             | The filepath to the test split pickle file.                                                                                        | <code>./test_split_0.pkl</code>           | <code>.pkl file path</code>                                                             |
| **batch_size**            | The batch size used in training.                                                                                                   | <code>4</code>                            | <code>[1, inf]</code>                                                                   |
| **epochs**                | The number of epochs used in training.                                                                                             | <code>5</code>                            | <code>[1, inf]</code>                                                                   |
| **learning_rate**         | The learning rate used in training.                                                                                                | <code>0.00001</code>                      | <code>[0.0, inf]</code>                                                                 |
| **weight_decay**          | The weight decay used in training.                                                                                                 | <code>0.00001</code>                      | <code>[0.0, inf]</code>                                                                 |
| **momentum**              | The momentum used in SGD/RMSProp optimizers for training.                                                                          | <code>0.9</code>                          | <code>[0.0, inf]</code>                                                                 |
| **train_val_split_ratio** | The ratio in which the training and validation datasets are split.                                                                 | <code>0.8</code>                          | <code>[0.0, 1.0]</code>                                                                 |
| **frame_num**             | The number of frames to use in each clip.                                                                                          | <code>8</code>                            | <code>[1, inf]</code>                                                                   |
| **frame_method**          | The algorithm to use to sample frames from a long clip.                                                                            | <code>space_fixed</code>                  | <code>[random, spaced_fixed, spaced_varied, spaced_fixed_new, spaced_varied_new]</code> |
| **spatial_size**          | The image size in pixels that they will be resized to.                                                                             | <code>224</code>                          | <code>[1, inf]</code>                                                                   |
| **freeze**                | Whether to freeze the gradients in the TimeSformer model.                                                                          | <code>False</code>                        | <code>[False, True]</code>                                                              |
| **dropout**               | The dropout value used in MLP.                                                                                                     | <code>[0.5, 0.5]</code>                   | <code>list of drop prob, [0.0, 1.0] each. dropout is before the linear</code>           |
| **activation**            | The activation function used in the MLP network.                                                                                   | <code>None</code>                         | <code>[None, ReLU, LeakyReLU, ELU, GELU]</code>                                                               |
| **topology**              | The hidden neurons topology between the TimeSformer model and the final output layer.                                              | <code>[512, 256]</code>                   | <code>list of ints [1, inf] each</code>                                                 |
| **output**                | The output filepath for the losses figure.                                                                                         | <code>./losses.png</code>                 | <code>.png file path</code>                                                             |
| **annotation_path**       | The path to the final annotations dict pickle file.                                                                                | <code>./final_annotations_dict.pkl</code> | <code>.pkl file path</code>                                                             |
| **attention_type**        | The type of attention used in the transformer.                                                                                     | <code>divided_space_time</code>           | <code>[divided_space_time, space_only, joint_space_time]</code>                         |
| **optimizer**             | The optimizer used in training.                                                                                                    | <code>AdamW</code>                        | <code>[Adam, AdamW, SGD, RMSProp]</code>                                                |
| **patch_size**            | The patch size used in the transformer.                                                                                            | <code>16</code>                           | <code>[1, inf]</code>                                                                   |
| **embed_dim**             | The embed dimensions output from the transformer.                                                                                  | <code>768</code>                          | <code>[1, inf]</code>                                                                   |
| **pretrained_model**      | The filepath to the pretrained .pyth model for the TimeSformer. If set to 'scratch', it will train a TimeSformer from scratch.     | <code>scratch</code>                      | <code>path/to/model.pyth or scratch</code>                                              |
| **evaluate**              | Whether to use evaluate on testing dataset.                                                                                        | <code>False</code>                        | <code>[False, True]</code>                                                              |
| **normalize**             | Whether to use normalize the RGB channels in the video clips as a preprocessing step.                                              | <code>False</code>                        | <code>[False, True]</code>                                                              |
| **data_aug**              | Whether to use randomly resize and crop the video clips as a preprocessing step.                                                   | <code>False</code>                        | <code>[False, True]</code>                                                              |
| **amsgrad**               | Whether to use amsgrad for Adam/AdamW optimizer.                                                                                   | <code>False</code>                        | <code>[False, True]</code>                                                              |
| **videos**                | The videos directory name to use. Default value is 'all'. To use select directories, only list their directory names, i.e. '01 02' | <code>all</code>                          | <code>all or list of directories (i.e. 01 02)</code>                                    |
| **loss_mse_weight**                | The weight given to the MSE loss. | <code>1</code>                          | <code>[-inf, inf]</code>                                    |
| **loss_spcoef_weight**                | The weight given to the differentiable Spearman Correlation loss. | <code>0</code>                          | <code>[-inf, inf]</code>                                    |
| **use_decoder**                | Whether to use a Transformer Decoder or revert to using MLP. | <code>False</code>                          | <code>[False, True]</code>                                    |

Next, you may want to modify the resources allocation of the training job (gpu, time, etc...) in <code>/home1/\<username\>/DeepSports/job_files/train_model.job</code> and then run:

```
sbatch "/home1/$USER/DeepSports/job_files/train_model.job"
```

# Final Directory View

The following tree-view is what you should expect at the end of your setup.

```bash
.
├── /home1/
│   ├── <username>/
│       ├── DeepSports/
│            ├── data_preparation/
│            |   └── ...
│            ├── job_files/
│            |   └── ...
│            ├── training/
│            |   └── ...
│            └── model_train_config.sh
│
├── /scratch1/
│   ├── <username>/
│       ├── DeepSports_dataset/
│            ├── whole_videos/
│            |   ├── 01.mp4
│            |   ├── 02.mp4
│            |   ├── 03.mp4
│            |   ├── 04.mp4
│            |   ├── 05.mp4
│            |   ├── 06.mp4
│            |   ├── 07.mp4
│            |   ├── 09.mp4
│            |   ├── 10.mp4
│            |   ├── 13.mp4
│            |   ├── 14.mp4
│            |   ├── 17.mp4
│            |   ├── 18.mp4
│            |   ├── 22.mp4
│            |   └── 26.mp4
│            └── whole_videos_frames/
│                ├── 01/
│                |   └── ...
│                ├── 02/
│                |   └── ...
│                ├── 03/
│                |   └── ...
│                ├── 04/
│                |   └── ...
│                ├── 05/
│                |   └── ...
│                ├── 06/
│                |   └── ...
│                ├── 07/
│                |   └── ...
│                ├── 09/
│                |   └── ...
│                ├── 10/
│                |   └── ...
│                ├── 13/
│                |   └── ...
│                ├── 14/
│                |   └── ...
│                ├── 17/
│                |   └── ...
│                ├── 18/
│                |   └── ...
│                ├── 22/
│                |   └── ...
│                └── 26/
│                    └── ...
```