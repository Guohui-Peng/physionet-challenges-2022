# Python example classifier code for the George B. Moody PhysioNet Challenge 2022

This project uses tensorflow version 2.9.1 and runs successfully in Docker environment.  

It supports running on GPU and can also run on CPU, I strongly recommend using Nvidia GPU.  

It takes about 3 hours to complete the training through GPU using Nvidia 3060 12G graphics card, and it takes about 200 hours to complete the training using CPU of AMD Ryzen 7 3700X.  

The default setting about Docker configuration of this project is to use GPU. If you don't have GPU, or decide to switch to CPU, please alter the first two lines of the Dockerfile in the following. Please comment the first line, and uncomment the second line. It is recommended to choose this method when GPU is not available.  

```Dockerfile
# FROM tensorflow/tensorflow:2.9.1-gpu
FROM tensorflow/tensorflow:2.9.1
```  

For detailed steps, please refer to "How do I run these scripts in Docker?"

Precautions:

1. If your computer is using Nvidia GPU acceleration for the first time, you need to confirm that you have installed the Nvidia graphics card driver. You may also need to install nvidia-container-runtime under Linux.  If you have already run other Nvidia GPU accelerated projects, there is no need to reinstall.  
For details, please refer to:  
[How to use TensorFlow Docker](https://www.tensorflow.org/install/docker)  
[Install NVIDIA Container Toolkit for Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)  
2. To enable GPU acceleration, you need to add the parameter `--gpus all` when running docker  

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that you can run your code reliably in other computing environments and operating systems.

To guarantee that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

    ```bash
        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs
    ```

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2022). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

    ```bash
        user@computer:~/example$ git clone https://github.com/Guohui-Peng/physionet-challenges-2022.git
    ```

4. Build a Docker image and run the example code in your terminal.

    ```bash
        user@computer:~/example$ ls
        model  physionet-challenges-2022  test_data  test_outputs  training_data

        user@computer:~/example$ cd physionet-challenges-2022/

        user@computer:~/example/physionet-challenges-2022$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/physionet-challenges-2022$ docker run --gpus all -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash

        root@[...]:/physionet# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py

        root@[...]:/physionet# python train_model.py training_data model

        root@[...]:/physionet# python run_model.py model test_data test_outputs

        root@[...]:/physionet# python evaluate_model.py labels test_outputs scores.csv
        [...]

        root@[...]:/physionet# exit
        Exit
    ```

## Analyze Challenge Data

There were 942 patient data.

The recorded data for each patient in the dataset were counted.

| Number of recordings | Number of patients |
|  ----  | ----:  |
| 1 | 62 |
| 2 | 156 |
| 3 | 123 |
| 4 | 588 |
| 5 | 10 |
| 6 | 3 |
| Total | 942 |

Recording lengths: 20608~258048

Recording classification qty:  
Unkown: 68  
Absent: 695  
Present: 179  

---

## What's in this repository?

This repository contains a simple example to illustrate how to format a Python entry for the George B. Moody PhysioNet Challenge 2022. You can try it by running the following commands on the Challenge training sets. These commands should take a few minutes or less to run from start to finish on a recent personal computer.

For this example, we implemented a random forest classifier with several features. You can use a different classifier, features, and libraries for your entry. This simpple example is designed **not** not to perform well, so you should **not** use it as a baseline for your model's performance.

This code uses four main scripts, described below, to train and run a model for the 2022 Challenge.

## How do I run these scripts?

You can install the dependencies for these scripts by creating a Docker image (see below) and running

    pip install requirements.txt

You can train and run your model by running

    python train_model.py training_data model
    python run_model.py model test_data test_outputs

where `training_data` is a folder with the training data files, `model` is a folder for saving your model, `test_data` is a folder with the test data files (you can use the training data for debugging and cross-validation), and `test_outputs` is a folder for saving your model outputs. The [2022 Challenge website](https://physionetchallenges.org/2022/) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2022) and running

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage; `outputs` is a folder containing files with your model's outputs for the data; `scores.csv` (optional) is a collection of scores for your model; and `class_scores.csv` (optional) is a collection of per-class scores for your model.

## Which scripts I can edit?

We will run the `train_model.py` and `run_model.py` scripts to train and run your model, so please check these scripts and the functions that they call.

Please edit the following script to add your training and testing code:

* `team_code.py` is a script with functions for training and running your model.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model.
* `run_model.py` is a script for running your trained model.
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your models, please edit the `train_challenge_model` function in the `team_code.py` script. Please do not edit the input or output arguments of the `train_challenge_model` function.

To load and run your trained model, please edit the `load_challenge_model` and `run_challenge_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of the functions of the `load_challenge_model` and `run_challenge_model` functions.

## How do I learn more?

Please see the [Challenge website](https://physionetchallenges.org/2022/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [Challenge website](https://physionetchallenges.org/2022/)
* [MATLAB example classifier code](https://github.com/physionetchallenges/matlab-classifier-2022)
* [Scoring code](https://github.com/physionetchallenges/evaluation-2022)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2022/faq/) 
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/) 
