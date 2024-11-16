---
aliases:
- Installing-Dependencies
date: '2024-11-13'
image: images/Python.png
layout: post
description: "Guide to install dependencies for fastAI."
title: "How to setup WSL and Python for fastAI"
categories:
- computer_usage

---


# How to install WSL in windows

WSL implies for Windows Subsystem for Linux (WSL), where you can run a Linux environment on your Windows machine without the hassle of dual-booting or setting up a virtual machine. This guide will take you through the steps to install WSL, focusing on WSL 2, which is faster, better, and more compatible than its predecessor.

The basic prerequisits are that You need Windows 10 version 2004 or higher (Build 19041 and above) or Windows 11. If you’re still rocking an older version, it might be time for an upgrade—like swapping out your flip phone for a smartphone!

If you get any error related to virtualization. You need to have virtualization enabled which you can do by enabling Hyper V in your device

## Step 1: Enable WSL:

**Open PowerShell as Administrator**: Search for "PowerShell" in the Start menu. Right-click on it and select "Run as administrator". If you get intimidated by the black screen don't panik you are inside terminal which helps us to execute commands. More on it later!
**Run the Installation Command**: In the PowerShell window, type the following command and press Enter

```bash
wsl --install
```

This command will enable all the necessary features for WSL, download the Linux Kernel, and install Ubuntu as your default distribution. A restart may be required.

If ubuntu is not your cup of tea then you can choose another distribution by

```bash
wsl --install -d Debian
```

for debian

## Step 2: Make a user account

After installation, launch your installed Linux distribution from the Start menu

You’ll be prompted to create a **username** and **password**. Choose wisely this is your secret identity! Remember, while typing your password, nothing will appear on the screen; this is normal behavior in Linux. It’s not broken; it’s just shy.


## Step 4: Update the system

Now that you’ve got your Linux environment set up, let’s make sure it’s up to date: you can do that by just typing
```bash
sudo apt update
sudo apt upgrade
```

# Terminal and CLI

Okay so now as you are staring at that black window let's talk a little bit about terminal and cli

A **terminal** is essentially a user interface that allows you to interact with your computer using text-based commands.

**CLI** or command line interface is the way we interact with the computer in text format.

Now by now 2 questions will arise in your mind by now -

- Why are we using WSL why not use windows
The programs we will be using are much more easier to operate on Linux than windows, you can modify linux in many ways in order to make your life easier than it is in windows.

- Why are we using CLI why not use the graphical apps which are more beautiful ?
Once you get the hang of it, typing commands can be faster than clicking through menus. You can write scripts to automate repetitive tasks saving time and effort.

## Some basic definitions and tools handy in cli

So here are some popular commands and definitions which we need to keep in mind while we use CLI

directory - for most cases directory is your 'folder' we can store different files and directories in a directory.

 `ls` : Lists the contents of a directory. Use options like `-a` for hidden files or `-l` for detailed information.

`cd [directory]`: Changes the current directory to the specified one. Use `cd ..` to go back

`mkdir [dirname]`: Creates a new directory with the specified name.

`rm -rf [dirname]` : Removes a directory and everything in it. This is done without confirmation so know what you are doing

# Installing python and other dependencies

Now if you type `python` on your terminal you will see that some application is getting activated.

Now the python that has been activated is not the python we are gonna use. This python is used by the system to run stuff. We are gonna something called **miniforge** for python.

Here are the steps in setting up miniforge in your system -

## Step 1: Install wget in you system

you can do so by executing -
```bash
sudo apt install wget
```

## Step 2: Download the setup script for miniforge

you can do that by following these commands

### for x86_64

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

### for arm

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
```

this will download the Miniforge installer in your WSL

## Step 3: Install miniforge in your WSL

execute the following command -

### For x86_64
```bash
bash Miniforge3-Linux-x86_64.sh
```

### For Arm

```bash
bash Miniforge3-Linux-aarch64.sh
```

A simple setup will appear which will ask you to accept the licence agreement


Then a prompt will appear telling you that setup will install miniforge in your home directory say yes to it

And then miniforge will install not only python but a whole bunch of libraries which will come handy to us later.

restart you shell by executing `bash` to make it initialize miniforge

## Step 4: Agree to initialize it

It will ask you if you want to initialize it whenever you start your machine and say yes to it.

What it will do is that it will execute python everytime we launch wsl

## Mamba and conda

Mamba and Conda are both powerful tools used for package and environment management in Python. They install everything we need for python and help us create virtual environments. This brings us to -
## Step 5: Enable virtual environment

A virtual environment is a self-contained directory that allows you to manage dependencies for different projects without stepping on each other’s toes. It will help seperate python we need with the system python

To create a virtual environment just execute:

```bash
mamba create -n fastai_env python=3.9
```

this will create a python virtual environment

but that's not all we also need to activate it for it to work this is done by executing

```bash
mamba activate fastai_env
```

- A pro tip - You can activate the virtual environment everytime you want by putting it in your .bashrc file. You can do that by

```bash
nano .bashrc
```

this opens a text editor which we will use to edit files. Edit it by adding `mamba activate fastai_env` at the end of the file.

then press ctrl + x to exit and y to save the file.


## Step 6: Installing ipython and jupyter lab

### Need for these tools
If you’re ready to kick your AI game up a notch, you need to get cozy with **IPython**, **JupyterLab**, **nbdev**. We have so many good reasons to use these tools

We will use Ipython because

- It helps us to display media like Images, Videos etc
- IPython includes special commands (prefixed with `%` or `%%`) that allow you to perform tasks like timing execution or running shell commands seamlessly.
- With improved tracebacks and debugging capabilities, it makes troubleshooting easier.

We will use Jupyter Lab because -

- Multi document UI - Open multiple notebooks, text files, and terminals all at once. You can juggle your projects too. No more switching tabs.

- **Extensions Galore**: Want to customize your experience? JupyterLab supports extensions that let you add new features or integrate with other tools. It’s like dressing up your notebook in the latest fashion make it yours

- **Interactive Widgets**: Create interactive visualizations and controls right in your notebooks. Want to tweak parameters on the fly? Just slide those sliders.

nbdev is Important because

1. **Literate Programming**: Write code, tests, and documentation together in Jupyter notebooks, enhancing readability and maintainability.
2. **Automatic Documentation**: Generate up-to-date documentation directly from your notebooks, streamlining the process of creating and maintaining libraries.
3. **Integrated Testing**: Write and run unit tests within your notebooks, ensuring code quality with automatic execution during builds and CI/CD processes.


Basically they help is making the experiance smoother for the journey.
### Installation

Here is the command to install these tools -

```bash
mamba install ipython jupyterlab nbdev
```

## Step 7: Install pytorch

PyTorch is a powerful and flexible tool for deep learning and machine learning projects. Here are some of its features

- **Dynamic Computation Graphs**: Allows changes to the model on-the-fly, making debugging easier.
- **Tensor Operations**: Supports efficient tensor computations with GPU acceleration for faster processing.
- **User-Friendly**: Intuitive and Pythonic interface, great for beginners and experienced users alike.
- **Rich Ecosystem**: Includes libraries for building neural networks and optimization, simplifying model development.
- **Strong Community**: Extensive documentation and active community support for learning and troubleshooting.

### Installation

Here is how to get it installed -


- For devices with Nvidia GPU - if your device have an Nvidia GPU then you can install pytorch with CUDA support by executing following command -

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

You will also need to install the cuda toolkit for pytorch to work. You can do it by executing -

```

wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
```

Then run

```bash
sudo sh cuda_12.6.2_560.35.03_linux.run
```



- For devices with integrated graphics - If you are poor student like me and have device with integrated graphics then you should install pytorch by using following command -

```bash
mamba install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Step 8: Install FastAI

Fast AI will be the main library we will be working with. It is designed to make deep learning accessible to everyone, regardless of their coding experience. It is is built on top of PyTorch designed to make the complex realm of artificial intelligence as approachable as your favorite recipe for instant noodles.

### Installation

To install it just execute this command in your terminal -

```bash
mamba install -c fastai fastai
```

And you have successfully installed the tools required for the course.

# Optional stuff

## Vim

Vim is the most supereme cli text editor that one can use in linux. Here is how to install and use it

1. **Open WSL Terminal**:
   - Launch your WSL terminal.

2. **Update Package List**:
   - Before installing any software, update the package list by running:
     ```bash
     sudo apt update
     ```

3. **Install Vim**:
   - Install Vim by executing the following command:
     ```bash
     sudo apt install vim -y
     ```
   - This command retrieves and installs Vim along with its necessary components.

4. **Launching Vim**:
   - To create or edit a file, use the command:
     ```bash
     vim filename.txt
     ```
   - Replace `filename.txt` with your desired file name. If the file does not exist, Vim will create it.

5. **Basic Navigation and Editing**:
   - Upon opening a file, you start in **Normal mode**. Press `i` to switch to **Insert mode**, where you can type text.
   - To return to Normal mode, press `Esc`.
   - You can go up down left and right in the document by either using arrow keys or using h,j,k,l keys (right,down,up,left).

6. **Saving and Exiting**:
   - To save changes, type `:w` and press `Enter`.
   - To exit Vim, type `:q` and press `Enter`. If you want to save and exit simultaneously, type `:wq`.


## Ranger

Ranger is a cli file manager which you can use to navigate through files easily.

To install and use Ranger, a VIM-inspired file manager, in Windows Subsystem for Linux (WSL), follow these detailed steps:

## Installation Steps

1. **Open WSL Terminal**:
   - Launch your WSL terminal

2. **Install Prerequisites**:
   - Update the package list and install the necessary packages (`make`, `git`, and `vim`) by running:
     ```bash
     sudo apt update
     sudo apt install make git vim -y
     ```

3. **Install Ranger**:

   ```bash
   sudo apt install ranger -y
   ```

4. **Configure Ranger**:
   - Run Ranger once to create the configuration directory:
     ```bash
     ranger
     ```


## Using Ranger

1. **Launching Ranger**:
   - Start Ranger by typing:
     ```bash
     ranger
     ```

2. **Interface Overview**:
   - The interface is divided into three columns:
     - **Left Column**: Displays the parent directory.
     - **Middle Column**: Shows contents of the current directory.
     - **Right Column**: Provides a preview of the selected file or folder.

3. **Basic Navigation**:
   - Use the following keys to navigate:
     - Arrow keys or `h`, `j`, `k`, `l` for left, down, up, and right respectively.
     - `Enter` to open a file or directory.
     - `q` to quit.

### Copying, Pasting, and Deleting Files

- **Copying Files**:
  - To copy a file or directory, navigate to it and press `yy` (yank).
  - To copy multiple files, select them using `Space` and then press `yy`.

- **Pasting Files**:
  - Navigate to the destination directory and press `p` to paste the copied files.

- **Deleting Files**:
  - To delete a file or directory, navigate to it and press `dd` (delete).
  - Confirm the deletion when prompted.


