---
aliases:
- Lesson-1-Making-a-Cyclist-recogniser
date: '2024-11-15'
image: images/lesson1.png
layout: post
description: "Documentation for lesson 1 of fastAI practical deep learning for coders."
title: "Making a Cyclist recognizer"
categories:
- computer_usage

---

The evolution of Machine Learning has transformed from a concept once deemed nearly impossible to a technology that is now easily accessible and widely utilized. It was considered so ridiculous in the early days that people joked about it. Here is one example:

![XKCD comic from 2015](posts/images/xkcd.png)

Above is an xkcd comic which shows how people joked about it. The good news is that we are going to make a computer vision model in this lesson today. So get excited!


# The Evolution of Neural networks
## Before Neural Networks

In the era before neural networks, people used a lot of workforce to identify images, then many mathematicians and computer scientists to process those images and create separate features for each one of them. After a lot of time and processing, they would fit it into a machine learning model. It became successful, but the problem was that making these models took a lot of time and energy, which was inefficient and tedious.

## The first neural network

Back in 1957 a neural network was described as something like a program. So in a traditional program we have some inputs and then we put them in program which have functions, conditionals, loops etc and these give us the result.

In deep learning the program is replaced by Model and we now also have weights(Also called parameters) with inputs. The model is not anymore a bunch of conditionals and loops and things. In case of a neural network it is a mathematical function which takes the inputs, multiplies them together by the weights and adds them up. And it does that will all the sets of inputs. And thus a neural network is formed

Now a model will not do anything useful unless these weights are carefully chosen, so we start by these weights being random. Initially these networks don't do anything useful.

We then take the inputs and weights put them in our model and get the results. The we decide how good they are, this is done by a number called loss. Loss describe how good the results are, think of it as something like accuracy. After we get loss we use it to update our weights and then repeat this process again and again, we get better and better results.

Once we do this enough times we stop putting inputs and weights and replace it with inputs and get some outputs.
## How Modern Neural Networks Work
With modern neural network methods, we don't teach the model features; we make them learn features. It is done by breaking the image into small parts and assigning them features (often called layer 1 features). After doing this for many images, we combine them to create more advanced features. So we train the basic neural network and make it a more advanced neural network, creating a kind of feature detector that finds the related features.

Coding these features would be very difficult, and many times you wouldn't even know what to code. This is how we make neural networks more efficient by not making them by code but by making them learn.

## Misconceptions About Deep Learning

As we saw earlier, to train a computer vision model, we didn't need expensive computers, we didn't need very high-level math, and we didn't need lots of data. This is the case with much of deep learning which we will learn. There will be some math that will be needed but mostly, either we will teach you the little bits, or we will refer you to some resources.

## PyTorch vs TensorFlow

In recent years, PyTorch is increasingly used in research while TensorFlow is declining in popularity. The library which is used in research is more likely to be used in industry; therefore, we will be using PyTorch for learning Deep Learning.

Another thing to note is that sometimes PyTorch uses a lot of code for some really basic tasks, and this is where fastai comes into play. The operations which are really lengthy to implement in PyTorch can be done with very few lines of code with fastai. This is not because PyTorch is bad but because PyTorch is designed so that many good things can be built on top of it.

The problem with having lots of code is that it increases the chances of mistakes. In fastai, the code you don't write is code that the developers have found best practices for and implemented for you.

## Jupyter Notebook

Jupyter notebook is a web-based application which is widely used in academia and teaching, and it is a very powerful tool to experiment, explore, and build with.

Nowadays, most people don't run Jupyter notebooks on their own local machines but on cloud servers. If you go to course.fast.ai, you can see how to use Jupyter and cloud servers. One of the good ones is Kaggle. Kaggle doesn't only have competitions but also has cloud servers where you can train neural networks. You can learn more about it at https://course.fast.ai/Resources/kaggle.html.

# Making a Traffic Recognizer

Let's say you want to make a self-driving car. A big challenge for it would be to identify between cyclists and pedestrians, so we are going to do that now. We are going to make a computer vision model that can differentiate between cyclists and pedestrians.

## Import Statements
```python
from duckduckgo_search import DDGS
from fastcore.all import *
import time
import json
from fastdownload import download_url
from fastai.vision.all import *
```
These lines import necessary libraries:
- `DDGS`: DuckDuckGo search API for finding images
- `fastcore`: Utility functions for deep learning
- `fastdownload`: For downloading files from URLs
- `fastai`: Deep learning library built on PyTorch
- `time`: For time-related operations

Note: You might get an error for duckduckgo_search while executing this part. Don't panic - just go to the console and execute:
```
pip install duckduckgo-search
```
This will install duckduckgo-search in your notebook.

## Image Search Function
```python
def search_images(keywords, max_images=400):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot("image")
```
This function:
- Takes search keywords and maximum number of images
- Uses DuckDuckGo to search for images
- Returns a list of image URLs
- `L()` creates a fastai list
- `itemgot("image")` extracts just the image URLs from the search results

## Initial Test Downloads
```python
urls = search_images("pedestrians", max_images=1)
print(urls[0])
dest = "pedestrians.jpg"
download_url(urls[0], dest, show_progress=False)
im = Image.open(dest)
```
This section:
- Searches for one pedestrian image
- Downloads it as 'pedestrians.jpg'
- Opens it to verify the download worked

Now we all know that computers don't understand images, but the good news is that computers can understand numbers, and all images are made up of pixels which contain information about the brightness of red, green, and blue colors. So every picture is just a collection of numbers representing the amount of red, green, and blue in each pixel.

```python
download_url(
    search_images("Cyclist", max_images=1)[0], "cyclist.jpg", show_progress=False
)
Image.open("cyclist.jpg").to_thumb(256, 256)
```

- We are now downloading the test image
- Our model will predict if this image is an image of cyclists
- Creates a 256x256 thumbnail version

## Dataset Creation
```python
searches = ["Cyclists", "Pedestrians"]
path = Path("pedestrians_or_cyclists")
for o in searches:
    dest = path / o
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f"{o} photo"))
    time.sleep(5)
    resize_images(path / o, max_size=400, dest=path / o)
```
This loop:
- Creates directories for each category
- Downloads multiple images for each category
- Adds "photo" to search terms for better results
- Waits 5 seconds between searches to be polite to the search API
- Resizes all images to a maximum size of 400 pixels

## Image Verification
```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
These lines:
- Check all downloaded images for corruption
- Delete any corrupt images
- Count how many images were removed

## DataLoader Creation
```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method="squish")],
).dataloaders(path, bs=10)
```

This creates a FastAI DataBlock with:
- Image inputs and category labels
- 80/20 train/validation split
- Directory names as labels
- Images resized to 192x192
- Batch size of 10 (Number of images shown to check the data)

Let's discuss these one by one:

The first thing that we tell fastai is what kind of input we have. This is defined by `blocks`. Here the input is Image, therefore we set blocks to `ImageBlock`. To find all the inputs to our model, we run `get_image_files` (function which returns all image files in a path).

We used the same function earlier to remove all the corrupted images.

Now it is important to put aside some data to test the accuracy of our model. It is so critical to do so that fastai won't let you train a model without that information. How much data is going to be set aside is determined by `RandomSplitter`. (In our case, we are randomly setting aside 20% of data for our validation set).

Next, we tell fastai the way to know the correct label of the photo. We do this with the `get_y` function. This function tells it to take the label of the photo from its parent directory (the directory in which the photo is stored).

All computer vision architectures need all of the input to be the same size. By using `item_tfms`, we are resizing every image to be (192,192) each, and the method of resize is going to be 'squish'. You can also crop it from the middle.

With the help of `dataloaders`, PyTorch can grab a lot of your data at one time, and this is done quickly using a GPU which can do a thousand things at once. So dataloader will feed the image model with a lot of data that is provided to it at once. We call these images a batch.

`bs=10` will show 10 images from the data it is feeding to the algorithm with labels.

Note: You can learn more about these functions by going to the tutorials and documentation of fastai.

## Model Training

Here we will train our model with resnet18. In fastai, a learner is something which combines our neural net and the data we train it with.
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
These lines:
- Create a vision model using ResNet18 architecture
- Fine-tune it for 3 epochs
- Track error rate as the metric

It will usually take about a minute or two if you use only CPU, but it can take about 10 seconds if you train it on GPU. This difference is because someone has already trained it (pretraining) to recognize over 14 Million images over 20,000 different types, something called the ImageNet dataset. So you actually start with a network which can do a lot, and they made all these parameters available to download from the internet.

`learn.fine_tune` takes those pre-trained weights and adjusts them to teach the model the differences between your datasets and what it was originally trained for. This method is called **fine tuning**.

You can get different architectures from https://timm.fast.ai/

## Prediction
```python
is_cyclist, _, probs = learn.predict(PILImage.create('cyclist.jpg'))
print(f"This is a: {is_cyclist}.")
print(f"Probability that it is cyclist: {probs[0]:f}")
```
Finally:
- Loads and predicts on a test image
- Prints the predicted class
- Shows the probability of the prediction

# Deep Learning Is Not Just for Image Classification

In above example we say the uses of deep learning for image classification. Now we will see another beautiful example. This is called Segmentation.

Segmentation where we take photos and we color every pixel to identify different components of the image.

```python
# Import all functions and classes from fastcore and fastai.vision libraries
from fastcore.all import *
from fastai.vision.all import *
```
These lines import the necessary libraries. `fastcore` provides core utilities, while `fastai.vision` contains computer vision-specific functionality.

```python
# Download and extract the CAMVID_TINY dataset to a local path
path = untar_data(URLs.CAMVID_TINY)
```
This downloads a small version of the Cambridge-driving Labeled Video Database (CamVid), which contains road scene images with pixel-level segmentation labels. `untar_data` downloads and extracts the dataset if not already present.

```python
dls = SegmentationDataLoaders.from_label_func(
    path,                   # Path to the dataset
    bs=8,                   # Batch size for training
    fnames = get_image_files(path/"images"),  # Get list of all image files
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',  # Function to find corresponding label file
    codes = np.loadtxt(path/'codes.txt', dtype=str)  # Load class names from codes.txt
)
```
This creates a `DataLoader` specifically for segmentation tasks:
- `bs=8` sets the batch size to 8 images
- `get_image_files()` gets all image files from the images directory
- The `label_func` is a lambda function that maps each image file to its corresponding label file by adding '_P' to the filename
- `codes.txt` contains the names of segmentation classes (like 'road', 'building', etc.)

```python
# Create a U-Net model with ResNet34 backbone
learn = unet_learner(dls, resnet34)
```
This creates a U-Net architecture (common for segmentation tasks) using ResNet34 as the backbone. U-Net is particularly effective for semantic segmentation because it combines detailed spatial information with deep features.

```python
# Fine-tune the model for 8 epochs
learn.fine_tune(8)
```
This trains the model using transfer learning. It first trains the newly added U-Net layers while keeping the pretrained ResNet34 frozen, then fine-tunes the entire network for 8 epochs.

```python
# Display prediction results for up to 6 images
learn.show_results(max_n=6, figsize=(7,8))
```
This displays a grid showing the original images, their true segmentation masks, and the model's predicted segmentation masks for up to 6 images. The `figsize` parameter sets the size of the display.


# The Future of deep learning

If there is something that a human can do pretty quickly even it needs to be an expert then deep learning will be go at it. If it is something that takes a lot of logical thought process in an extended period of time then Deep learning will not be able to do it perfectly.

Remember: The best time to start learning machine learning was yesterday. The second best time is now. So what are you waiting for? Let's teach some computers to see! 🚀

---

_P.S. No neural networks were harmed in the making of this blog post. They were just mildly confused._
