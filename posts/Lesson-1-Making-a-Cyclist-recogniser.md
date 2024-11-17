---
aliases:
- Lesson-1-Making-a-Cyclist-recogniser
date: '2024-11-15'
image: images/lesson1.png
layout: post
description: "Documentation for lesson 1 of fastAI practical deep learning for coders."
title: "Lesson 1: Vision models and Fundamentals"
categories:
- fastAI

---

The evolution of Machine Learning has transformed from a concept once deemed nearly impossible to a technology that is now easily accessible and widely utilized. It was considered so ridiculous in the early days that people joked about it. Here is one example:

![XKCD comic from 2015](/posts/images/xkcd.png)

Above is an xkcd comic which shows how people joked about it. The good news is that we are going to make a computer vision model in this lesson today. So get excited!


# The Evolution of Neural networks
## Before Neural Networks

In the era before neural networks, people used a lot of workforce to identify images, then many mathematicians and computer scientists to process those images and create separate features for each one of them. After a lot of time and processing, they would fit it into a machine learning model. It became successful, but the problem was that making these models took a lot of time and energy, which was inefficient and tedious.

## The first neural network

Back in 1957 a neural network was described as something like a program. So in a traditional program we have some inputs and then we put them in program which have functions, conditionals, loops etc and these give us the result.

## Deep learning models

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


```bash
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

- Create a vision model using ResNet18 architecture. It's like creating a baby AI, except it won't keep you up at night (your debugging sessions will do that instead).
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


## Tabular Analysis


We can also help analyze structured data like spreadsheets. Let's start by breaking down each line of code and understanding what it does:

```python
from fastai.tabular.all import *
```

This line imports all the necessary functions and classes from fastai's tabular module.

```python
path = untar_data(URLs.ADULT_SAMPLE)
```

Here, we're downloading and extracting a sample dataset called "Adult" that predicts whether someone makes over $50K per year. The `untar_data` function:

- Downloads the dataset if it's not already present
- Extracts it from its compressed format
- Returns the path where the data is stored

```python
dls = TabularDataLoaders.from_csv(
    path/'adult.csv',
    path=path,
    y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status',
                 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize]
)
```

This is where the magic begins! Let's break this down:

- We're creating a DataLoader object that handles how we feed data to our model
- `path/'adult.csv'`: Specifies the CSV file containing our data
- `y_names="salary"`: Indicates that "salary" is our target variable (what we're trying to predict)
- `cat_names`: Lists our categorical columns (text or discrete data) like workclass, education etc
- `cont_names`: Lists our continuous numerical columns like age, final weight(a census data concept), education num(numerical encoding of education level). These are the columns which can take any real number
- `procs`: Specifies the preprocessing steps:
  - Categorify: Converts categorical variables into numbers
  - FillMissing: Handles any missing values
  - Normalize: Scales numerical data to a similar range

```python
learn = tabular_learner(dls, metrics=accuracy)
```

This line creates our machine learning model:

- `tabular_learner`: Creates a neural network designed for tabular data
- `metrics=accuracy`: Tells the model to track prediction accuracy during training

```python
dls.show_batch()
```

This displays a sample of our processed data, helping us verify that everything looks correct before training.
The beautiful thing about this function is that it uses type dispatch which is particularly used in a language called julia and it allows us to define functions that can adapt their behavior according to input types. Basically it will provide realistic data for age, fnlwgt etc

```python
learn.fit_one_cycle(2)
```

Finally, we train our model:

- We don't say fine tune model because for tables because every table of data is very different. So we just 'fit' the data.
- The number 2 indicates we'll train for 2 epochs (full passes through the data)
- "one_cycle" refers to the One Cycle Policy, a training technique that helps achieve better results faster


### Why Tabular Deep Learning?

You might wonder why we'd use deep learning for tabular data when we have traditional methods like Random Forests or XGBoost. Here's why:

It can help in financial predictions, risk assesment, sales forcasting etc

## Collaborative Filtering

Collaborative filtering is the basis of most recommandation systems today. From your youtube recommandation to the recommandations you see in spotify.

It works by finding which users liked which products and then it use that to guess what other product smilar user liked and based on those smilar users what users might like.

Note that here Similar Users don't mean similar demographically but similar in sense that those people liked the same kind of product that you liked.


### 1. Import the necessary modules:


```python
from fastai.collab import *
```

- This imports everything from the `fastai.collab` module, which provides tools for collaborative filtering and recommendation systems.

### 2. Download and extract sample data:


```python
path = untar_data(URLs.ML_SAMPLE)
```

- Again we are downloading and extracting data here
- `URLs.ML_SAMPLE` points to a small sample dataset from the MovieLens dataset, commonly used in recommendation system experiments.

### 3. Create data loaders for collaborative filtering:


```python
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
```

- `CollabDataLoaders.from_csv` reads a CSV file containing user-item interaction data (e.g., user ratings for movies) to create a data loader.
- `path/'ratings.csv'` specifies the path to the CSV file that contains the ratings dataset.
- The `dls` object now holds the data loaders, which manage the data used for training and validation during the model's training process.


### 4. Display a sample batch of data:


```python
dls.show_batch()
```

- This displays a sample batch of user-item-rating triplets (e.g., a user ID, a movie ID, and the user's rating for the movie).

### 5. Create a collaborative filtering model:


```python
learn = collab_learner(dls, y_range=(0.5,5.5))
```

- `collab_learner` initializes a collaborative filtering model based on a neural network architecture.
- The `dls` object is passed to define the input data.
- `y_range=(0.5, 5.5)` specifies the range of the target values (ratings), helping the model output predictions in the expected range.



### 6. Train the model with fine-tuning:


```python
learn.fine_tune(10)
```

- `fine_tune` trains the model for a specified number of epochs—in this case, 10 epochs.
- The model first uses a pre-trained embedding (if available) and then fine-tunes it on the given dataset.

### 7. Display the results:


```python
learn.show_results()
```

- This method shows the predictions made by the model alongside the actual ratings from the test dataset.
- It provides an intuitive way to evaluate the model's performance by comparing predicted and actual ratings.


# The Future of deep learning

We are still scratching the tip of the iceberg in the field of AI, particularly deep learning, despite its widespread adoption and heavy marketing. The advancements we've witnessed so far, though impressive, represent only the beginning of what’s possible. The landscape of AI is rich with pre-trained models, open-source frameworks, and affordable access to GPUs, making it easier than ever for individuals and organizations to learn, innovate, and expand within this domain.

Deep learning thrives in areas where tasks require processing vast amounts of data to uncover patterns that humans often miss. If a task can be completed by a human in a short period, even if it requires significant expertise, deep learning models have shown remarkable capabilities to replicate and sometimes surpass human performance. Examples abound, from image and speech recognition to real-time language translation and game-playing strategies.

However, the limitations of deep learning become evident in areas demanding prolonged logical reasoning, complex problem-solving, or tasks rooted in abstract thought. These tasks require an understanding of nuanced context, long-term planning, or forming new concepts beyond the data it has been trained on—areas where humans still hold a decisive advantage.

Deep learning, at its core, is an excellent pattern recognizer but not an innate reasoner. While models like GPT and DALL-E can generate creative outputs, their reasoning capabilities are constrained by the structure and scope of their training data. They excel at interpolation within known data distributions but falter when extrapolating to scenarios far removed from their training domain.

The future of deep learning lies in addressing these limitations. Innovations such as integrating symbolic reasoning with neural networks, neuromorphic computing, and leveraging advances in unsupervised learning hold promise for overcoming the current bottlenecks. Moreover, as the field pushes towards AGI (Artificial General Intelligence), the convergence of deep learning with other disciplines like neuroscience, quantum computing, and systems biology may pave the way for breakthroughs that are currently unimaginable.
Remember: The best time to start learning machine learning was yesterday. The second best time is now. So what are you waiting for? Let's teach some computers to see! 🚀

---

_P.S. No neural networks were harmed in the making of this blog post. They were just mildly confused._

Now go forth and may your loss curves be ever decreasing! 📉✨

P.S. If your model starts predicting lottery numbers, please share them. For scientific purposes, of course.
