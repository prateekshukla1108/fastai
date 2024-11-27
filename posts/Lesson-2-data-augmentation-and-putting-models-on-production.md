---
aliases:
- Lesson-2-data-augmentation-and-putting-models-on-production
date: '2024-11-27'
image: images/lesson1.png
layout: post
description: "Training a model with data augmentation and huggingface spaces."
title: "Lesson 2: Data Augmentation and Putting the model on production"
categories:
- fastAI

---

This lesson is going to be about making a bear classifier and putting it into production. The project that we will be using will be a bear classifier and it will discriminate between black bears, Grizzly Bears, Teddy Bears. Also at last we will be also discussing on how to make things faster when using jupyter

## 1. Setting Up Our Tools

First, we need to import some special tools (we call them libraries) that will help us build our bear classifier:

```python
from duckduckgo_search import DDGS
from fastcore.all import *
import time
from fastai.vision.widgets import *
from fastai.vision.all import *
from fastdownload import download_url
```

What do these tools do?
- `duckduckgo_search`: Helps us find bear pictures on the internet
- `fastai`: A powerful library that makes AI easy to use
- `time`: Helps us add delays when downloading images
- `fastdownload`: Makes downloading images easy

## 2. Getting Our Bear Images

Now, let's create a function to search for bear images:

```python
def search_images(keywords, max_images=400):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot("image")
```

This function is like having a personal assistant who can search the internet for bear pictures! We tell it what kind of bear we want, and how many pictures we need.

Let's test it by downloading one image of each bear type:

```python
# Download a grizzly bear image
urls = search_images("grizzly bear", max_images=1)
download_url(urls[0], "grizzly.jpg", show_progress=False)

# Download a teddy bear image
download_url(search_images("Teddy Bear photos", max_images=1)[0], "teddy.jpg", show_progress=False)

# Download a black bear image
download_url(search_images("Black Bears", max_images=1)[0], "Black.jpg", show_progress=False)
```

## 3. Organizing Our Data

Now we need to create folders for our different types of bears:

```python
bear_types = 'grizzly', 'black', 'teddy'
path = Path('bears')
searches = ["Grizzly Bear", "Black Bear", "Teddy bear"]

# Create folders and download multiple images
for o in searches:
    dest = path / o
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f"{o} photo"))
    time.sleep(5)  # Wait 5 seconds between searches
    resize_images(path / o, max_size=400, dest=path / o)
```

This code:
1. Creates a folder called "bears"
2. Inside it, creates three more folders: one for each bear type
3. Downloads multiple images for each bear type
4. Resizes them to make sure they're not too big

## 4. Preparing Images for Training

Before we can train our AI, we need to check our images and prepare them:

```python
# Find all our image files
fns = get_image_files(path)

# Check for any broken images and remove them
failed = verify_images(fns)
failed.map(Path.unlink)
```

Now, we'll create a DataBlock to handle images

```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # We're working with images and categories
    get_items=get_image_files,          # How to find our images
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # Split data into training and testing
    get_y=parent_label,                 # Use folder names as labels
    item_tfms=Resize(128))              # Resize all images to the same size
```

## 5. Training Our AI Model

Now we will be training our model with Resnet18

```python
dls = bears.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)  
```
We can make this faster and more efficient by resizing or cropping the images. We can do this by 3 ways
### Resizing

By default `Resize` _crops_ the images to fit a square shape of the size requested, using the full width or height. This can result in losing some important details. 

```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish)) dls = bears.dataloaders(path) dls.valid.show_batch(max_n=4, nrows=1)
```

### Padding

Alternatively, you can ask fastai to pad the images with zeros (black). This fills the empty spaces with black.

```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros')) dls = bears.dataloaders(path) dls.valid.show_batch(max_n=4, nrows=1)
```

All of these approaches seem somewhat wasteful, or problematic. If we squish or stretch the images they end up as unrealistic shapes, leading to a model that learns that things look different to how they actually are, which we would expect to result in lower accuracy. If we crop the images then we remove some of the features that allow us to perform recognition. For instance, if we were trying to recognize breeds of dog or cat, we might end up cropping out a key part of the body or the face necessary to distinguish between similar breeds. If we pad the images then we have a whole lot of empty space, which is just wasted computation for our model and results in a lower effective resolution for the part of the image we actually use.
### RandomResizedCrop

Instead, what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch (which is one complete pass through all of our images in the dataset) we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images. It also reflects how images work in the real world: different photos of the same thing may be framed in slightly different ways.

In fact, an entirely untrained neural network knows nothing whatsoever about how images behave. It doesn't even recognize that when an object is rotated by one degree, it still is a picture of the same thing! So actually training the neural network with examples of images where the objects are in slightly different places and slightly different sizes helps it to understand the basic concept of what an object is, and how it can be represented in an image.

Here is the exmaple - 

```python

bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2)) dls = bears.dataloaders(path) dls.train.show_batch(max_n=8, nrows=2, unique=True)

```

The `batch_tfms=aug_transforms(mult=2)` applies data augmentation techniques, effectively doubling the number of training samples by creating variations of the original images through transformations like rotation, flipping, or color adjustments.

## 6. Checking How Well It Works

Finally, let's see how well our AI does:

```python
interp = ClassificationInterpretation.from_learner(learn)
```


This creates an interpretation object from your trained model (the `learn` object). It's a crucial tool for analyzing how your model performs, containing information about predictions, actual labels, losses, and other metrics.

```python
interp.plot_confusion_matrix()
```

This creates a confusion matrix visualization - a table showing:

- Rows: Actual classes
- Columns: Predicted classes
- Each cell: Number of predictions This helps you see where your model is getting confused. For example, if you're classifying cats and dogs, it shows how many cats were correctly identified as cats, how many cats were mistakenly labeled as dogs, etc.


```python
 interp.plot_top_losses(5, nrows=1)
```


This shows the 5 images where your model performed worst (had highest loss):

- Each image is displayed with:
    - The actual label
    - The predicted label
    - The loss value
    - The probability the model assigned to its prediction This is invaluable for understanding what types of images your model struggles with


## Extra: Cleaning Up Our Dataset

If we want to improve our model, we can use this tool to review and clean our dataset:

```python
cleaner = ImageClassifierCleaner(learn)
```

This creates an interactive tool (works in Jupyter notebooks) that lets you:

- Browse through your dataset
- Flag incorrect labels
- Delete problematic images
- Remove duplicates It's especially useful when you discover through the interpretation tools that your dataset has issues like mislabeled images or poor quality samples.

You can find which images are bad and you can keep it or remove it or put it in the correct catagory. 

# Huggingface and Gradio

After training the AI model you might want to deploy it somewhere and that is where huggingface and gradio comes into picture. 

## What is Gradio?

Gradio is a Python library that allows you to quickly create customizable user interfaces for machine learning models. With just a few lines of code, you can turn your model into an interactive web application, making it easy for others to test and use your model without needing any programming skills.

## What is Hugging Face?

Hugging Face is a popular platform that provides state-of-the-art pre-trained models for natural language processing (NLP), computer vision, and more. Their library, `transformers`, offers a wide range of models that you can use directly in your applications.

## Starting Up

To start firstly you need to take your model. You can do this by using - 

```python
learn = load_learner('export.pkl')
```

This will export your model into a file named export.pkl 

Then you would need to install gradio and transformers. You can do it by using pip or conda

```bash
conda install `gradio transformers`
```

Creating a blog post for beginners about using Gradio with Hugging Face can be an exciting way to introduce them to the world of machine learning and user interfaces. Below is a simplified and engaging blog post inspired by the provided content.


## Setting up huggingface

Go to huggingface.co/spaces and create a new space. You will need to sign up to make a new space. After signing up, you will be prompted with this kind of interface:

Fill in the relevant details and create your Hugging Face space. After you create it, you will be greeted by a beautiful page like this:


You need to generate a token, clone the repository on your machine, edit a file, and push it back to the Hugging Face platform.

If you feel intimidated by all this jargon, consider learning how to use Git or GitHub through resources available online.

### Step 1: Generate Your Access Token

Before cloning your space, you need to generate an access token. This token allows you to authenticate and push changes to your repository. To generate a token:

1. Navigate to your account settings on Hugging Face.
2. Click on "Access Tokens" in the sidebar.
3. Create a new token with write access.

### Step 2: Clone Your Repository

Once you have your access token, open your terminal and navigate to the directory where you want to clone your space. Use the following command:

```bash
git clone https://<your-username>:<your-access-token>@huggingface.co/spaces/<your-space-name>
```

Replace `<your-username>`, `<your-access-token>`, and `<your-space-name>` with your actual Hugging Face username, the generated token, and the name of your space respectively. This command will create a local copy of your space where you can make changes.

### Step 3: Edit Your Files

Navigate into the cloned directory:

```bash
cd <your-space-name>
```

Open the folder in your preferred code editor (like Visual Studio Code). You will typically find a few default files such as `.gitattributes` and `README.md`. 

To create functionality for your space, you’ll need to add an `app.py` file. This is where you'll write the code for your application.

### Step 4: Create Your Application

In `app.py`, you can start coding your application. Here is the default text which comes with gradio

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

This code snippet sets up a basic Gradio interface that allows users to upload an image and receive a classification result.

### Step 5: Prepare for Deployment

Before pushing your changes back to Hugging Face, create a `.gitignore` file in your project directory. This file helps prevent unnecessary files from being uploaded. Here’s an example of what to include in `.gitignore`:

```
__pycache__/
*.pyc
venv/
```

### Step 6: Push Your Changes

After editing and saving your files, it's time to push them back to Hugging Face. First, check what changes are staged for commit:

```bash
git status
```

If everything looks good, add your changes:

```bash
git add .
```

Then commit them with a message:

```bash
git commit -m "Initial commit of my Hugging Face Space"
```

Finally, push the changes:

```bash
git push origin main
```

- **Note: If you are afraid of cli then you can also use github desktop which is beautiful GUI version to manage git repositories**

> Note: You can also create your app.py file from your browser


### Step 7: View Your Deployed Space

After pushing successfully, navigate back to your Hugging Face Space URL (e.g., `https://huggingface.co/spaces/<your-username>/<your-space-name>`). You should see your application live! It may take a moment for the deployment process to complete.

Congratulations! You have successfully set up and deployed your first Hugging Face Space. Now you can share it with friends or colleagues and explore adding more features or models.


# Setting up environment for Python

Here we will not be discussing about setting up our python environment for development. To learn about that you need to see the guide here - https://prateekshukla1108.github.io/fastai/posts/Installing-Dependencies.html
