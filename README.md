![VergeML](Images/Header.png "VergeML")


VergeML is a **command line based environment** for exploring, training and running state-of-the-art Machine Learning models. It provides ***ready-to-use models***, handles ***data preprocessing and augmentation***, tracks your AI's ***training sessions*** and provides other goodies such as an ***automatic REST interface***.

Here's how it looks in action:

<img src="Images\term.png" alt="terminal" width="691px"/>


Installation
============

VergeML runs on Windows, Linux and MacOS. You need to have Python 3.6 and [TensorFlow installed](Installation/Installation.md). 

Get VergeML via pip: 

    pip install vergeml

Verify your installation by typing:

    ml help

Congratulations, you have successfully installed VergeML! If you need further help, see the full [installation guide](Installation/Installation.md).

Quick Start
===========

Let´s say we want to build a skin cancer detector using an image classifier. Using VergeML it will take us 5 commands and around 15 minutes (incl. downlad and trainin time)! During this quick start tutorial we will show you the basics, where we will get our skin detector at around **XY% accuracy (F1: XY)**. This is not bad, but as we are eager to discover the upper limits, we will continue to push this further to up to **XY% accuracy (F1: XY%)** in the subsequent chapters.

First, we create a new project for our skin cancer detector. Projects help you organize your data, save your training results and compare the performance between trained AIs. 

Go to the directory where you want to create your project and type: 

    ml --model=imagenet new cancer_detector

This sets up a model called ```imagenet```, which is based on transfer learning. 

Let's change to this project directory and have a look: 

    cd cancer_detector

VergeML will automatically create a samples folder and a configuration file (vergeml.yaml). Among other things, this configuration file defines the current model.
 
Let's get some help on what we can do with the current model:

    ml help

In the output you will see a section on model functions. It says we have two model functions, train and predict. Let's try training first!

Start training!
-----------

To start training an AI we will need a dataset. We prepared a dataset with skin lesions for our skin detector (based on the[ham10000 dataset](Installation/Installation.md)).

    ml download:ham400

> Info: VergeML provides several datasets to get you started. To see a list type ```ml help download```

After the download has finished, you will see a lot of images in your ```samples``` directory divided into two folders: cats and dogs. 

Later, when you use your own data, simply copy your images into subdirectories of the samples directory. VergeML will automatically pick up the directory names as labels. 

To start training, type:

    ml train

As a first step, VergeML will feed each of our images into a pretrained neural network, extract their features as output and cache it on disk. (On a GPU, this will typically take around 15 minutes.) Then it will train a new neural network based on this output. As a last step it will combine these two networks into a single network tailored for our task of classifying cats and dogs. This process is called "transfer learning".

VergeML will print out the test accuracy after our training is finished to evaluate the model's final performance. Our cats-and-dogs classifier achieves 98.6%, which is pretty good.

> Info: By default, VergeML reserves 10% of your samples as validation and 10% as testing data. This step is required to measure the accuracy of your model. 

We can inspect our model's performance using the list command:

    ml list

This will give you the name (prefixed by the @ sign) and several performance metrics.

For instance, the training accuracy (```acc```) will tell you how good your AI can classify the images it sees during training, while validation accuracy (```val_acc```) tells you how well it performs with unseen images.

Using the AI from the command line
-----------

Our cats-and-dogs classifier is now ready to use. Let's point it to an image of a mole and see what it predicts: 

    ml @name-of-your-AI predict <filename>

> Info: You can even point it to a directory: ```ml @name-of-your-AI predict my_cats_and_dogs_pictures/*```

Launching a REST service
-----------
Finally, let's deploy our newly trained AI on a web service:

    ml @name-of-your-AI run:rest 

VergeML provides an API explorer that will launch in a new browser window. (If you don't want the browser to open use the ```--no-browser``` option.)

For example, to use the REST interface with cURL: 

    curl -F 'files=@path/to/image' http://localhost:2204/predict 

Digging deeper into VergeML
============

1. [Choosing a model for your task](/Models/Models.md)
2. [Creating a new project](/Projects/Projects.md)
5. [Understanding general training procedures](/Training/General_training.md)
    * [Automating your training configurations](/Projects/Configuration.md)
    * [Reading and understanding your performance](/Training/Performance_metrics.md)
    * [Allocating available ressources for training](/Training/Resource_management.md)
5. [Making predictions with your AI](/Trainging/Get_Started.md)

    * [Tensorboard](/Tools/Tensorboard.md)
    * [REST](/Tools/Jupyter.md)



Motivation
============

Why we built VergeML...

License
============
[MIT](/LICENSE.md) 

Copyright (c) 2018-present, Markus Ecker & Camillo Pachmann 