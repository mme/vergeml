
What is the configuration file
============

The configuration file will be the starting point when you create a new project. It is meant to store all default values for your model and help you automate your workflow in VergeML. 

You can adjust every command and subcommand including model specific functions. 

> Info: In addition, it is a good way to share your configuration with others, so they have a good method to exactly reproduce your work. (Spoiler alert! Here is our best performing [config file](/Installation/Installation.md) for the skin cancer detector)

VergeML general settings
------------
These are the top level settings, which are always available:

* model: contains all model relevant defaults
* data: all data relevant functions, parameters and preprocessing steps.
* samples-dir: path where your data samples are stored.
* trainings-dir: path where your trainings are stored.
* cache-dir: path where your caching file is stored.
* device: computational resources allocation for training and inference.
* test-split: to specify your test directory or test split.
* val-split: to specify your vaildation directory or validation split.
* random-seed: your random seed for data preprocessing and data loading.

Command specific settings
------------
Using different command and subcommand specific settings allows you to include your preferences in your working processes. These are the top level commands currently available:

* list: specify how your performance list is shown.
* run: specify preferences for run commands.

Model specific settings
------------
Model specific settings will appear after you have specified a model (e.g. "Imagenet") in your project file. These settings will only affect commands and subcommands from the model. As soon as you change model, the settings might become redundant. 

These are the settings available for the model "imagenet":
* train: all default hyperparameters for training will go in here.
* predict: all default hyperparameters for inference.

> Info: You also can set or change parameters in the command line for singular use (not as default values). 

How to use the config file
============

So lets mingle with your config file! Down below we will show you examples on how to use it. Lets start with VergeML general settings:

VergeML general settings
------------

model:
------------

The model is the heart of your project as it defines what task you want to the AI to acomplish. For example, to classify images into lables using using transfer learning you would specify "imagenet" as your model. 

When you start a new project, include the model name in your command: 

    ml --model=imagenet new my_project

You can always switch to a new model in a project by changing the model parameter in your configuration file: 

~~~python
model: imagenet
#Defines the ML model to use in the current project.
~~~

data:
------------

In Machine Learning your data becomes the code. The normal way to create an AI is by driving a set of samples through a ML model. So, the more data you have, the better you can train your AI. 

VergeML integrates a complete pipeline of operations that can enhance your dataset. This is called the preprocessing pipeline and is most easily defined in your configuration file. So lets have a look how VergeML handles this! 

> Info: You can leave this whole section blank, VergeML will handle the data pipeline automatically. It will load the samples images directly into your model.

### Preprocessing operations ###

Preprocessing operations define the steps you take before passing your samples data to the model. There are some preprocessing operations that VergeML handles automatically, for example setting the correct size of images for a model. Other preprocessing steps can be included manually in your configuration file. 

> Info: Preprocessing operations are singular steps where your data is processed in a given way. Augmentations multiply your samples, whereas transformations change the data in its core. 

VergeML currently supports an array of image based preprocessing steps. 

Our skin cancer classifier has around 1.100 images of melanomas and 6600 images of normal moles. Here, the data needs to be balanced, so lets augment the melanoma dataset: 

~~~python
data:
  preprocessing:
    - op: augment
          variants 5
          #Multiplies your sample data five times.   
~~~

We now have nearly data equilibrium, but the new data generated are just copies. Let´s transform them so

The operations defined will be processed chronologically, starting with the first and moving downward. This is the only case where chronological order matters within your configuration file.

> Info: To see all available image preprocessing steps currently available, see [Image transformations](/Data/Image_transformation.md).

### Input ###

Normally you don´t need change anything in this parameter, as the model automatically handles the correct input format. But if you want specify a samples format, you can do it here. 

or example, you could change the input format to be MNIST:

~~~python
data:
  input: 
    type: mnist 
# Changing the input format to MNIST 
~~~
> Info: MNIST is a specialized data format often used in ML research. It contains images of handwritten numbers.

### Output ###

The output defines how the model needs your data to look like. VergeML handles this automatically and it is a very advanced option.

### Cache ###

Your data is automatically cached before starting your first training. This ensures the best speeds during training and opitmal usage of your hardware. 

> Info: The default chaching is via creating a chache file and stored in your .cache directory. 

You can opt to change how VergeML caches your samples. To read more on how to do this, see the more advances section on [Caching](/Data/Caching). 

device:
------------

Here you can define on what devices and and how many resources you want to allocate for training and inference. 

~~~python
device: GPU
  GPU: 1
  memory: 20%
#You can define which GPU and how much memroy allocation you want to use.
~~~

In the below example, the second GPU was chosen (GPU=0 would be the first) with a maximum memory allocation of 20%.

If you want to run just on your CPU, type:

~~~python
device: CPU
~~~

test-split:
------------
You might want to allocate specific test samples for your final performance testing. This might be usefull, if the test dataset comes closer to the real data when operating your AI. 

So here you can define your test directory or in what split your test dataset is generated from your samples.

~~~python
test-split: 10% 
# or test-split: /path/to/your/test/dir
# specify your test directory and your test split based on a percentage or an integer.
~~~

val-split:
------------
This is similar to the test split but for your validation dataset. 

~~~python
val-split: 10% 
# or val-split: /path/to/your/val/dir
# specify your validation directory and your validation split based on a percentage or an integer.
~~~

cache-dir:
------------
Specifying a different cache directory might help you with optimizing your disk space. If you leave this section blank, VergeML will automatically allocate the cache directory in your project directory.

~~~python
cache-dir: /your/new/caching/directory
#Path to the new caching directory. 
~~~

random seed
------------
Machine Learning takes many input fields randomly, from data loading to preprocessing steps. By defining a specific random seed, you always will get the same results. By default, VergeML has a random seed of ```1234```. 

~~~python
random-seed: 1234
#A number or string that defines your random seed.
~~~

> Info: Although this is true for all processes concerning data, the training process of a model can not be defined by the random seed. Even if you feed the model with the exact same data, the model still can show slightly different results. 

Next read
============

If you want to continue reading, this might be your next chapter: [Training overview](/Training/Overview.md)

or jump the [Table of Content](/TOC.md)