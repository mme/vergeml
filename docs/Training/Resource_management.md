Ressource Management
============

This chapter will show you on one side how to allocate computational ressources of your machine(s) and show you what other tricks you can apply in order to increase training performance. 

CPU vs GPU allocation
-----------

VergeML handles resource allocation so you can  to train on your CPU or GPU. 

### CPU allocation ###

In your [config file](/Projects/Configuration.md) you can define the CPU as your computational device by including this: 

~~~python
device: CPU
~~~

Now all training and inference tasks will be allocated to your CPU.

### GPU allocation ###

The same as above, just include this in your config file for GPU usage: 

~~~python
device: GPU
    GPU = 1
~~~

For GPU you could also include a memory allocation. This can be helpful if your GPU has enough memory space and you want to train simultaneously on the same GPU. Just include this: 

~~~python
device: GPU
    GPU = 1
    memory = 50%
~~~

You could also allocate your through the command line. For training, type: 

    ml train GPU=1 

Next read
============

Next chapter will all be about predictions: [Predictions](/Making_predictionsg/Performance_metrics.md).

Or jump to the [Table of Content](/TOC.md).