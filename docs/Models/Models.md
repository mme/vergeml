Models
============

In VergeML you can find an ever growing list of Machine Learning models and algorithms. For now, we only support image based models but we aim to include soon many more. 

These are the models and algorithms officially supported by VergeML. They are curated and ready to use! 

### Image Classifiers based on Transer Learning ###

In recent years, one of the biggest applications of machine learning has been in image classification — the ability for a computer to intelligently recognize an object. Specify ```imagenet``` as your model, VergeML will by default select ```resnet-50``` as the image classification architecture.

These are the currently supported architectures:

| Model Name | Architecture | Description  | Code Author | 
| -----------|:-------------:| -----:| -----:|
| imagenet | densenet | A deep convolutional network with more densly connected layers. | VergeML|
| imagenet| inception-v2 | Version 2 of the popular inception architecture. | VergeML |
| imagenet| inception-v3 | Latest version of the inception network. | VergeML| 
| imagenet| mobilenet | Efficient Convolutional Neural Networks for mobile vision applications. |VergeML| 
| imagenet| mobilenet-v2 | Improved version of MobileNet for mobile applications. | VergeML |
| imagenet| nasnet | A deep convolutional network designed for large datasets.| VergeML|
| imagenet| resnet-50 | Deep convolutional network with residual learning. | VergeML|
| imagenet| vgg-16 | A 16 layer deep convolutional neural network. | VergeML|
| imagenet| vgg-19 | Similar to VGG16 but with additional 3 convolutional layers. | VergeML|
| imagenet| xception | Network with depthwise separable convolution operation. | VergeML|

### Image Similiarity Search ###

Image similarity search is the ability to find within a database similar looking images based on based on principal component analysis (such as shape, color, textures, etc).  Specify ```similarity``` as your model, VergeML will by default select ```resnet-50``` as the archtitecture for your image similarity search. 

The following architectures can be used for image similarity search:

| Model Name | Architecture | Description  | Code Author | 
| -----------|:-------------:| -----:| -----:|
| similar | densenet | A deep convolutional network with more densly connected layers. | VergeML|
| similar | inception-v2 | Version 2 of the popular inception architecture. | VergeML |
| similar | inception-v3 | Latest version of the inception network. | VergeML| 
| similar | mobilenet | Efficient Convolutional Neural Networks for mobile vision applications. |VergeML| 
| similar | mobilenet-v2 | Improved version of MobileNet for mobile applications. | VergeML |
| similar | nasnet | A deep convolutional network designed for large datasets.| VergeML|
| similar | resnet-50 | Deep convolutional network with residual learning. | VergeML|
| similar | vgg-16 | A 16 layer deep convolutional neural network. | VergeML|
| similar | vgg-19 | Similar to VGG16 but with additional 3 convolutional layers. | VergeML|
| similar | xception | Network with depthwise separable convolution operation. | VergeML|


Porting a model
============

If you want to contribute by porting new models into VergeML, you can do this by using the plug-in system in VergeML. We will provide soon a detailed guide on how to do this.



