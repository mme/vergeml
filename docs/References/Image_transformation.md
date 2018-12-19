Image transformations
============

Image transformations may help you boost your training and test datasets.
The following operations are currently available and can be included in your project file below "preprocessing:". 

Any number of transformations can be pieced together to form a preprocessing pipeline. The order is relevant, as the first entry will be the first operation.

**Crop:**

Crops a specific part of an image of a definable size:
~~~yaml
data:
  preprocessing:
  - op: crop
    width: 500
    height: 500
    
    # optional x-coordinate of the rectangle
    # x: 100

    # optional y-coordinate of the rectangle
    # y: 100
   
    # optional, sets position of the rectangle (top-left, top-right, bottom-left, bottom-right, center)
    # position: center
    
~~~

**Random Crop:**

Randomly crops a part of an image of a definable size:
~~~yaml
data:
  preprocessing:
  - op: random-crop
    width: 500
    height: 500
~~~

**Resize:**

Sets a new size of your images.
~~~yaml
data:
  preprocessing:
  - op: resize
    width: 500
    height: 500
   
    # set number of channels (0,3) 
    # channels: 3
~~~

**Flip Horizontally:**

Flips the images horizontally.
~~~yaml
data:
  preprocessing:
  - op: flip-horizontal
      
    # optional, set a chance (>=0, <=1, default=1.) 
    # chance: 0.25
~~~

**Flip Vertically:**

Flips the images vertically.
~~~yaml
data:
  preprocessing:
  - op: flip-vertical
      
    # optional, set a chance (>=0, <=1, default=1.) 
    # chance: 0.25
~~~

**Grayscale:**

Converts image to grayscale mode (only one channel).
~~~yaml
data:
  preprocessing:
  - op: grayscale
~~~

**RGB:**

Converts image to RGB mode (with three color channels).
~~~yaml
data:
  preprocessing:
  - op: rgb
~~~