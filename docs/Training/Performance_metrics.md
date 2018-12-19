How to read performance metrics
============

During training your model will generate values that indicate its performance. A very common performance metric is accuracy (sometimes split into training, validation and test accuracy). Accuracy measures how well your AI does at a given task. So, the higher, the better (within a range of 0 - 1).

Another metric often used is the loss value. This one is more dificult to read, as it states the distance between prediction and truth. Or in other simple words, how far the AI is on being perfect. Here, you want to achieve a downward trend torwards minimazing the loss value.

How does VergeML handle performance metrics?
============

During training your model will generate model specific performance metrics. To keep it simple, lets assume it just creates accuracy and loss values. For every sample the model learns, it will generate training values for accuracy and loss.

After a training has finished, you can plot different charts to your trained AI by: 

    ml @ai-name plot:chart-name

We have included several plot functions within VergeML, such as:

* Reciever Operating Characteristics Curve (ROC): ```plot:roc```
* Confusion Matrix: ```plot:confusion-matrix```
* Precision-Recall: ```plot:confusion-matrix --class=label1```

### What is overfitting? ###

During training you will notice how the stats change. Overfitting happens, when the training accuracy is very high (approaching 1.0) but the validation accuracy keeps low. This happens because the AI "remembers" the training dataset and does not extract the relevant features. 

To overcome this you will need to enlarge your training dataset either by adding new data or by augmenting your current dataset. See chapter [image transformations](/TOC.md) to learn how to do this. In addition you could include a ```dropout``` value to reduce the AI's overall performance. 

### What is underfitting? ###

Underfitting occurs when your model cannot capture the underlying trend of the data. Specifically, underfitting occurs if the model shows low variance but high bias.  It is often a result of an excessively simple model. 

You can detect signs of underfitting when the training accuracy stays low and the loss function high. In addition, there will also be no clear trend torwads increasing accuracy nor decreasing loss value.

To solve this problem you need to either increase the capacity of your model (e.g. adding layers) or decrease the complexity of your data distribution.

Saved performance metrics
============

VergeML automatically saves performance metrics in the ```trainings/stats``` directory. Here, you will find the follwing files:

* events file: this is a file generated from ```tensorflow``` which saves during each training steps performance metrics. You can read this file using ```tensorboard```. 
* stats.csv file: this saves the stats in a .csv file.
* predictions.csv file: this saves all prediction values for your test sample split.

Next read
============

If you want to continue reading, this might be your next chapter: [Resource management](/Training/Resource_management.md).

Or jump to the [Table of Content](/TOC.md).
