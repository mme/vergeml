Using Tensorboard for a clearer view
============

Tensorboard is a visualization tool created by Google for Tensorflow. It can help you understand and debug your trained AIs. To use tensorboard, simply type:

    ml run:tensorboard

Your internet browser should now open and direct you automatically to tensorboard showing you the training performance of all your AIs. This might look like this: 

<img src="Images\Tensorboard.PNG" alt="tensorboard"/>

VergeML will include all trained AIs into the tensorboard command. If you want to include only specific AIs, adapt the command to look like this: 

ml @touchy-automaton run:tensorboard

> Info: You can compare multiple AIs by just adding more before run:tensorboard. 