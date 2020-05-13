<<<<<<< HEAD
# BERT on MOCO


## Example usage

#### Data Preparation

You need to augment your data in two different ways and save them in the *'augment.csv' in the same form.

#### Model Output

Before training, you need to build the moco_model with *mkdir moco_model*

#### Train
You need to change the number of negtive samples in MOCO.py *line 84* , you can also change the epoch: *line 41*, batch size:*line 45*, learning rate:*line 50*, and temperature: *line 90*

You can train on the MOCO task with:

*python MOCO.py*

#### Transform Model

After training, you can extract encoder_k from the whole model with

*python trans.py*
=======
