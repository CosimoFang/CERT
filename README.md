# BERT on MOCO


## Example usage

#### Data Preparation

You need to augment your data in two different ways and save them in the *'augment.csv' in the same form.

##### model output

Before training, you need to build the moco_model with *mkdir moco_model*

##### train
You need to change the number of negtive samples in MOCO.py *line 85* , you can also change the epoch: *line *, batch size:*line 43*, learning rate:*line 48*.

You can train on the MOCO task with:
*python MOCO.py*

##### transform model

After training, you can extract encoder_k from the whole model with
*python trans.py*
