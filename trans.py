import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
import argparse
import csv
from transformers import BertTokenizer,BertForSequenceClassification
path_to_biobert = '/home/fanghongchao/biobert/pretrained'
usemoco=True
if usemoco:
    model = BertForSequenceClassification.from_pretrained(
        path_to_biobert,  # Use the 12-layer BERT model, with an unc
        num_labels=1000,  # The number of output labels--2 for binary classif# You can increase this for mult
        output_attentions=False,  # Whether the model returns attention
	output_hidden_states=False,  # Whether the model returns all hidden-states
	)
    checkpoint = torch.load('/home/fanghongchao/biobert/moco_model/moco.tar')
    print(checkpoint.keys())
    print(checkpoint['arch'])
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        if 'module.encoder_q' in key:
            new_key = key[17:]
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
    for key in list(state_dict.keys()):
        if key == 'classifier.0.weight':
            new_key = 'classifier.weight'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.0.bias':
            new_key = 'classifier.bias'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.2.weight' or key == 'classifier.2.bias':
            del state_dict[key]
    state_dict['classifier.weight'] = state_dict['classifier.weight'][:1000, :]
    state_dict['classifier.bias'] = state_dict['classifier.bias'][:1000]
    model.load_state_dict(checkpoint['state_dict'])						
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, 2)
    torch.save(model.state_dict(), "./moco_model/moco.p")
    print('finished')
