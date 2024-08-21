import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
torch.manual_seed(777)

def make_path(local_dir_path, wav_path):
    List = []
    for i in range(0,len(wav_path['Path']),1):
        List.append(local_dir_path+wav_path['Path'].values[i]+'.flac')
    List = np.array(List)
    return List

def make_label(Label_pdf):
    label=[]
    for i in range(0,len(Label_pdf['Path']),1):
        if  Label_pdf["Label"][i] == 'bonafide' :
            label.append(0)
        elif  Label_pdf["Label"][i] == 'spoof' :
            label.append(1)
    label = np.array(label)
    return label

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.argmax(dim=1,keepdim=True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_f1 = f1_score(target.cpu().detach().numpy(),pred.t().cpu().detach().numpy(),average = 'micro')

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], pred.t()

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef

        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        weights = torch.FloatTensor()
        self.criterion  = nn.CrossEntropyLoss()
        self.fc1 = nn.Linear(nOut,128)
        self.fc2 = nn.Linear(128,nClasses)

        print('Initialised Softmax Loss')

    def forward(self, x, label=None):

        x = self.fc1(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)
        nloss = self.criterion(x, label)
        prec1,pred	= accuracy(x.detach(), label.detach(), topk=(1,))

        return nloss, prec1,pred
