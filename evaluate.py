import dataset
import model
import Multiordersimulation
import glob
import utils
import warnings
warnings.filterwarnings('ignore')

import torch
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

from sklearn.metrics import f1_score
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import time
from tqdm import tqdm
from scipy.io import wavfile
import argparse
from scipy.signal import lfilter, stft
import librosa
import soundfile as sf
import torchaudio
import random
import torchaudio.functional as AF
import numpy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--rir_path', type=str, default = '/voice_spoofing_detection/AIR/*.wav', help='path to RIR audio file')
parser.add_argument('--clean_path', type=str, default = '/voice_spoofing_detection/wav48_silence_trimmed/*/*.flac', help='path to target clean_audio file')
parser.add_argument('--load_weight', type=str, default = '/voice_spoofing_detection/all_layers.pt', help='path to model weight')
parser.add_argument('--musan_path', type=str, default = '/voice_spoofing_detection/musan_split/', help='path to musan')
parser.add_argument('--test_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_eval/flac/', help='path to test file')
parser.add_argument('--protocol_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_cm_protocols/', help='path to protocol')

# Model configurations.

config = parser.parse_args()

torch.manual_seed(777)
room = glob.glob(config.rir_path)
vctk_path = glob.glob(config.clean_path)
musan_path = glob.glob(config.musan_path)

PA_eval_df_F=pd.read_csv(config.protocol_path+"ASVspoof2019.PA.cm.eval.trl.txt", sep=" ", names=["Speaker","Path","room","device","Label"] )
PA_eval_data_dir = config.test_path
PA_eval_path = utils.make_path(PA_eval_data_dir,PA_eval_df_F)
PA_eval_label =utils.make_label(PA_eval_df_F)

vctk_label = np.ones(len(vctk_path),dtype=int)*0
rir_path = config.rir_path
augment = Multiordersimulation.AugmentWAV(musan_path,rir_path,600)

test_dataset = dataset.AudioDataset(PA_eval_path, PA_eval_label,augment,False)
test_loader = DataLoader(test_dataset, batch_size=64,  num_workers=40,shuffle = False,drop_last = False)

model = model.resnet34().to(device)

criterion = utils.LossFunction(256,2).to(device)
optimizer_softmax = optim.AdamW(criterion.parameters(),lr=0.001, weight_decay=0.0005)
optimizer = optim.AdamW(model.parameters(),lr=0.001, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer,4,gamma=0.5)
sheduler_softmax = lr_scheduler.StepLR(optimizer_softmax,4,gamma=0.5)

def evaluate():
    print('evaluating...')
    criterion.load_state_dict(torch.load(config.load_weight)['loss'])
    model.load_state_dict(torch.load(config.load_weight)['model_state_dict'])

    with torch.no_grad():
        model.eval()
        val_correct = 0
        idx = 0
        val_loss = 0
        val_f1_score = 0
        f1 = 0
        pred_list, emb_list, labels, test_list = [], [], [], []
        for sign , target in tqdm(test_loader,total = len(test_loader),leave=False):

            sign = sign.to(device).float()
            target = target.to(device)

            ip1 = model.forward(sign)
            emb_list.extend(ip1.cpu().detach().numpy())
            labels.extend(target.cpu().detach().numpy())

            emb,out,prediction = criterion.forward(ip1,target)
            pred_list.extend(prediction.cpu().detach().numpy())

            val_correct += out
            val_loss += emb
            idx += 1
            val_f1_score+=f1

        val_eer = val_correct/idx
        test_list.append(val_eer)
        emb_list = np.array(emb_list)
        labels = np.array(labels)
        print("test_acc :",val_eer,"test_loss : ",val_loss/idx, "f1_score", f1_score(pred_list,labels, average='macro'))

def main():
     evaluate()
        
if __name__ == "__main__":
    main()
