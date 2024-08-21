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

torch.manual_seed(777)
parser = argparse.ArgumentParser()

parser.add_argument('--rir_path', type=str, default = '/voice_spoofing_detection/AIR/*.wav', help='path to RIR audio file')
parser.add_argument('--clean_path', type=str, default = '/voice_spoofing_detection/wav48_silence_trimmed/*/*.flac', help='path to target clean_audio file')
parser.add_argument('--load_weight', type=str, default = '/voice_spoofing_detection/acoustic_resnet.pt', help='path to model weight')
parser.add_argument('--musan_path', type=str, default = '/voice_spoofing_detection/musan_split/', help='path to musan')
parser.add_argument('--test_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_eval/flac/', help='path to test file')
parser.add_argument('--protocol_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_cm_protocols/', help='path to protocol')
parser.add_argument('--train_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_train/flac/', help='path to protocol')
parser.add_argument('--val_path', type=str, default = '/voice_spoofing_detection/PA/ASVspoof2019_PA_dev/flac/', help='path to protocol')
parser.add_argument('--save_path', type=str, default='/voice_spoofing_detection/all_layers.pt', help='path to save')
parser.add_argument('--epoch', type=int, default=10, help='set epoch')
config = parser.parse_args()

room = glob.glob(config.rir_path)
vctk_path = glob.glob(config.clean_path)
musan_path = glob.glob(config.musan_path)


PA_train_df_F=pd.read_csv(config.protocol_path+"ASVspoof2019.PA.cm.train.trn.txt", sep=" ", names=["Speaker","Path","room","device","Label"] )
PA_dev_df_F=pd.read_csv(config.protocol_path+"ASVspoof2019.PA.cm.dev.trl.txt", sep=" ", names=["Speaker","Path","room","device","Label"] )
PA_val_df_F=pd.read_csv(config.protocol_path+"ASVspoof2019.PA.cm.eval.trl.txt", sep=" ", names=["Speaker","Path","room","device","Label"] )

PA_train_data_dir = config.train_path
PA_dev_data_dir = config.val_path
PA_val_data_dir = config.test_path

PA_train_path = utils.make_path(PA_train_data_dir,PA_train_df_F)
PA_dev_path = utils.make_path(PA_dev_data_dir,PA_dev_df_F)
PA_val_path = utils.make_path(PA_val_data_dir,PA_val_df_F)

PA_train_label =utils.make_label(PA_train_df_F)
PA_dev_label =utils.make_label(PA_dev_df_F)
PA_val_label =utils.make_label(PA_val_df_F)
vctk_label = np.ones(len(vctk_path),dtype=int)*0

train_path = np.concatenate((PA_train_path,PA_dev_path),axis = 0)
train_label = np.concatenate((PA_train_label,PA_dev_label),axis = 0)

musan_path = config.musan_path
rir_path = config.rir_path

augment = Multiordersimulation.AugmentWAV(musan_path,rir_path,600)

train_dataset = dataset.AudioDataset(train_path, train_label ,augment,True)
train_loader = DataLoader(train_dataset, batch_size=64,  num_workers=40,shuffle = True,drop_last = True)

test_dataset = dataset.AudioDataset(PA_val_path, PA_val_label,augment,False)
test_loader = DataLoader(test_dataset, batch_size=64,  num_workers=40,shuffle = False,drop_last = False)

model = model.resnet34().to(device)
model.load_state_dict(torch.load(config.load_weight))
print("Initialized pretrained with multi-order simulation....")

criterion = utils.LossFunction(256,2).to(device)
optimizer_softmax = optim.AdamW(criterion.parameters(),lr=0.001, weight_decay=0.0005)
optimizer = optim.AdamW(model.parameters(),lr=0.001, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer,4,gamma=0.5)
sheduler_softmax = lr_scheduler.StepLR(optimizer_softmax,4,gamma=0.5)

def train(epoch):
    print('start training....')
    best = 0
    best_epoch = 0
    train_acc, train_loss, val_acc, validate_loss = [], [], [], []

    val_acc = []
    validate_loss = []

    for epoch in range(epoch):
        sheduler.step()
        total_loss, val_correct = 0,0
        EER = []
        pred = 0
        val_eer = 0
        f1 = 0
        test_list, prediction_list, label_list, ip1_loader, idx_loader, labels, emb_list = [],[],[],[],[],[],[]
        cnt, idx, loss, top1, train_f1_score = 0, 0, 0, 0, 0
        for i,(data, target) in tqdm(enumerate(train_loader),total = len(train_loader),leave=False):
            model.train()
            data, target = data.to(device), target.to(device)
            labels.extend(target.cpu().detach().numpy())


            ip1 = model.forward(data) ## output = batch, output_dim
            emb_list.extend(ip1.cpu().detach().numpy())

            sloss, pred,predict = criterion.forward(ip1,target)

            optimizer.zero_grad()
            optimizer_softmax.zero_grad()

            sloss.backward()

            optimizer.step()
            optimizer_softmax.step()


            loss += sloss.detach().cpu().item()
            top1 += pred.detach().cpu().item()
            cnt+=1

        sheduler_softmax.step()
        sheduler.step()
        train_acc.append(top1/cnt)
        train_loss.append(loss/cnt)
        emb_list = np.array(emb_list)
        labels = np.array(labels)
        print("epoch : ",epoch,"Accuracy : ",top1/cnt, "loss : ", loss/cnt)

        with torch.no_grad():
            model.eval()
            val_correct = 0
            idx = 0
            val_loss = 0
            val_f1_score = 0
            label_list, pred_list, emb_list, labels = [], [], [], []
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
            validate_loss.append(val_loss/idx)
            emb_list = np.array(emb_list)
            labels = np.array(labels)
            print("test_acc :",val_eer,"test_loss : ",val_loss/idx)

            if val_eer > best :
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion.state_dict(),
                }, config.save_path) # non_aug #masking # speed
                best = val_eer
                best_epoch = epoch
                print("model saved")


def main():
     train(config.epoch)
if __name__ == "__main__":
    main()
