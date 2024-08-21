import torch
from scipy.signal import lfilter, stft
import random
import torchaudio.functional as AF
import numpy as np
import torchaudio
from torch.utils.data import Subset, Dataset, DataLoader
import numpy

torch.manual_seed(777)

def loadWAV(filename, max_frames, evalmode=True, num_eval=1):

    audio, sample_rate = torchaudio.load(filename)
    max_audio = max_frames * int(sample_rate/100)
    resampler = torchaudio.transforms.Resample(sample_rate,16000)
    audio = resampler(audio)
    audio = 0.5*audio/torch.max(audio)
    audio = audio.squeeze()
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1
        audio       = torch.nn.functional.pad(audio, (0, shortage), 'constant',value=0)
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = torch.stack(feats,dim=0)

    return feat.cpu().detach().numpy().astype(np.float32)


class AudioDataset(Dataset):
    def __init__(self,x,y,augment,train=False):
        self.x = x
        self.y = y
        self.train = train
        self.augment_wav = augment
        self.max_audio = 600*160

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        if self.train == True:
            label = self.y[idx]
            audio = loadWAV(self.x[idx],600,evalmode=False,num_eval=2)

            augment = random.randint(0,5)

            if augment == 2 or augment ==1 :
                    audio = self.augment_wav.speed_perturb(audio)
                    if audio.shape[1] > self.max_audio:
                        audio = audio[:, 0 : self.max_audio]
                    else:
                        audio = np.pad(audio[0], (0, self.max_audio-audio.shape[1]), 'wrap')
                        audio = np.expand_dims(audio, 0)
            audio = torch.FloatTensor(audio).squeeze()

        elif self.train ==False :
            audio = loadWAV(self.x[idx],600,evalmode=False,num_eval=2)
            label = self.y[idx]
            audio = torch.FloatTensor(audio).squeeze()

        return audio, label

    
class Multiorder(Dataset):
    def __init__(self, x, y, augment, room):
        self.x = x
        self.y = y
        self.room = room
        self.augment_wav = augment

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
      
        
        audio = loadWAV(self.x[idx],600,evalmode=False,num_eval=2)  
        audio,label = self.augment_wav.reverberate(audio,self.room,self.room)
        audio = torch.FloatTensor(audio).squeeze()
            
            
        return audio, label
