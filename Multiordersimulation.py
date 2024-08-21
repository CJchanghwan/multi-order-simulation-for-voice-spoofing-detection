import glob
from scipy import signal
from scipy.signal import fftconvolve, deconvolve
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
import time
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import lfilter, stft
import librosa
import soundfile as sf
import torchaudio
import random
torch.manual_seed(777)

class Resample(torch.nn.Module):
    def __init__(
        self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):

        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform

    def _perform_resample(self, waveforms):

        batch_size, num_channels, wave_len = waveforms.size()
        window_size = self.weights.size(1)
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = torch.zeros(
            (batch_size, num_channels, tot_output_samp),
            device=waveforms.device,
        )
        self.weights = self.weights.to(waveforms.device)


        if waveforms.device != self.weights.device:
            self.weights = self.weights.to(waveforms.device)

        eye = torch.eye(num_channels, device=waveforms.device).unsqueeze(2)

        for i in range(self.first_indices.size(0)):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                wave_to_conv = wave_to_conv[..., first_index:]

            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = torch.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding)
            )
            conv_wave = torch.nn.functional.conv1d(
                input=wave_to_conv,
                weight=self.weights[i].repeat(num_channels, 1, 1),
                stride=self.conv_stride,
                groups=num_channels,
            )

            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=self.conv_transpose_stride
            )

            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.size(-1)
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding)
            )
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out
        last_output_samp = interval_length // ticks_per_output_period

        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = torch.arange(
            start=0.0, end=self.output_samples, device=waveforms.device,
        )
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * self.orig_freq)
        max_input_index = torch.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = torch.arange(max_weight_width, device=waveforms.device)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        weights[inside_window_indices] = 0.5 * (
            1
            + torch.cos(
                2
                * math.pi
                * lowpass_cutoff
                / self.lowpass_filter_width
                * delta_t[inside_window_indices]
            )
        )

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_audio= max_frames * 160 #+ 240
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist = {}
        #augment_files = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        '''
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]] += [file]
        '''
        self.rir_files = rir_path
        self.perturb_prob = 1.0
        self.speeds = [90,100,110]
        self.sample_rate = 16000
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.sample_rate,
                "new_freq" : self.sample_rate*speed//100,
            }
            self.resamplers += [Resample(**config)]

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0]**2) + 1e-4)
            noises += [np.sqrt(10**((clean_db - noise_db - noise_snr) / 10)) * noiseaudio]
        return np.sum(np.concatenate(noises,axis=0), axis=0,keepdims=True) + audio

    def reverberate(self, audio,rir_files,rir_file2):    
     
        global label

        rir_file = random.choice(rir_files)
        rir_file2 = random.choice(rir_files)
        
        rir, sample_rate = librosa.load(rir_file,sr=16000)
        rir = torch.FloatTensor(rir)
        resampler = torchaudio.transforms.Resample(sample_rate,16000)
        rir = resampler(rir)
        rir = rir.squeeze()
        rir = rir.cpu().detach().numpy()
        
        rir2, sample_rate = librosa.load(rir_file2)
        rir2 = torch.FloatTensor(rir2)
        resampler = torchaudio.transforms.Resample(sample_rate,16000)
        rir2 = resampler(rir2)
        rir2 = rir2.squeeze()
        rir2 = rir2.cpu().detach().numpy()

        rir = 0.5 * rir/np.max(rir)
        rir2 = 0.5 * rir/np.max(rir2)
      
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir2 = np.expand_dims(rir2.astype(np.float), 0)
        
        dice = random.randint(0,2)
        
        if dice == 0:
            audio = audio
            audio = audio[:,:self.max_audio]
            audio = 0.5*audio/np.max(audio)
        
        elif dice == 1:
            audio = fftconvolve(audio, rir, mode='full')[:,:self.max_audio]
            audio = 0.5*audio/np.max(audio)
        
        elif dice == 2 :
            audio = fftconvolve(audio, rir, mode='full')[:,:self.max_audio]
            audio = fftconvolve(audio, rir2, mode='full')[:,:self.max_audio]
            audio = 0.5*audio/np.max(audio)
    
        return audio, dice
    
    def speed_perturb(self, audio):
        samp_index = random.randint(0,2)
        return self.resamplers[samp_index](torch.FloatTensor(audio)).detach().cpu().numpy()
