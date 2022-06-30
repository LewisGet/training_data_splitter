import torch
import torchaudio
import config
import glob
import librosa
import numpy as np
import os

import matplotlib.pyplot as plt


vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

n_frames = config.n_frames
sample_rate = config.sample_rate
mel_normalized = list()
rev_list = list()
mel_list = list()

wav_files = glob.glob(config.data_files, recursive=True)

for wav_file_path in wav_files:
    wav, _ = librosa.load(wav_file_path, sr=sample_rate, mono=True)
    wav = wav[16000 * 180:16000 * 250]
    librosa.output.write_wav(os.path.join(config.test_wav_save, "org.wav"), wav, sample_rate)

    spec = vocoder(torch.tensor([wav]))
    mel = spec.cpu().detach().numpy()[0]
    mel_list.append(mel)

mel_concatenated = np.concatenate(mel_list, axis=1)
mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

for mel in mel_list:
    step = 64

    for i in range(0, mel.shape[-1], step):
        start, end = i, i + step
        this_mel = mel[:, start:end]
        app = (this_mel - mel_mean) / mel_std

        rev_mel = app * mel_std + mel_mean
        rev_mel = torch.tensor(rev_mel)
        rev = vocoder.inverse(rev_mel.unsqueeze(0)).cpu()
        rev_list.append(rev)

        mel_normalized.append(torch.tensor(app))


for i, rev in enumerate(rev_list):
    file_path = os.path.join(config.test_wav_save, str(i) + "-rev.wav")
    torchaudio.save(file_path, rev, sample_rate=sample_rate)
