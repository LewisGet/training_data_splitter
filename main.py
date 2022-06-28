import sys
import os
import torch
import config
import glob
from torch.utils.data.dataset import Dataset
import librosa
import numpy as np

_parentdir = os.path.join(os.path.curdir, "..", "MaskCycleGAN-VC")
sys.path.insert(0, str(_parentdir))

from mask_cyclegan_vc.model import Discriminator

sys.path.remove(str(_parentdir))

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


class RawDataset(Dataset):
    def __init__(self):
        self.n_frames = config.n_frames
        self.sample_rate = config.sample_rate
        self.mel_normalized = list()

        wav_files = glob.glob(config.data_files, recursive=True)

        mel_list = list()

        for wav_file_path in wav_files:
            wav, _ = librosa.load(wav_file_path, sr=self.sample_rate, mono=True)
            spec = vocoder(torch.tensor([wav]))
            mel = spec.cpu().detach().numpy()[0]
            mel_list.append(mel)

        self.mel_concatenated = np.concatenate(mel_list, axis=1)
        self.mel_mean = np.mean(self.mel_concatenated, axis=1, keepdims=True)
        self.mel_std = np.std(self.mel_concatenated, axis=1, keepdims=True) + 1e-9

        for mel in mel_list:
            if mel.shape[-1] < 64:
                continue

            app = (mel - self.mel_mean) / self.mel_std
            self.mel_normalized.append(app)

    def __getitem__(self, index):
        return self.mel_normalized[index]

    def __len__(self):
        return len(self.mel_normalized)


def load_model(model, name):
    name = f'{str(config.load_epoch).zfill(5)}_{name}.pth.tar'
    ckpt_path = os.path.join(config.load_model_path, name)
    checkpoint = torch.load(ckpt_path, map_location=config.gpu_ids[0])
    model.load_state_dict(checkpoint['model_state'])


class DataDiscriminator:
    def __init__(self):
        self.device = config.device
        self.d = Discriminator().to(self.device)
        self.vocoder = vocoder
        self.sample_rate = config.sample_rate
        load_model(self.d, "discriminator_A")
