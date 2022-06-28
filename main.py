import sys
import os
import torch
import config
import glob
from torch.utils.data.dataset import Dataset
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment

_parentdir = os.path.join(os.path.curdir, "..", "MaskCycleGAN-VC")
sys.path.insert(0, str(_parentdir))

from mask_cyclegan_vc.model import Discriminator
from mask_cyclegan_vc.utils import decode_melspectrogram

sys.path.remove(str(_parentdir))

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


def conversion_to_wav(org_dir, types, target_dir):
    for type_name in types:
        files = glob.glob(os.path.join(org_dir, "*." + type_name))

        for file in files:
            name = os.path.splitext(os.path.basename(file))[0]
            name = name + "_" + type_name
            audio = AudioSegment.from_file(file)

            # , frame_rate=22050, channels=1, sample_width=2

            audio = audio.set_frame_rate(config.sample_rate)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)

            audio.export("%s.wav" % os.path.join(target_dir, name), format="wav")


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

            step = 64
            for i, ii in enumerate(range(0, mel.shape[-1], step)):
                start, end = i * step, (i + 1) * step
                app = (mel[:, start:end] - self.mel_mean) / self.mel_std
                self.mel_normalized.append(app)

    def __getitem__(self, index):
        return self.mel_normalized[index]

    def __len__(self):
        return len(self.mel_normalized)


def load_model(model, name):
    name = f'{str(config.load_epoch).zfill(5)}_{name}.pth.tar'
    ckpt_path = os.path.join(config.load_model_path, name)
    checkpoint = torch.load(ckpt_path, map_location=torch.cuda.set_device(config.gpu_ids[0]))
    model.load_state_dict(checkpoint['model_state'])


class DataDiscriminator:
    def __init__(self):
        self.device = config.device
        self.d = Discriminator().to(self.device)
        self.vocoder = vocoder
        self.sample_rate = config.sample_rate
        load_model(self.d, config.load_model_name)
        self.dataset = RawDataset()
        self.feed_dataset = torch.utils.data.DataLoader(self.dataset)

    def pick(self):
        for i, data in enumerate(self.feed_dataset):
            data = data.to(self.device, dtype=torch.float)
            label = torch.mean(self.d(data))
            wav = decode_melspectrogram(self.vocoder, data[0].detach().cpu(), self.dataset.mel_mean, self.dataset.mel_std).cpu()

            if label > config.bias:
                filepath = os.path.join(config.save_good_path, "good_" + str(i) + ".wav")
                torchaudio.save(filepath, wav, sample_rate=self.sample_rate)
                print("save: " + filepath)
            else:
                filepath = os.path.join(config.save_bad_path, "bad_" + str(i) + ".wav")
                torchaudio.save(filepath, wav, sample_rate=self.sample_rate)
                print("save: " + filepath)


if __name__ == "__main__":
    conversion_to_wav(config.pre_conver_path, config.pre_conver_types, config.conver_save_path)
    e = DataDiscriminator()
    e.pick()
