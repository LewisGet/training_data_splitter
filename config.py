import os

sample_rate = 16000
n_frames = 64
data_files = os.path.join(".", "dataset", "*.wav")
device = 'cuda'
bias = 0.8
save_path = os.path.join(".", "output")
save_good_path = os.path.join(save_path, "good")
save_bad_path = os.path.join(save_path, "bad")
load_epoch = 8
load_model_path = os.path.join(os.sep, "home", "results", "debug", "ckpts")
load_model_name = "discriminator_A"
gpu_ids = [0]

pre_conver_path = os.path.join(".", "pre_conver")
pre_conver_types = ["m4a", "mp3", "mp4"]
conver_save_path = os.path.join(".", "dataset")

test_wav_save = os.path.join(".", "test")
