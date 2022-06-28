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
