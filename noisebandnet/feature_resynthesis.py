import torch
from dataset_tool import compute_loudness, compute_centroid
from IPython.display import Audio
import pickle
import librosa as li


from noisebandnet.model import NoiseBandNet
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import sounddevice as sd
import soundfile as sf
import os
from os import path
import argparse


device = 'cuda'

'''
TRAIN_PATH = 'trained_models/drill/2025_02_09_13_32_30'
MODEL_PATH = f'{TRAIN_PATH}/model_4950.ckpt'
CONFIG_PATH = f'{TRAIN_PATH}/config.pickle'

CONTROL_PARAM_PATH = 'synth_curve/drill/drill_power.npy'
'''

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', help='Directory of the training sound/sounds', default=None)
parser.add_argument('--param_dir', help='Cesta ke složce pro uložení vygenerovaného audia.', default="./test_sounds/")
parser.add_argument('--save_audio_dir', help='Cesta ke složce pro uložení vygenerovaného audia.', default="./test_sounds/")
config = parser.parse_args()

SAVE_DIR = config.save_audio_dir
TRAIN_PATH = config.model_path
MODEL_PATH = f'{TRAIN_PATH}/model_4.ckpt'
CONFIG_PATH = f'{TRAIN_PATH}/config.pickle'

CONTROL_PARAM_DIR = config.param_dir


'''
TRAIN_PATH = 'trained_models/car_neutral2/2025_02_19_08_34_25'
MODEL_PATH = f'{TRAIN_PATH}/model_19999.ckpt'
CONFIG_PATH = f'{TRAIN_PATH}/config.pickle'

CONTROL_PARAM_PATH = 'synth_curve/car_neutral2/rpm.npy'
'''
with (open(CONFIG_PATH, "rb")) as f:
    config = pickle.load(f)

synth = NoiseBandNet(hidden_size=config.hidden_size, n_band=config.n_band, synth_window=config.synth_window, n_control_params=config.n_control_params).to(device).float()

synth.load_state_dict(torch.load(MODEL_PATH))


'''
control_param = np.load(CONTROL_PARAM_PATH)
control_param = torch.from_numpy(control_param).unsqueeze(0).unsqueeze(0).float().to(device)
# Add interpolation to match expected time resolution
control_param = F.interpolate(input=control_param, scale_factor=1/config.synth_window, mode='linear')

# Ensure correct shape and contiguity
control_param = [control_param.permute(0,2,1)]

#control_param = [control_param.float().permute(0,2,1)]

#print(control_param[0].shape)
'''

control_params = []
for filename in sorted(os.listdir(CONTROL_PARAM_DIR)):
    if filename.endswith('.npy'):
        control_param_path = os.path.join(CONTROL_PARAM_DIR, filename)
        control_param = np.load(control_param_path)
        control_param = torch.from_numpy(control_param).unsqueeze(0).unsqueeze(0).float().to(device)
        # Interpolace
        control_param = F.interpolate(input=control_param, scale_factor=1/config.synth_window, mode='linear').permute(0,2,1)
        control_params.append(control_param)

with torch.no_grad():
    y_audio = synth(control_params=control_params)
#Audio(y_audio[0][0].detach().cpu().numpy(), rate=config.sampling_rate)

sd.play(y_audio[0].squeeze().detach().cpu().numpy(),44100)
sd.wait()


sf.write(path.join(SAVE_DIR,"civetta_kotva_nbn.wav"),y_audio[0].squeeze().detach().cpu().numpy(),44100 )

#fig, ax = plt.subplots()
#D = li.stft(y_audio[0][0].detach().cpu().numpy(), n_fft=1024, hop_length=256)
#S_db = li.amplitude_to_db(np.abs(D), ref=np.max)
#img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax, sr=config.sampling_rate, cmap='magma', hop_length=256)