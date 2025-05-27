'''
import torch
import yaml
from effortless_config import Config

from ddsp.core import mean_std_loudness
from preprocess import Dataset
import soundfile as sf
import sounddevice as sd


class args(Config):
    CONFIG = "config4.yaml"
    ROOT = "output"
    BATCH = 16

args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

dataset = Dataset(config["preprocess"]["out_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    shuffle=False,
    drop_last=False,
)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_loudness, std_loudness = mean_std_loudness(dataloader)


#model2 = torch.jit.load("export/ddsp_debug_pretrained.ts")
#model = torch.jit.load("export_sax/ddsp_debug_pretrained.ts")
#model = torch.jit.load("export_stav/ddsp_debug_pretrained.ts")
model = torch.jit.load("export_skrabani/ddsp_debug_pretrained.ts")


#mode2 = torch.jit.load("sax/ddsp_demo_pretrained.ts")

#mean_loudness = model.mean_loudness.item()
#std_loudness = model.std_loudness.item()

#mean_loudness2 = model2.mean_loudness.item()
#std_loudness2 = model2.std_loudness.item()


for s, p, l in dataloader:
    p = p.unsqueeze(-1)
    l = l.unsqueeze(-1)

    l = (l - mean_loudness) / std_loudness

    y = model(p, l).squeeze(-1)
    #l = (l - mean_loudness2) / std_loudness2
    #y2 = model2(p, l).squeeze(-1)

    a1 = y.reshape(-1).detach().cpu().numpy()
    sd.play(a1,config["preprocess"]["sampling_rate"])
    sd.wait()
    #a1 = y2.reshape(-1).detach().cpu().numpy()
    #sd.play(a1, config["preprocess"]["sampling_rate"])
    #sd.wait()





audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()
#audio2 = torch.cat([s, y2], -1).reshape(-1).detach().cpu().numpy()

sf.write(
    "outputeval9.wav",
    audio,
    config["preprocess"]["sampling_rate"],
)

#sf.write(
    #"outputeval5.wav",
    #audio2,
    #config["preprocess"]["sampling_rate"],
#)

'''

"""
from os import path

import numpy as np
import soundfile as sf
import torch
import yaml
from effortless_config import Config
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness, multiscale_fft, safe_log
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset, preprocess

import sounddevice as sd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Directory of the training sound/sounds', default=None)
parser.add_argument('--pth_file', help='Directory of the training sound/sounds', default=None)
parser.add_argument('--audio_file', help='Directory of the training sound/sounds', default=None)
parser.add_argument('--save_audio_dir', help='Directory of the training sound/sounds', default=None)
CONFIG = parser.parse_args()


class args(Config):
    CONFIG = "runs_skrabani/debug2/config.yaml"
    NAME = "debug2"
    ROOT = "runs_skrabani"
    EPOCHS = 20000
    BATCH = 8
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    


#args.parse_args()


#with open(args.CONFIG, "r") as config:
 #   config = yaml.safe_load(config)


with open(CONFIG.config, "r") as config:
    config = yaml.safe_load(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = Dataset(config["preprocess"]["out_dir"])

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     args.BATCH,
#     True,
#     drop_last=True,
# )
# mean_loudness, std_loudness = mean_std_loudness(dataloader)

mean_loudness = config["data"]["mean_loudness"]
std_loudness = config["data"]["std_loudness"]

model = DDSP(**config["model"]).to(device)
model.eval()

model.load_state_dict(torch.load(CONFIG.pth_file, weights_only=True))

x, p, l = preprocess(CONFIG.audio_file, **config["preprocess"])

x = torch.from_numpy(x).to(device).to(torch.float32)
p = torch.from_numpy(p).unsqueeze(-1).to(device).to(torch.float32)
l = torch.from_numpy(l).unsqueeze(-1).to(device).to(torch.float32)
l = (l - mean_loudness) / std_loudness

y = model(p, l).squeeze(-1)

audio_predicted = y.reshape(-1).detach().cpu()
x = x.reshape(-1).detach().cpu()

sd.play(x, config["preprocess"]["sampling_rate"])
sd.wait()

sd.play(audio_predicted, config["preprocess"]["sampling_rate"])
sd.wait()


'''
sf.write(
    "./test_sounds/pottery_input.wav",
    x,
    16000,
)
'''
sf.write(
    "./test_sounds/pottery_predict.wav",
    audio_predicted,
    16000,
)

"""

from os import path, makedirs

import numpy as np
import soundfile as sf
import torch
import yaml
from effortless_config import Config
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness, multiscale_fft, safe_log
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset, preprocess

import sounddevice as sd
import argparse

def apply_regression_torch(tensor):

    a5, a4, a3, a2, a1, a0 = -132.882376, 317.680449, -264.407627, 88.613942, -6.465998, -0.891762
    out = torch.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            x = float(tensor[i, j])
            z = float(a5 * (x ** 5) + a4 * (x ** 4) + a3 * (x ** 3) + a2 * (x ** 2) + a1 * x + a0)
            out[i, j] = z

    return out



parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Cesta ke konfiguračnímu souboru YAML.', default=None)
parser.add_argument('--pth_file', help='Cesta k souboru .pth s uloženými parametry modelu.', default=None)
parser.add_argument('--audio_file', help='Cesta k vstupnímu audio souboru.', default=None)
parser.add_argument('--preproc_dir', help='Cesta k vstupnímu audio souboru.', default=None)
parser.add_argument('--save_audio_dir', help='Cesta ke složce pro uložení vygenerovaného audia.', default="./test_sounds/")
CONFIG = parser.parse_args()

with open(CONFIG.config, "r") as config_file:
    config = yaml.safe_load(config_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_loudness = config["data"]["mean_loudness"]
std_loudness = config["data"]["std_loudness"]

model = DDSP(**config["model"]).to(device)
model.eval()

model.load_state_dict(torch.load(CONFIG.pth_file, weights_only=True))

#x, p, l = preprocess(CONFIG.audio_file, **config["preprocess"])

p = np.load(path.join(CONFIG.preproc_dir,"pitchs.npy"))
l = np.load(path.join(CONFIG.preproc_dir,"loudness.npy"))
rpm = np.load(path.join(CONFIG.preproc_dir,"revolutions_per_minute.npy"))
vel = np.load(path.join(CONFIG.preproc_dir,"velocity_m_s.npy"))
thr = np.load(path.join(CONFIG.preproc_dir,"throttle.npy"))




#x = torch.from_numpy(x).to(device).to(torch.float32)
p = torch.from_numpy(p).unsqueeze(-1).to(device).to(torch.float32)
l = torch.from_numpy(l).unsqueeze(-1).to(device).to(torch.float32)
l = (l - mean_loudness) / std_loudness

rpm = apply_regression_torch(torch.from_numpy(rpm)).unsqueeze(-1).to(device)
vel = torch.from_numpy(vel).unsqueeze(-1).to(device)
thr = torch.from_numpy(thr).unsqueeze(-1).to(device)

y = model(p, rpm, vel, thr).squeeze(-1)

audio_predicted = y.reshape(-1).detach().cpu()
#x_cpu = x.reshape(-1).detach().cpu()

"""
sd.play(x_cpu, config["preprocess"]["sampling_rate"])
sd.wait()
"""

sd.play(audio_predicted, config["preprocess"]["sampling_rate"])
sd.wait()

# Vytvoř složku pro uložení, pokud neexistuje
makedirs(CONFIG.save_audio_dir, exist_ok=True)

"""
sf.write(
    path.join(CONFIG.save_audio_dir, "pottery_input.wav"),
    x_cpu,
    config["preprocess"]["sampling_rate"],
)
"""

sf.write(
    path.join(CONFIG.save_audio_dir, "gavril_20kep.wav"),
    audio_predicted,
    config["preprocess"]["sampling_rate"],
)



