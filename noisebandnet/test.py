import torch
import sounddevice as sd

# Vypsání dostupných zařízení
print(sd.query_devices())

# Příklad: nastavení vzorkovací frekvence podle vašeho zařízení (upravte dle potřeby)
fs = 44100  # nebo např. 48000

duration = 1  # délka signálu v sekundách
t = torch.linspace(0, duration, int(fs * duration))
audio_tensor = torch.sin(2 * torch.pi * 440 * t)

if audio_tensor.is_cuda:
    audio_tensor = audio_tensor.cpu()

audio_np = audio_tensor.numpy()

# Přehrajeme zvuk, případně můžete explicitně nastavit zařízení
sd.play(audio_np, fs)
sd.wait()