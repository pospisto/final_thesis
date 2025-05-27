"""
import argparse
import csv
import os
from os import path
import time
import datetime
import dateutil.tz

import soundfile as sf
import torch
import yaml
from auraloss import freq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='DDSP Training Script')
    parser.add_argument('--config', default='./config.yaml', help='Path to config.yaml file')
    parser.add_argument('--save_dir_path', default='runs_car/debug2', help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--stop_lr', type=float, default=1e-3, help='Stopping learning rate')
    parser.add_argument('--decay_over', type=int, default=400000, help='Number of steps to decay learning rate over')
    parser.add_argument('--audio_save_interval', type=int, default=10, help='Save audio every N epochs')
    parser.add_argument('--num_samples', type=int, default=64000, help='Number of samples in signal')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of samples in signal')
    return parser.parse_args()

def create_save_dir(save_dir):
    current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'trained_models/{save_dir}/{current_time}'
    if os.path.exists(output_dir) == False:
        print(f'Creating saving directory in {output_dir}')
        os.makedirs(output_dir)
    else:
        print(f'Saving model in: {output_dir}')
    return output_dir


def load_config(config_path,num_samples, hidden_size):

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["preprocess"]["signal_length"] = num_samples
    config["model"]["hidden_size"] = hidden_size

    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file)
    return config


def setup_device():

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(config, batch_size):

    dataset = Dataset(config["preprocess"]["out_dir"])
    return torch.utils.data.DataLoader(dataset, batch_size, True, drop_last=True)


def setup_training(config, args, device,save_dir):

    model = DDSP(**config["model"]).to(device)
    dataloader = create_dataloader(config, args.batch)
    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness
    writer = SummaryWriter(save_dir, flush_secs=20)
    with open(path.join(save_dir, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    scheduler = get_scheduler(len(dataloader), args.start_lr, args.stop_lr, args.decay_over)
    return model, dataloader, optimizer, scheduler, writer


def setup_logging_files(output_directory):

    loss_file_path = path.join(output_directory, "loss_log.csv")
    rmse_file_path = path.join(output_directory, "rmse_log.csv")
    correlation_file_path = path.join(output_directory, "correlation_log.csv")
    for file_path, header in [(loss_file_path, ["Epoch", "MRSTFT Loss"]),
                                (rmse_file_path, ["Epoch", "RMSE"]),
                                (correlation_file_path, ["Epoch", "Correlation"])]:
        if not path.exists(file_path):
            with open(file_path, mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
    return loss_file_path, rmse_file_path, correlation_file_path


def train_epoch(model, dataloader, optimizer, scheduler, writer, config, device, epoch,
                csv_file_path, rmse_file_path, correlation_file_path, audio_save_interval):

    model.train()
    mean_loss = 0
    n_element = 0
    step = 0
    mrstft = freq.MultiResolutionSTFTLoss(
        fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
        hop_sizes=[8192 // 4, 4096 // 4, 2048 // 4, 1024 // 4, 512 // 4, 128 // 4, 32 // 4],
        win_lengths=[8192, 4096, 2048, 1024, 512, 128, 32],
    )

    for s, p, l in dataloader:
        s = s.unsqueeze(1).to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)
        l = (l - config["data"]["mean_loudness"]) / config["data"]["std_loudness"]
        y = model(p, l).permute(0, 2, 1)
        loss = mrstft(y, s.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), step)
        step += 1
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    return mean_loss, s, y, loss.item()  # Return mean_loss, s, y, and loss.item() for logging


def log_epoch_results(model, writer, scheduler, mean_loss, s, y, config, epoch,
                      loss_file_path, rmse_file_path, correlation_file_path, audio_save_interval, mrstft_loss, save_dir):

    writer.add_scalar("lr", scheduler(epoch), epoch)
    writer.add_scalar("reverb_decay", model.reverb.decay.item(), epoch)
    writer.add_scalar("reverb_wet", model.reverb.wet.item(), epoch)
    if epoch % audio_save_interval == 0:
        with open(loss_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, mrstft_loss]) # Ukládáme mrstft_loss
        rmse = torch.sqrt(torch.mean((s - y.squeeze(1)) ** 2)).item()
        correlation = torch.corrcoef(torch.stack((s.flatten().cpu(), y.squeeze(1).flatten().cpu())))[0, 1].item()
        with open(rmse_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, rmse])
        with open(correlation_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, correlation])

        first_s = s[0].squeeze(0).detach().cpu().numpy()
        first_y = y[0].squeeze(0).detach().cpu().numpy()
        sf.write(path.join(save_dir, f"x_audio_{epoch:06d}.wav"),first_s, config["preprocess"]["sampling_rate"])
        sf.write(path.join(save_dir, f"y_audio_{epoch:06d}.wav"), first_y,config["preprocess"]["sampling_rate"])

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config,args.num_samples,args.hidden_size)

    output_dir = create_save_dir(args.save_dir_path)

    device = setup_device()
    model, dataloader, optimizer, scheduler, writer = setup_training(config, args, device,output_dir)
    loss_file_path, rmse_file_path, correlation_file_path = setup_logging_files(output_dir)

    best_loss = float("inf")

    start_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "w") as f:
        f.write(f"Training started at: {time.ctime(start_time)}\n")

    for epoch in tqdm(range(args.epochs)):
        mean_loss, s, y, mrstft_loss = train_epoch(model, dataloader, optimizer, scheduler, writer, config, device, epoch,
                    loss_file_path, rmse_file_path, correlation_file_path, args.audio_save_interval)
        log_epoch_results(model, writer, scheduler, mean_loss, s, y, config, epoch,
                      loss_file_path, rmse_file_path, correlation_file_path, args.audio_save_interval, mrstft_loss,output_dir) # Předáme mrstft_loss do log_epoch_results
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), path.join(output_dir, "state.pth"))

    end_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "a") as f:
        f.write(f"Training ended at: {time.ctime(end_time)}\n")
        f.write(f"Total training time: {end_time - start_time} seconds\n")
"""




import argparse
import csv
import os
from os import path
import time
import datetime
import dateutil.tz

import soundfile as sf
import torch
import yaml
from auraloss import freq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='DDSP Training Script')
    parser.add_argument('--config', default='./config.yaml', help='Path to config.yaml file')
    parser.add_argument('--save_dir_path', default='runs_car/debug2', help='Output directory for results')
    parser.add_argument("--features", default=None, help="Comma-separated list of features to process.")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--stop_lr', type=float, default=1e-3, help='Stopping learning rate')
    parser.add_argument('--decay_over', type=int, default=400000, help='Number of steps to decay learning rate over')
    parser.add_argument('--audio_save_interval', type=int, default=10, help='Save audio every N epochs')
    parser.add_argument('--num_samples', type=int, default=64000, help='Number of samples in signal')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of samples in signal')
    return parser.parse_args()


def apply_regression_torch(tensor):

    a5, a4, a3, a2, a1, a0 = -132.882376, 317.680449, -264.407627, 88.613942, -6.465998, -0.891762
    out = torch.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            x = float(tensor[i, j])
            z = float(a5 * (x ** 5) + a4 * (x ** 4) + a3 * (x ** 3) + a2 * (x ** 2) + a1 * x + a0)
            out[i, j] = z

    return out

def create_save_dir(save_dir):
    current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'trained_models/{save_dir}/{current_time}'
    if os.path.exists(output_dir) == False:
        print(f'Creating saving directory in {output_dir}')
        os.makedirs(output_dir)
    else:
        print(f'Saving model in: {output_dir}')
    return output_dir


def load_config(config_path,num_samples, hidden_size):

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["preprocess"]["signal_length"] = num_samples
    config["model"]["hidden_size"] = hidden_size

    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file)
    return config


def setup_device():

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(config, batch_size, features):

    dataset = Dataset(config["preprocess"]["out_dir"],features)
    return torch.utils.data.DataLoader(dataset, batch_size, True, drop_last=True)


def setup_training(config, args, device,save_dir,features):

    model = DDSP(**config["model"]).to(device)
    dataloader = create_dataloader(config, args.batch,features)
    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness
    writer = SummaryWriter(save_dir, flush_secs=20)
    with open(path.join(save_dir, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    scheduler = get_scheduler(len(dataloader), args.start_lr, args.stop_lr, args.decay_over)
    return model, dataloader, optimizer, scheduler, writer


def setup_logging_files(output_directory):

    loss_file_path = path.join(output_directory, "loss_log.csv")
    rmse_file_path = path.join(output_directory, "rmse_log.csv")
    correlation_file_path = path.join(output_directory, "correlation_log.csv")
    for file_path, header in [(loss_file_path, ["Epoch", "MRSTFT Loss"]),
                                (rmse_file_path, ["Epoch", "RMSE"]),
                                (correlation_file_path, ["Epoch", "Correlation"])]:
        if not path.exists(file_path):
            with open(file_path, mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
    return loss_file_path, rmse_file_path, correlation_file_path


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config,args.num_samples,args.hidden_size)

    output_dir = create_save_dir(args.save_dir_path)

    device = setup_device()

    features_names = [feature.strip().replace(" ", "_").replace("(m/s)", "m_s") for feature in args.features.split(',')]

    model, dataloader, optimizer, scheduler, writer = setup_training(config, args, device,output_dir,features_names)
    loss_file_path, rmse_file_path, correlation_file_path = setup_logging_files(output_dir)

    best_loss = float("inf")
    mean_loss = 0
    n_element = 0
    step = 0

    mrstft = freq.MultiResolutionSTFTLoss(
        fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
        hop_sizes=[8192 // 4, 4096 // 4, 2048 // 4, 1024 // 4, 512 // 4, 128 // 4, 32 // 4],
        win_lengths=[8192, 4096, 2048, 1024, 512, 128, 32],
    )

    start_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "w") as f:
        f.write(f"Training started at: {time.ctime(start_time)}\n")

    for epoch in tqdm(range(args.epochs)):
        model.train()
        for s, p, l, f in dataloader:
            s = s.unsqueeze(1).to(device)
            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)
            l = (l - config["data"]["mean_loudness"]) / config["data"]["std_loudness"]

            rpm = apply_regression_torch(f['revolutions_per_minute']).unsqueeze(-1).to(device)
            vel = f['velocity_m_s'].unsqueeze(-1).to(device)
            thr = f['throttle'].unsqueeze(-1).to(device)

            y = model(p, rpm, vel, thr).permute(0, 2, 1)
            mrstft_loss = mrstft(y, s.unsqueeze(1))
            optimizer.zero_grad()
            mrstft_loss.backward()
            optimizer.step()
            writer.add_scalar("loss", mrstft_loss.item(), step)
            step += 1
            n_element += 1
            mean_loss += (mrstft_loss.item() - mean_loss) / n_element

        if not (epoch + 1) % args.audio_save_interval:
            writer.add_scalar("lr", scheduler(epoch), epoch)
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), epoch)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), epoch)

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), path.join(output_dir, "state.pth"))

            with open(loss_file_path, mode="a", newline='') as file:
                csv.writer(file).writerow([epoch + 1, mean_loss])

            rmse = torch.sqrt(torch.mean((s - y.squeeze(1)) ** 2)).item()
            correlation = torch.corrcoef(torch.stack((s.flatten().cpu(), y.squeeze(1).flatten().cpu())))[0, 1].item()
            with open(rmse_file_path, mode="a", newline='') as file:
                csv.writer(file).writerow([epoch + 1, rmse])
            with open(correlation_file_path, mode="a", newline='') as file:
                csv.writer(file).writerow([epoch + 1, correlation])

            first_s = s[0].squeeze(0).detach().cpu().numpy()
            first_y = y[0].squeeze(0).detach().cpu().numpy()
            sf.write(path.join(output_dir, f"x_audio_{(epoch + 1):06d}.wav"),first_s, config["preprocess"]["sampling_rate"])
            sf.write(path.join(output_dir, f"y_audio_{(epoch + 1):06d}.wav"), first_y,config["preprocess"]["sampling_rate"])

            mean_loss = 0
            n_element = 0

    end_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "a") as f:
        f.write(f"Training ended at: {time.ctime(end_time)}\n")
        f.write(f"Total training time: {end_time - start_time} seconds\n")











"""
import argparse
import csv
import os
from os import path
import time
import datetime
import dateutil.tz

import soundfile as sf
import torch
import yaml
from auraloss import freq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='DDSP Training Script')
    parser.add_argument('--config', default='./config.yaml', help='Path to config.yaml file')
    parser.add_argument('--save_dir_path', default='runs_car/debug2', help='Output directory for results')
    parser.add_argument("--features", default=None, help="Comma-separated list of features to process.")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--stop_lr', type=float, default=1e-3, help='Stopping learning rate')
    parser.add_argument('--decay_over', type=int, default=400000, help='Number of steps to decay learning rate over')
    parser.add_argument('--audio_save_interval', type=int, default=10, help='Save audio every N epochs')
    parser.add_argument('--num_samples', type=int, default=64000, help='Number of samples in signal')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of samples in signal')
    return parser.parse_args()


def apply_regression_torch(tensor):

    a5, a4, a3, a2, a1, a0 = -132.882376, 317.680449, -264.407627, 88.613942, -6.465998, -0.891762
    out = torch.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            x = float(tensor[i, j])
            z = float(a5 * (x ** 5) + a4 * (x ** 4) + a3 * (x ** 3) + a2 * (x ** 2) + a1 * x + a0)
            out[i, j] = z

    return out

def create_save_dir(save_dir):
    current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'trained_models/{save_dir}/{current_time}'
    if os.path.exists(output_dir) == False:
        print(f'Creating saving directory in {output_dir}')
        os.makedirs(output_dir)
    else:
        print(f'Saving model in: {output_dir}')
    return output_dir


def load_config(config_path, num_samples, hidden_size):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["preprocess"]["signal_length"] = num_samples
    config["model"]["hidden_size"] = hidden_size

    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file)
    return config


def setup_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(config, batch_size, features):
    dataset = Dataset(config["preprocess"]["out_dir"],features)
    return torch.utils.data.DataLoader(dataset, batch_size, True, drop_last=True)


def setup_training(config, args, device, save_dir,features):
    model = DDSP(**config["model"]).to(device)
    dataloader = create_dataloader(config, args.batch, features)
    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness
    writer = SummaryWriter(save_dir, flush_secs=20)
    with open(path.join(save_dir, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    scheduler = get_scheduler(len(dataloader), args.start_lr, args.stop_lr, args.decay_over)
    return model, dataloader, optimizer, scheduler, writer


def setup_logging_files(output_directory):
    loss_file_path = path.join(output_directory, "loss_log.csv")
    rmse_file_path = path.join(output_directory, "rmse_log.csv")
    correlation_file_path = path.join(output_directory, "correlation_log.csv")
    for file_path, header in [(loss_file_path, ["Epoch", "MRSTFT Loss"]),
                              (rmse_file_path, ["Epoch", "RMSE"]),
                              (correlation_file_path, ["Epoch", "Correlation"])]:
        if not path.exists(file_path):
            with open(file_path, mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
    return loss_file_path, rmse_file_path, correlation_file_path


def train_epoch(model, dataloader, optimizer, scheduler, writer, config, device, epoch,
                csv_file_path, rmse_file_path, correlation_file_path, audio_save_interval):
    model.train()
    mean_loss = 0
    n_element = 0
    step = 0
    mrstft = freq.MultiResolutionSTFTLoss(
        fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
        hop_sizes=[8192 // 4, 4096 // 4, 2048 // 4, 1024 // 4, 512 // 4, 128 // 4, 32 // 4],
        win_lengths=[8192, 4096, 2048, 1024, 512, 128, 32],
    )

    for s, p, l, f in dataloader:
        s = s.unsqueeze(1).to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)
        l = (l - config["data"]["mean_loudness"]) / config["data"]["std_loudness"]

        rpm = apply_regression_torch(f['revolutions_per_minute']).unsqueeze(-1).to(device)
        vel = f['velocity_m_s'].unsqueeze(-1).to(device)
        thr = f['throttle'].unsqueeze(-1).to(device)

        y = model(p, rpm, vel, thr).permute(0, 2, 1)
        loss = mrstft(y, s.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), step)
        step += 1
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    return mean_loss, s, y, loss.item()  # Return mean_loss, s, y, and loss.item() for logging


def log_epoch_results(model, writer, scheduler, mean_loss, s, y, config, epoch,
                      loss_file_path, rmse_file_path, correlation_file_path, audio_save_interval, mrstft_loss,
                      save_dir):
    writer.add_scalar("lr", scheduler(epoch), epoch)
    writer.add_scalar("reverb_decay", model.reverb.decay.item(), epoch)
    writer.add_scalar("reverb_wet", model.reverb.wet.item(), epoch)
    if epoch % audio_save_interval == 0:
        with open(loss_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, mrstft_loss])  # Ukládáme mrstft_loss
        rmse = torch.sqrt(torch.mean((s - y.squeeze(1)) ** 2)).item()
        correlation = torch.corrcoef(torch.stack((s.flatten().cpu(), y.squeeze(1).flatten().cpu())))[0, 1].item()
        with open(rmse_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, rmse])
        with open(correlation_file_path, mode="a", newline='') as file:
            csv.writer(file).writerow([epoch, correlation])

        first_s = s[0].squeeze(0).detach().cpu().numpy()
        first_y = y[0].squeeze(0).detach().cpu().numpy()
        sf.write(path.join(save_dir, f"x_audio_{epoch:06d}.wav"), first_s, config["preprocess"]["sampling_rate"])
        sf.write(path.join(save_dir, f"y_audio_{epoch:06d}.wav"), first_y, config["preprocess"]["sampling_rate"])


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config, args.num_samples, args.hidden_size)

    output_dir = create_save_dir(args.save_dir_path)

    device = setup_device()
    features_names = [feature.strip().replace(" ", "_").replace("(m/s)", "m_s") for feature in args.features.split(',')]
    model, dataloader, optimizer, scheduler, writer = setup_training(config, args, device, output_dir,features_names)
    loss_file_path, rmse_file_path, correlation_file_path = setup_logging_files(output_dir)

    best_loss = float("inf")

    start_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "w") as f:
        f.write(f"Training started at: {time.ctime(start_time)}\n")

    for epoch in tqdm(range(args.epochs)):
        mean_loss, s, y, mrstft_loss = train_epoch(model, dataloader, optimizer, scheduler, writer, config, device,
                                                   epoch,
                                                   loss_file_path, rmse_file_path, correlation_file_path,
                                                   args.audio_save_interval)
        log_epoch_results(model, writer, scheduler, mean_loss, s, y, config, epoch,
                          loss_file_path, rmse_file_path, correlation_file_path, args.audio_save_interval, mrstft_loss,
                          output_dir)  # Předáme mrstft_loss do log_epoch_results
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), path.join(output_dir, "state.pth"))

    end_time = time.time()
    with open(path.join(output_dir, "training_time.txt"), "a") as f:
        f.write(f"Training ended at: {time.ctime(end_time)}\n")
        f.write(f"Total training time: {end_time - start_time} seconds\n")
"""