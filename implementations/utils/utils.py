import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn as nn
import logging

import torch
import torchvision.transforms as transforms
from torch_fidelity import calculate_metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os

import warnings

# Suppress the specific warning from torch_fidelity
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")



logging.basicConfig(
    filename="training.log",  # Log file path
    filemode='w',  # Append to the log file (use 'w' to overwrite)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    level=logging.INFO  # Log level (can be DEBUG, INFO, WARNING, ERROR)
)
logging.getLogger().addHandler(logging.StreamHandler())


def evaluate_model(generator, dataloader, latent_dim, num_samples=1000):
    # necessary if the num_samples are not 1000, since the default
    # kid_subset_size is 1000 and it gives a mismatch error
    # if we want other amount of num of samples
    kid_subset_size = num_samples

    output_dir_fakes = "temp/fake_samples"
    output_dir_reals = "temp/real_samples"

    # Create directories to save real and fake samples
    os.makedirs(output_dir_fakes, exist_ok=True)
    os.makedirs(output_dir_reals, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Generate fake samples and save them
    z = torch.randn(num_samples, latent_dim)
    if cuda:
        z = z.cuda()
    with torch.no_grad():
        fake_samples = generator(z)

    # Save fake samples to disk
    logging.info(f"Saving {num_samples} fake samples to disk")
    for i, sample in enumerate(fake_samples):
        # need to save since calcualte_metrics fun expects a path for the input data, not a list of images directly
        save_image(sample, f"temp/fake_samples/{i}.png",normalize=True)

    # Collect 1000 real samples
    real_samples_collected = []
    for i, (real_imgs, _) in enumerate(dataloader):
        real_samples_collected.append(real_imgs)
        if len(torch.cat(real_samples_collected)) >= num_samples:
            break

    # Truncate to exactly 1000 samples
    real_samples = torch.cat(real_samples_collected)[:num_samples]

    # Save real samples to disk
    logging.info(f"Saving {num_samples} real samples to disk")
    for i, sample in enumerate(real_samples):
        # need to save since calcualte_metrics fun expects a path for the input data
        save_image(sample, f"temp/real_samples/{i}.png", normalize=True)

    # Use paths to saved images for torch_fidelity
    logging.info('compute metrics')
    metrics = calculate_metrics(
        input1="temp/real_samples",
        input2="temp/fake_samples",
        cuda=cuda,
        isc=True,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        verbose=False
    )

    return metrics['inception_score_mean'], metrics['frechet_inception_distance'], metrics[
        'kernel_inception_distance_mean']


def save_best_model(generator, is_score, fid_score, kid_score, epoch):
    if is_score > generator.best_is:
        generator.best_is = is_score
        torch.save(generator.state_dict(), f'generator_best_is.pth')
        logging.info(f"Best IS {is_score} from epoch {epoch} saved as generator_best_is.pth")

    if fid_score < generator.best_fid:
        generator.best_fid = fid_score
        torch.save(generator.state_dict(), f'generator_best_fid.pth')
        logging.info(f"Best FID {fid_score} from epoch {epoch} saved as generatr_best_fid.pth")

    if kid_score < generator.best_kid:
        generator.best_kid = kid_score
        torch.save(generator.state_dict(), f'generator_best_kid.pth')
        logging.info(f"Best KID {kid_score} from epoch {epoch} saved as generator_best_kid.pth")


def show_train_hist(D_losses, G_losses, path='Train_hist.png'):
    df = pd.DataFrame({'D_losses': D_losses, 'G_losses': G_losses})
    df.to_csv(Path(path).with_name('Train_hist.csv'), index=False)

    x = range(len(D_losses))

    y1 = D_losses
    y2 = G_losses

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def getDataloader(dataset, batch_size=4, shuffle=True) -> DataLoader:
    root_folder = Path(__file__).parent / '../../data'
    if dataset == "mnist":
        root_folder = root_folder / "mnist"
        os.makedirs(root_folder, exist_ok=True)
        dataset = datasets.MNIST(
            root=root_folder,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
    else:
        raise RuntimeError("Invalid dataset")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
