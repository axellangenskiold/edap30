# Based on https://github.com/mseitzer/pytorch-fid/tree/master
# Modified to use built-in models and be standalone

#%% 0. Imports
import numpy as np
import torch
from scipy import linalg

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

#%% 1. Model and preprocessor

# Function to preprocess the image
def create_fid_tranforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class InceptionV3Activations(nn.Module):
    def __init__(self):
        super(InceptionV3Activations, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity(2048) # remove final head and replace it with an identity layer
    def forward(self, x):
        return self.inception(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def fid(real_samples, fake_samples, device):
    """Calculation of the statistics used by the FID.
    Params:
    -- real_samples   : dataloader or iterable producing batches of images
    -- fake_samples   : iterable producing batches of generated samples (new set each time an iterator is requested) 
                        expected to produce the same number of of samples as in the real one.
    -- device      : Device to run calculations

    Returns:
    -- fid   : frechet distance score
    """
    
    # Load the model
    model = InceptionV3Activations()
    model = model.to(device)
    model.eval()

    transform = create_fid_tranforms()

    real_activations = []
    fake_activations = []

    with torch.no_grad():
        for real in tqdm(real_samples,desc="Computing activations from real samples"):
            real = real.to(device)
            if real.shape[1] == 1: # grayscale -> expand to color
                real = real.expand(real.size(0), 3, real.size(2), real.size(3))

            real_act = model(transform(real))

            real_pred = real_act.cpu().numpy()
            real_activations.append(real_pred)

        for fake in tqdm(fake_samples,desc="Computing activations from fake samples"):
            fake = fake.to(device)
            if fake.shape[1] == 1: # grayscale -> expand to color
                fake = fake.expand(fake.size(0), 3, fake.size(2), fake.size(3))

            fake_act = model(transform(fake))
            fake_pred = fake_act.cpu().numpy()

            fake_activations.append(fake_pred)

    real_arr = np.concat(real_activations)
    fake_arr = np.concat(fake_activations)

    assert real_arr.shape == fake_arr.shape, "real shape %s does not match fake shape %s" % (real_arr.shape, fake_arr.shape)    

    mu1 = np.mean(real_arr, axis=0)
    sigma1 = np.cov(real_arr, rowvar=False)

    mu2 = np.mean(fake_arr, axis=0)
    sigma2 = np.cov(fake_arr, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

# %% 2. Primitive test
def fake_generator(B,N,C,W,H,seed=1667):
    gen = torch.manual_seed(seed)
    for _ in range(B):
        yield torch.randn(N,C,W,H, generator=gen)

def main():
    x = fake_generator(8, 64, 1, 32,32, seed=42) # the first value indicates number of batches
    y = fake_generator(8, 64, 1, 32,32, seed=1667)
    print(fid(x, y, "cpu")) # change to your device here

if __name__ == "__main":
    main()
