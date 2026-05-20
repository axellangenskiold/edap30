#%% 0. Imports

# TODO

#%% 1. Define Discriminator Model
class Discriminator(nn.Module):
    # TODO
    pass

#%% 2. Define Generator Model
class Generator(nn.Module):
    # TODO
    pass

# %% 3. Prepare the dataset

# TODO

#%% 4. Define Weight initialization function
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

#%% 5. Define hyperparameters

# TODO

#%% 6. Define dataloaders

# TODO

#%% 7. Test dataloaders by making a image grid

# TODO

#%% 8. Define model

# TODO

#%% 9. Initialize optimizers, criterion and utilities

# TODO

#%% 10. Train loop

# TODO

#%% 11. Save generator and discriminator weights

# TODO

#%% 12. Implement FID evaluation
from fid import fid

# TODO: Load your saved generator

def generate_samples(N_samples, gen, latent_dim, batch_size=64):
    # TODO: create a python generator using yield for every sample
    pass

def generate_real_samples(train_dl):
    # TODO: create python generator using yield for every real sample
    pass
    
# TODO: Implement fid using the generators above
