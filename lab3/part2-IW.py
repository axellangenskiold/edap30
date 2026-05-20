#%% 0. Imports
from torch import nn
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch
import os
from torchvision.utils import save_image

# TODO

#%% 1. Define Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, ndf=64, num_channels=1):
        super().__init__()
        self.net = nn.Sequential(
          
            nn.Conv2d(num_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

           
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

           
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

           
            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)   

#%% 2. Define Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, num_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            
            nn.ConvTranspose2d(latent_dim, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

          
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

           
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

         
            nn.ConvTranspose2d(ngf, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),     
        )

    def forward(self, z):
        
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

# %% 3. Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                       
    transforms.Normalize((0.5,), (0.5,)),        
])


train_ds = FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = FashionMNIST(root="./data", train=False, download=True, transform=transform)
# TODO

#%% 4. Define Weight initialization function
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

#%% 5. Define hyperparameters
# Architecture
LATENT_DIM   = 100      
NGF          = 64          
NDF          = 64         
IMAGE_SIZE   = 32
NUM_CHANNELS = 1        

# Training
BATCH_SIZE   = 128
NUM_EPOCHS   = 15


LR           = 2e-4
BETA1        = 0.5      
BETA2        = 0.999

# Logging
SAMPLE_EVERY = 500         
# TODO

#%% 6. Define dataloaders

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)

# TODO

#%% 7. Test dataloaders by making a image grid

# Grab a batch
xb, yb = next(iter(train_loader))
print(f"Batch shape: {xb.shape}")         
print(f"Range: [{xb.min():.2f}, {xb.max():.2f}]")  


xb_display = xb * 0.5 + 0.5               

# Make an 8×8 grid from the first 64 images
grid = make_grid(xb_display[:64], nrow=8, padding=2, pad_value=1.0)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, img, label in zip(axes.flat, xb_display[:16], yb[:16]):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(train_ds.classes[label], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# TODO

#%% 8. Define model
G = Generator()
D = Discriminator()

# TODO

#%% 9. Initialize optimizers, criterion and utilities
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

G.apply(initialize_weights)
D.apply(initialize_weights)

criterion = nn.MSELoss()                 

fixed_z = torch.randn(64, LATENT_DIM)   

print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

# TODO

#%% 10. Train loop

os.makedirs("samples_mnist", exist_ok=True)

losses = {"step": [], "D": [], "G": [], "D_real_mean": [], "D_fake_mean": []}
global_step = 0

print(f"Starting LSGAN training: {NUM_EPOCHS} epochs, "
      f"{len(train_loader)} batches/epoch")

for epoch in range(1, NUM_EPOCHS + 1):
    G.train()
    D.train()

    running_D, running_G, n_batches = 0.0, 0.0, 0

    for real_imgs, _ in train_loader:
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        
        z = torch.randn(batch_size, LATENT_DIM)
        fake_imgs = G(z)

        opt_D.zero_grad()
        d_real = D(real_imgs)
        d_fake = D(fake_imgs.detach())       
        loss_D = 0.5 * (criterion(d_real, real_labels) +
                        criterion(d_fake, fake_labels))
        loss_D.backward()
        opt_D.step()

       
        opt_G.zero_grad()
        d_fake_for_g = D(fake_imgs)               
        loss_G = criterion(d_fake_for_g, real_labels)   
        loss_G.backward()
        opt_G.step()

       
        running_D += loss_D.item()
        running_G += loss_G.item()
        n_batches += 1

        losses["step"].append(global_step)
        losses["D"].append(loss_D.item())
        losses["G"].append(loss_G.item())
        losses["D_real_mean"].append(d_real.mean().item())
        losses["D_fake_mean"].append(d_fake.mean().item())

     
        if global_step % SAMPLE_EVERY == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z)
            G.train()
            save_image(samples * 0.5 + 0.5,       
                       f"samples_mnist/step_{global_step:06d}.png",
                       nrow=8, padding=2, pad_value=1.0)

        global_step += 1

    # End-of-epoch summary
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
          f"loss_D={running_D/n_batches:.4f} | "
          f"loss_G={running_G/n_batches:.4f} | "
          f"D(real)={losses['D_real_mean'][-1]:.3f} | "
          f"D(fake)={losses['D_fake_mean'][-1]:.3f}")

# Save loss history for later plotting
import json
with open("mnist_losses.json", "w") as f:
    json.dump(losses, f)


# TODO

#%% 11. Save generator and discriminator weights

torch.save(G.state_dict(), "G_mnist_final.pt")
torch.save(D.state_dict(), "D_mnist_final.pt")
print("Saved G_final.pt and D_final.pt")

# TODO

#%% 12. Implement FID evaluation
from fid import fid

# TODO: Load your saved generator

# Load saved generator
G = Generator(latent_dim=LATENT_DIM, ngf=NGF, num_channels=NUM_CHANNELS)
G.load_state_dict(torch.load("G_final.pt"))
G.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(N_samples, gen, latent_dim, batch_size=64):
    """Yield N_samples generated images as 3-channel tensors."""
    gen.eval()
    yielded = 0
    with torch.no_grad():
        while yielded < N_samples:
            current = min(batch_size, N_samples - yielded)
            z = torch.randn(current, latent_dim)
            batch = gen(z)                             
            yield batch.expand(-1, 3, -1, -1)          
            yielded += current

def generate_real_samples(N_samples, train_dl):
    """Yield batches of real images as 3-channel tensors, capped at N_samples."""
    yielded = 0
    for imgs, _ in train_dl:
        if yielded >= N_samples:
            break
        
        remaining = N_samples - yielded
        if imgs.size(0) > remaining:
            imgs = imgs[:remaining]
        yield imgs.expand(-1, 3, -1, -1)
        yielded += imgs.size(0)

    
# TODO: Implement fid using the generators above

# %%


# Run FID
N_SAMPLES = 1000

fid_value = fid(
    generate_real_samples(N_SAMPLES, train_loader),
    generate_samples(N_SAMPLES, G, LATENT_DIM, batch_size=64),
    device,
)
print(f"FID: {fid_value:.4f}")


#%%

G.eval()
with torch.no_grad():
    z = torch.randn(32, LATENT_DIM)
    samples = G(z)                              

samples_display = samples * 0.5 + 0.5            


# Option 2: save to file
save_image(samples_display, "generated_grid_pictures.png",
           nrow=8, padding=2, pad_value=1.0)


# %%

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                       
    transforms.Normalize((0.5,), (0.5,)),        
])


train_ds = MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = MNIST(root="./data", train=False, download=True, transform=transform)


# %%
