import os
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

SEED = 42
DATA_ROOT = "./data"
OUTPUT_DIR = "./distilled"
CHECKPOINT_DIR = "./checkpoints"
NUM_CLASSES = 10 # for CIFAR 10
BATCH_SIZE = 1024 # number of CIFAR 10 images loaded
STEPS = 10
EPOCHS = 3
INITS_PER_ITER = 4 # train n randomly initialised cnns on synthetic data
INIT_LR = 0.001 # learning rate for training cnns on synthetic data
LR = 0.01 # learning rate used by adam to change synthetic images
NUM_ITERS = 1000

EVAL_EVERY = 50        # evaluate every n outer iterations
EVAL_INITS = 10          # n networks used for intermediate evaluation
FINAL_EVAL_INITS = 200   # paper uses 200 for the final evaluation

# settings sourced from paper github
BETAS = (0.5, 0.999) # used by adam 
# learning rate is halved every n epochs of image optimisation
DECAY_EPOCHS = 40 
DECAY_FACTOR = 0.5

# used to normalise cifar images
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        ),
    ])

# uses local gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model taken from paper's github: https://github.com/SsnL/dataset-distillation
class AlexCifarNet(nn.Module):
    def __init__(self):
        super(AlexCifarNet, self).__init__()
        nc = 3
        self.features = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x

# use Xavier initialisation to initialise cnn weights
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias) # no bias

# create cnn and return params
def sample_init():
    model = AlexCifarNet().to(device)
    model.apply(init_weights)
    return {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }

# load training dataset
def make_train_loader():
    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=TRANSFORM
    )

    # retrieves batches of images
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader

# load test dataset
def make_test_loader():
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=TRANSFORM
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    return test_loader

# runs model with supplied params
def forward(model, params, x):
    return functional_call(model, params, (x,))

# infinite generator of image batches
def batch_generator(loader):
    while True:
        for images, labels in loader:
            yield images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

# checks gradient existence
def report_grad(name, g):
    if g is None:
        print(f"{name}.grad is None")
        return
    print(f"{name}.grad: abs sum = {g.abs().sum().item():.6f}, "
          f"finite = {torch.isfinite(g).all().item()}")

# simulates training neural network on synthetic data
def unroll(template_model, params, images, labels, raw_lrs):
    for e in range(EPOCHS):
        for s in range(STEPS):
            x = images[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]
            y = labels[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]

            logits = forward(template_model, params, x) # passes synthetic images through cnn
            loss = F.cross_entropy(logits, y) # calculates cross entropy loss
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True) # calculates gradient of loss

            lr = F.softplus(raw_lrs[e, s]) # use softplus to optimise learning rate
            params = { # update network params
                name: p - lr * g
                for (name, p), g in zip(params.items(), grads)
            }
    return params

# simulates training neural network on synthetic data (for evaluation)
def unroll_eval(template_model, params, images, labels, raw_lrs):
    images = images.detach() # don't calculate gradients
    lrs = F.softplus(raw_lrs.detach())
    params = {k: v.detach().requires_grad_(True) for k, v in params.items()} # detach params

    for e in range(EPOCHS):
        for s in range(STEPS):
            x = images[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]
            y = labels[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]

            logits = forward(template_model, params, x)
            loss = F.cross_entropy(logits, y)
            grads = torch.autograd.grad(loss, list(params.values()))

            params = {
                name: (p - lrs[e, s] * g).detach().requires_grad_(True)
                for (name, p), g in zip(params.items(), grads)
            }
    return {k: v.detach() for k, v in params.items()}

# calculate accuracy
def test_accuracy(template_model, params, test_loader):
    correct = 0
    total = 0
    with torch.no_grad(): # stops gradient calculations to reduce memory usage
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = forward(template_model, params, x).argmax(dim=1) # get indices of predicted classes
            correct += (preds == y).sum().item() # count correct predictions
            total += y.numel() # add to total images predicted
    return correct / total

# evaluates synthetic data performance on randomly intialised networks
def evaluate(template_model, images, labels, raw_lrs, test_loader, num_inits):
    accs = []
    for _ in range(num_inits):
        params = sample_init()
        final = unroll_eval(template_model, params, images, labels, raw_lrs) # create and train cnn on images
        accs.append(test_accuracy(template_model, final, test_loader)) # record accuracy
    accs_t = torch.tensor(accs)
    return accs_t.mean().item(), accs_t.std(unbiased=False).item() # return mean and std

# undo image normalisation for saving
def unnormalise(x):
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

# save images and checkpoints
def save_distilled(images, raw_lrs, tag):
    save_image(
        unnormalise(images.detach()),
        os.path.join(OUTPUT_DIR, f"distilled_{tag}.png"),
        nrow=NUM_CLASSES,  # one row per step, one column per class
    )
    torch.save(
        {"images": images.detach().cpu(), "raw_lrs": raw_lrs.detach().cpu()},
        os.path.join(CHECKPOINT_DIR, f"distilled_{tag}.pt"),
    )

def main():
    set_seed(SEED) # intialise random number generators
    os.makedirs(OUTPUT_DIR, exist_ok=True) # create saving directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    template_model = AlexCifarNet().to(device) # initialise cnn
    template_model.eval()
    template_model.requires_grad_(False) # disable gradient calculations

    train_loader = make_train_loader()
    test_loader = make_test_loader()
    real_batches = batch_generator(train_loader) # initialise image batch generator

    images = torch.randn(STEPS * NUM_CLASSES, 3, 32, 32, device=device, requires_grad=True) # create synthetic iamges
    labels = torch.arange(NUM_CLASSES, device=device).repeat(STEPS) # intialise labels
    raw_lrs = torch.full( # initalise learning rates
        (EPOCHS, STEPS),
        math.log(math.exp(INIT_LR) - 1),
        device=device,
        requires_grad=True
    )
    optimiser = torch.optim.Adam( # intialise image optimiser
        [images, raw_lrs],
        lr=LR,
        betas=BETAS
    )
    scheduler = torch.optim.lr_scheduler.StepLR( # intialise scheduler that halves learning rate
        optimiser, 
        step_size=DECAY_EPOCHS * len(train_loader),
        gamma=DECAY_FACTOR
    )
    mean, std = evaluate(template_model, images, labels, raw_lrs, test_loader, EVAL_INITS) # evaluate initial synthetic images
    print(f"iter 0 (before distillation): test acc = {mean:.4f} +/- {std:.4f}")
    images_before = images.detach().clone()

    # train images n times
    for i in range(1, NUM_ITERS + 1):
        x_real, y_real = next(real_batches) # get next CIFAR images
        optimiser.zero_grad() # clear gradients
        losses = []
        # train INITS_PER_ITER randomly initialised networks
        for _ in range(INITS_PER_ITER):
            params = sample_init()
            final = unroll(template_model, params, images, labels, raw_lrs)
            outer_loss = F.cross_entropy(forward(template_model, final, x_real), y_real)
            losses.append(outer_loss)
        total_loss = torch.stack(losses).mean() # calculate average losss of cnns
        total_loss.backward() # calculate loss gradients by backpropagation
        optimiser.step() # update images and lrs
        scheduler.step()
        if i % 5 == 0:
            lrs = F.softplus(raw_lrs.detach())
            print(
                f"iter {i}: loss = {total_loss.item():.4f}, "
                f"lr range = [{lrs.min().item():.4f}, {lrs.max().item():.4f}]"
            )

        if i % EVAL_EVERY == 0:
            mean, std = evaluate(template_model, images, labels, raw_lrs, test_loader, EVAL_INITS)
            print(f"iter {i}: test acc = {mean:.4f} +/- {std:.4f}")
            save_distilled(images, raw_lrs, f"iter{i}")

    # measure total image change
    change = (images.detach() - images_before).abs().mean().item()
    print(f"Mean absolute image change: {change:.8f}")

    # perform final evaluation
    mean, std = evaluate(template_model, images, labels, raw_lrs, test_loader, FINAL_EVAL_INITS)
    print(f"FINAL ({FINAL_EVAL_INITS} held-out inits): test acc = {mean:.4f} +/- {std:.4f}")
    save_distilled(images, raw_lrs, "final")
if __name__ == "__main__":
    main()