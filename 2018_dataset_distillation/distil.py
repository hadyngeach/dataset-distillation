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
NUM_CLASSES = 10
BATCH_SIZE = 1024
STEPS = 10
EPOCHS = 3
INITS_PER_ITER = 4
INIT_LR = 0.001
LR = 0.01
NUM_ITERS = 1000

EVAL_EVERY = 50        # evaluate every N outer iterations
EVAL_INITS = 10          # held-out inits for quick evals
FINAL_EVAL_INITS = 200   # paper uses 200 for the final number
BETAS = (0.5, 0.999)
DECAY_EPOCHS = 40
DECAY_FACTOR = 0.5


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        ),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model taken from paper author's github
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

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sample_init():
    model = AlexCifarNet().to(device)
    model.apply(init_weights)
    return {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }

def make_train_loader():
    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=TRANSFORM
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader

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

def forward(model, params, x):
    return functional_call(model, params, (x,))

def batch_generator(loader):
    while True:
        for images, labels in loader:
            yield images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

def report_grad(name, g):
    if g is None:
        print(f"{name}.grad is None")
        return
    print(f"{name}.grad: abs sum = {g.abs().sum().item():.6f}, "
          f"finite = {torch.isfinite(g).all().item()}")

def unroll(template_model, params, images, labels, raw_lrs):
    for e in range(EPOCHS):
        for s in range(STEPS):
            x = images[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]
            y = labels[s * NUM_CLASSES:(s + 1) * NUM_CLASSES]

            logits = forward(template_model, params, x)
            loss = F.cross_entropy(logits, y)
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)

            lr = F.softplus(raw_lrs[e, s])
            params = {
                name: p - lr * g
                for (name, p), g in zip(params.items(), grads)
            }
    return params

def unroll_eval(template_model, params, images, labels, raw_lrs):
    """Same updates as unroll, but with no graph back to images or lrs."""
    images = images.detach()
    lrs = F.softplus(raw_lrs.detach())
    params = {k: v.detach().requires_grad_(True) for k, v in params.items()}

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


def test_accuracy(template_model, params, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = forward(template_model, params, x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


def evaluate(template_model, images, labels, raw_lrs, test_loader, num_inits):
    """Train num_inits fresh networks on the distilled data; return mean/std test accuracy."""
    accs = []
    for _ in range(num_inits):
        params = sample_init()
        final = unroll_eval(template_model, params, images, labels, raw_lrs)
        accs.append(test_accuracy(template_model, final, test_loader))
    accs_t = torch.tensor(accs)
    return accs_t.mean().item(), accs_t.std(unbiased=False).item()


def unnormalise(x):
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


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
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    template_model = AlexCifarNet().to(device)
    template_model.eval()
    template_model.requires_grad_(False)

    train_loader = make_train_loader()
    test_loader = make_test_loader()
    real_batches = batch_generator(train_loader)

    images = torch.randn(STEPS * NUM_CLASSES, 3, 32, 32, device=device, requires_grad=True)
    labels = torch.arange(NUM_CLASSES, device=device).repeat(STEPS)
    raw_lrs = torch.full(
        (EPOCHS, STEPS),
        math.log(math.exp(INIT_LR) - 1),
        device=device,
        requires_grad=True
    )
    optimiser = torch.optim.Adam(
        [images, raw_lrs],
        lr=LR,
        betas=BETAS
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, 
        step_size=DECAY_EPOCHS * len(train_loader),
        gamma=DECAY_FACTOR
    )
    mean, std = evaluate(template_model, images, labels, raw_lrs, test_loader, EVAL_INITS)
    print(f"iter 0 (before distillation): test acc = {mean:.4f} +/- {std:.4f}")
    images_before = images.detach().clone()

    for i in range(1, NUM_ITERS + 1):
        x_real, y_real = next(real_batches)
        optimiser.zero_grad() 
        losses = []
        for _ in range(INITS_PER_ITER):
            params = sample_init()
            final = unroll(template_model, params, images, labels, raw_lrs)
            outer_loss = F.cross_entropy(forward(template_model, final, x_real), y_real)
            losses.append(outer_loss)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimiser.step()
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

    change = (images.detach() - images_before).abs().mean().item()
    print(f"Mean absolute image change: {change:.8f}")

    mean, std = evaluate(template_model, images, labels, raw_lrs, test_loader, FINAL_EVAL_INITS)
    print(f"FINAL ({FINAL_EVAL_INITS} held-out inits): test acc = {mean:.4f} +/- {std:.4f}")
    save_distilled(images, raw_lrs, "final")
if __name__ == "__main__":
    main()