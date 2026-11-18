import os
import random
import math
import argparse

from models.ConvNet import ConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename='./gradient_matching/out_log.log'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradient Matching"
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./gradient_matching/distilled")
    parser.add_argument("--checkpoint-dir", type=str, default="./gradient_matching/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10) # for CIFAR 10
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256) # number of CIFAR 10 images loaded

    parser.add_argument("--num-iters", type=int, default=1000)  # Outer loop (K)
    parser.add_argument("--inner-steps", type=int, default=1)  # Middle loop (T)
    parser.add_argument("--theta-steps", type=int, default=1)   # Network updates
    parser.add_argument("--image-lr", type=float, default=0.1)  # Synthetic image LR
    parser.add_argument("--theta-lr", type=float, default=0.01) # Network LR

    parser.add_argument("--eval-epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50) # evaluate every n outer iterations
    parser.add_argument("--eval-inits", type=int, default=10) # n networks used for intermediate evaluation
    parser.add_argument("--final-eval-inits", type=int, default=200) # paper uses 200 for the final evaluation
    return parser.parse_args()

# used to normalise cifar images
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
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

# create cnn and return params
def sample_init(args):
    model = ConvNet(
        channel=3,
        num_classes=args.num_classes,
        net_width=128,
        net_depth=3,
        net_act="relu",
        net_norm="instancenorm",
        net_pooling="avgpooling",
        im_size=(32, 32)
    ).to(device)
    return {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }

# load training dataset
def make_class_loaders(args):
    train_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=TRANSFORM
    )

    targets = torch.tensor(train_dataset.targets)
    class_loaders = []
    for c in range(args.num_classes):
        indices = torch.where(targets == c)[0].tolist()
        class_dataset = Subset(train_dataset, indices)
        # retrieves batches of images for a class
        train_loader = DataLoader(
            class_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )
        class_loaders.append(train_loader)

    return class_loaders

# load test dataset
def make_test_loader(args):
    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=TRANSFORM
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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
def evaluate(template_model, images, labels, test_loader, num_inits, args):
    accs = []
    train_accs = []
    images_eval = images.detach()
    for _ in range(num_inits):
        params = sample_init(args)
        optimiser = torch.optim.SGD(
            list(params.values()),
            lr=args.theta_lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        for epoch in range(args.eval_epochs + 1):
            optimiser.zero_grad(set_to_none=True)
            logits = forward(template_model, params, images_eval)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimiser.step()
            if epoch == args.eval_epochs // 2 + 1:
                optimiser = torch.optim.SGD(
                    list(params.values()),
                    lr=args.theta_lr*0.1,
                    momentum=0.9,
                    weight_decay=0.0005
                )
        with torch.no_grad():
            train_preds = forward(template_model, params, images_eval).argmax(dim=1)
            train_acc = ((train_preds == labels).float().mean().item())
        test_acc = test_accuracy(template_model, params, test_loader)
        train_accs.append(train_acc)
        accs.append(test_acc)
    accs_t = torch.tensor(accs)
    logger.info(
        "Evaluation: train acc = %.4f, test acc = %.4f",
        sum(train_accs) / len(train_accs),
        accs_t.mean().item()
    )
    return accs_t.mean().item(), accs_t.std(unbiased=False).item() # return mean and std

# undo image normalisation for saving
def unnormalise(x):
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

# save images and checkpoints
def save_distilled(images, labels, tag, args):
    save_image(
        unnormalise(images.detach()),
        os.path.join(args.output_dir, f"distilled_{tag}.png"),
        nrow=args.num_classes,  # one row per step, one column per class
    )
    torch.save(
        {"images": images.detach().cpu(), "labels": labels.detach().cpu()},
        os.path.join(args.checkpoint_dir, f"distilled_{tag}.pt"),
    )

def matching_loss(s_grads, t_grads):
    loss = torch.tensor(0.0, device=device)
    for s_grad, t_grad in zip(s_grads, t_grads):
        if s_grad.dim() == 1:
            continue
        s_grad = s_grad.reshape(s_grad.shape[0], -1)
        t_grad = t_grad.reshape(t_grad.shape[0], -1)
        dot_product = torch.sum(
            s_grad * t_grad,
            dim=1
        )
        norm_product = (
            torch.norm(s_grad, dim=1)
            * torch.norm(t_grad, dim=1)
        )
        distance = 1 - dot_product / (norm_product + 1e-6)
        loss += distance.sum()
    return loss

def update_network(template_model, params, images, labels, args):
    for _ in range(args.theta_steps):
        logits = forward(template_model, params, images)
        loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(loss, tuple(params.values()))
        params = { # update network params
            name: (p - args.theta_lr * g).detach().requires_grad_(True)
            for (name, p), g in zip(params.items(), grads)
        }
    return params

def main():
    args = parse_args()
    set_seed(args.seed) # intialise random number generators
    os.makedirs(args.output_dir, exist_ok=True) # create saving directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    template_model = ConvNet(
        channel=3,
        num_classes=args.num_classes,
        net_width=128,
        net_depth=3,
        net_act="relu",
        net_norm="instancenorm",
        net_pooling="avgpooling",
        im_size=(32, 32)
    ).to(device) # initialise cnn
    template_model.eval()
    template_model.requires_grad_(False) # disable gradient calculations

    class_loaders = make_class_loaders(args)
    class_batches = [
        batch_generator(loader) # initialise image batch generator
        for loader in class_loaders
    ]
    test_loader = make_test_loader(args)

    images = nn.Parameter(torch.randn(args.num_classes * args.images_per_class, 3, 32, 32, device=device)) # create synthetic iamges
    labels = torch.arange(args.num_classes, device=device).repeat_interleave(args.images_per_class) # intialise labels
    optimiser = torch.optim.SGD( # intialise image optimiser
        [images],
        lr=args.image_lr,
        momentum=0.5
    )

    mean, std = evaluate(template_model, images, labels, test_loader, args.eval_inits, args) # evaluate initial synthetic images
    logger.info(f"iter 0 (before distillation): test acc = {mean:.4f} +/- {std:.4f}")
    images_before = images.detach().clone()

    # train images n times
    for k in range(1, args.num_iters + 1):
        losses = []
        params = sample_init(args)
        for t in range(args.inner_steps):
            class_losses = []
            optimiser.zero_grad()
            for c in range(args.num_classes):
                x_real, y_real = next(class_batches[c]) # get next CIFAR images
                start = c * args.images_per_class
                end = start + args.images_per_class
                x_synthetic = images[start:end]
                y_synthetic = labels[start:end]

                real_loss = F.cross_entropy(forward(template_model, params, x_real), y_real)
                real_grads = torch.autograd.grad(real_loss, tuple(params.values()))

                synthetic_loss = F.cross_entropy(forward(template_model, params, x_synthetic), y_synthetic)
                synthetic_grads = torch.autograd.grad(synthetic_loss, tuple(params.values()), create_graph=True)

                gradient_matching_loss = matching_loss(synthetic_grads, real_grads)
                class_losses.append(gradient_matching_loss.detach())

                gradient_matching_loss.backward(inputs=(images,))

            optimiser.step()
            mean_class_loss = torch.stack(class_losses).mean()
            losses.append(mean_class_loss)

            if t < args.inner_steps - 1:
                params = update_network(template_model, params, images, labels, args)

        mean_loss = torch.stack(losses).mean()
        if k % 5 == 0:
            logger.info(f"iter {k}: loss = {mean_loss.item():.4f}")

        if k % args.eval_every == 0:
            mean, std = evaluate(template_model, images, labels, test_loader, args.eval_inits, args)
            logger.info(f"iter {k}: test acc = {mean:.4f} +/- {std:.4f}")
            save_distilled(images, labels, f"iter{k}", args)

    # measure total image change
    change = (images.detach() - images_before).abs().mean().item()
    logger.info(f"Mean absolute image change: {change:.8f}")

    # perform final evaluation
    mean, std = evaluate(template_model, images, labels, test_loader, args.final_eval_inits, args)
    logger.info(f"FINAL ({args.final_eval_inits} held-out inits): test acc = {mean:.4f} +/- {std:.4f}")
    save_distilled(images, labels, "final", args)
if __name__ == "__main__":
    main()