import torch
from torchvision import datasets, transforms

from train_cifar10 import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
model = CNN().to(device)
model.load_state_dict(
    torch.load(
        "checkpoints/cifar10_cnn.pth",
        map_location=device,
        weights_only=True
    )
)


model.eval()
image, label = test_dataset[0]
image = image.unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(image)
    probabilities = torch.softmax(logits, dim=1)
    
prediction = probabilities.argmax(dim=1).item()

print("Predicted class:", classes[int(prediction)])
print("Actual class:", classes[label])
print(f"Confidence: {probabilities[0, int(prediction)].item():.2%}")