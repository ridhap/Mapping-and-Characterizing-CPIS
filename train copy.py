import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from matplotlib import pyplot as plt
from evaluate import check_accuracy
plt.switch_backend('agg')
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)

import sys

# sys.stdout = open('/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/trainLog2.txt', 'w')
# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 250
NUM_WORKERS = 5
IMAGE_HEIGHT = int(1003 * 0.32)   # 1003 originally
IMAGE_WIDTH = int(1546 * 0.32)   # 1546 originally
# IMAGE_HEIGHT = 320
# IMAGE_WIDTH = 480
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/train/"
TRAIN_MASK_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/trainannot/"
VAL_IMG_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/val/"
VAL_MASK_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/valannot/"
 
# def train_fn(loader, model, optimizer, loss_fn, scaler):
#     loop = tqdm(loader)
#     total_loss = 0
#     criterion = nn.BCEWithLogitsLoss()      

#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)

#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         # backward
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update tqdm loop
#         loop.set_postfix(loss=loss.item())

def visualize_samples(loader, num_samples=5, device="cuda"):
    # Get a few samples from the loader
    for batch_idx, (data, targets, file_names) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Display the original images
        for i in range(num_samples):
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i].permute(1, 2, 0).cpu().numpy())
            plt.title(f"Sample {i + 1}\nOriginal\n{file_names[i]}")
            plt.axis("off")

        # Display the corresponding masks
        for i in range(num_samples):
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(targets[i].cpu().numpy(), cmap="gray")
            plt.title(f"Sample {i + 1}\nMask\n{file_names[i]}")
            plt.axis("off")

        plt.show()
        break  # Display only the first batch



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-04, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()


    train_ds, val_ds, train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    # visualize_samples(train_loader)
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler() 

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader)
        total_loss = 0

        for batch_idx, (data, targets,_) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # save model
        # if epoch % 10 == 0:
        #     checkpoint = {
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint)

        # Validation phase
        val_loss = check_accuracy(val_loader, model, DEVICE, loss_fn)  # Modify check_accuracy to return loss
        val_losses.append(val_loss)
        # print some examples to a folder
        # if epoch % 5 == 0:
        #     save_predictions_as_imgs(
        #             val_loader, model, folder="saved_image3/", device=DEVICE)
        f = open("Train_Val_Loss.txt", "a")
        f.write(f"Epoch {epoch} Train Loss: {train_loss} Val Loss: {val_loss}\n")
        f.close()
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig('/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/lossPlot_f1.png')
    plt.show()

if __name__ == "__main__":
    main()
    
# close system buffer
    sys.stdout.close()
