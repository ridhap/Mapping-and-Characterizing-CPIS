import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.ensemble.tests.test_weight_boosting import y_class
plt.switch_backend('agg')

def save_checkpoint(state, filename="my_check.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=5,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_ds, val_ds, train_loader, val_loader

def check_accuracy(loader, model, device, loss_fn):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    diceScore = []
    criterion = nn.BCEWithLogitsLoss()      

    # Initialize counters for TP, TN, FP, FN
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    y_pos_values = 0
    y_neg_values = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y,_ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            loss = criterion(preds, y)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            
            total_loss += loss.item()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # Calculate TP, TN, FP, FN
            true_positives += ((preds == 1) & (y == 1)).sum()
            true_negatives += ((preds == 0) & (y == 0)).sum()
            false_positives += ((preds == 1) & (y == 0)).sum()
            false_negatives += ((preds == 0) & (y == 1)).sum()

            y_pos_values += (y == 1).sum()
            y_neg_values += (y == 0).sum()
            # Append true and predicted labels
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    diceScore = diceScore.append(dice_score/len(loader))
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    
    # Calculate percentages
    tp_percentage = (true_positives / num_pixels) * 100
    tn_percentage = (true_negatives / num_pixels) * 100
    fp_percentage = (false_positives / num_pixels) * 100
    fn_percentage = (false_negatives / num_pixels) * 100

    # Print percentages
    print(f"True Positives Percentage: {tp_percentage:.2f}%")
    print(f"True Negatives Percentage: {tn_percentage:.2f}%")
    print(f"False Positives Percentage: {fp_percentage:.2f}%")
    print(f"False Negatives Percentage: {fn_percentage:.2f}%")
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    avg_loss = total_loss / len(loader)
    print(f"Average Loss: {avg_loss}")
    print(f"F1 Score: {f1_score}")

    model.train()

    return avg_loss  # Return average validation loss for plotting


# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()

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




def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    # Turn off interactive mode
    plt.ioff()
    for idx, (x, y,_) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Save each prediction and corresponding ground truth individually
        for i in range(len(preds)):
            # Plot predicted image and original image side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Display predicted image
            axs[0].imshow(torchvision.transforms.ToPILImage()(preds[i]))
            axs[0].set_title(f"Predicted Image {idx}_{i}")

            # Display original image
            axs[1].imshow(torchvision.transforms.ToPILImage()(y[i]))
            axs[1].set_title(f"Original Image {idx}_{i}")

            # Save the combined plot
            plt.savefig(f"{folder}/pred_{idx}_{i}.png")
            plt.close(fig)

            # Show the plot (optional)
            # plt.show()
    model.train()
