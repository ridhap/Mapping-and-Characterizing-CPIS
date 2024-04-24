import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.ensemble.tests.test_weight_boosting import y_class
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from matplotlib import pyplot as plt
from evaluate import check_accuracy
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)
import os
from PIL import Image
from torch.utils.data import Dataset
import sys
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import sklearn.metrics 
from sklearn.metrics import jaccard_score, f1_score, precision_recall_curve, average_precision_score, accuracy_score
import seaborn as sns
import tabulate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 5
IMAGE_HEIGHT = int(1003 * 0.32)   # 1003 originally
IMAGE_WIDTH = int(1546 * 0.32)   # 1546 originally
TRAIN_IMG_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/train/"
TRAIN_MASK_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/trainannot/"
VAL_IMG_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/val/"
VAL_MASK_DIR = "/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/data/valannot/"
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

# Model
checkpoint = torch.load("/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/" + 'my_checkpoint.pth.tar')

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])

folder = "ValPredictions"
os.makedirs(folder, exist_ok=True)
model.eval()
total_loss = 0
num_correct = 0
num_pixels = 0
dice_score = 0
diceScore = []
criterion = nn.BCEWithLogitsLoss()      
TP = 0
FP = 0
FN = 0
TN = 0
y_true = []
y_pred = []
y_scores = []

overlap_counter = {
    0.25: 0,
    0.5: 0,
    0.75: 0,
    0.95: 0
}


with torch.no_grad():

    for overlap_key in overlap_counter.keys():
        for x, y,_ in tqdm(val_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            preds = model(x)
            loss = criterion(preds, y)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            scores = torch.sigmoid(preds).cpu().numpy().flatten()  # These are the scores we need
            y_scores.extend(scores)
            total_loss += loss.item()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            for i in range(len(preds)):
                total_gt = (y[i] == 1).sum().cpu().numpy().flatten()[0]

                pixels_matched = ((preds[i] == 1) & (y[i] == 1)).sum().cpu().numpy().flatten()[0]
                _x_percent_total_ones_in_gt = overlap_key * total_gt

                if pixels_matched >= _x_percent_total_ones_in_gt:
                    overlap_counter[overlap_key] += 1


            TP += ((preds == 1) & (y == 1)).sum()
            FP += ((preds == 1) & (y == 0)).sum()
            FN += ((preds == 0) & (y == 1)).sum()
            TN += ((preds == 0) & (y == 0)).sum()
            # for i in range(len(preds)): 
            #     if y[i]==preds[i]==1:
            #         TP += 1
            #     if preds[i]==1 and y[i]!=preds[i]:
            #         FP += 1
            #     if y[i]==preds[i]==0:
            #         TN += 1
            #     if preds[i]==0 and y[i]!=preds[i]:
            #         FN += 1
            # y_true.extend(y.cpu())
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())


diceScore = diceScore.append(dice_score / len(val_loader))
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
# Now you can use these scores to calculate the precision and recall
precision, recall, _ = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)
# print(f"Sensitivity (Recall): {sensitivity}")
# print(f"Specificity: {specificity}")
# print(f"PPV (Precision): {PPV}")
# print(f"NPV: {NPV}")
# print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
# print(f"Dice score: {dice_score/len(val_loader)}")
# Calculate additional metrics
accuracy = num_correct / num_pixels
dice = dice_score / len(val_loader)
IOU = jaccard_score(y_true, y_pred)
f1_score = f1_score(y_true, y_pred)
metrics = {
    'TP': TP.item(),
    'FP': FP.item(),
    'FN': FN.item(),
    'TN': TN.item(),
    'Sensitivity(TPR)': sensitivity.item(),
    'Specificity(TNR)': specificity.item(),
    'PPV': PPV.item(),
    'NPV': NPV.item(),
    'Accuracy': accuracy.item(),
    'Dice': dice.item(),
    'IOU': IOU.item(),
    'Average Precision': average_precision,
    'F1 Score': f1_score
}

file = open("metrics.txt", "w")
file.write(tabulate.tabulate(metrics.items(), headers=['Metric', 'Value']))
file.close()

print(tabulate.tabulate(metrics.items(), headers=['Metric', 'Value']))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
import numpy as np
# Plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True, cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Plotting ROC Curve
def plot_roc_curve(y_true, y_pred_scores):
    fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc_curve.png')

# Plotting Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    average_precision = average_precision_score(y_true, y_pred_scores)

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()
    plt.savefig('precision_recall_curve.png')
# Convert y_true and y_pred to a flat list/array if needed
# y_true = np.array(y_true).flatten()
# y_pred = np.array(y_pred).flatten()

# Assuming y_pred are binary predictions for the confusion matrix.
# If you have prediction scores/probabilities, you'll need to adjust the threshold to get binary predictions.
plot_confusion_matrix(y_true, y_pred)

# Assuming you have prediction scores in y_pred_scores for ROC and Precision-Recall curves.
# If y_pred is binary, you'll need the scores instead of binary predictions to plot ROC and Precision-Recall curves.
# y_pred_scores = model.predict_proba(x)[:,1] # This is just an example, replace with your own method.
plot_roc_curve(y_true, y_scores)
plot_precision_recall_curve(y_true, y_scores)

# Tabulate the metrics
print(tabulate.tabulate(metrics.items(), headers=['Metric', 'Value']))

