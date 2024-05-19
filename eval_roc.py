import cv2
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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

from watershed import get_instances

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
import numpy as np


# Plotting ROC Curve
def plot_roc_curve(y_true, y_pred_scores, unet_pred, watershed_threshold):
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
    plt.savefig(f'roc_curve_{unet_pred}_{watershed_threshold}.png')

# Plotting Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred_scores, unet_pred, watershed_threshold):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    average_precision = average_precision_score(y_true, y_pred_scores)

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()
    plt.savefig(f'precision_recall_curve_{unet_pred}_{watershed_threshold}.png')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
PIN_MEMORY = True
NUM_WORKERS = 5
IMAGE_HEIGHT = int(1003 * 0.32)   # 1003 originally
IMAGE_WIDTH = int(1546 * 0.32)   # 1546 originally
TRAIN_IMG_DIR = "/home/kashis/Desktop/Capstone/ridha_Unet/Dataset/train_images/"
TRAIN_MASK_DIR = "/home/kashis/Desktop/Capstone/ridha_Unet/Dataset/train_masks"
VAL_IMG_DIR = "/home/kashis/Desktop/Capstone/ridha_Unet/Dataset/val_images"
VAL_MASK_DIR = "/home/kashis/Desktop/Capstone/ridha_Unet/Dataset/val_masks"

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
checkpoint = torch.load("/home/kashis/Desktop/Capstone/pipeline/pretrained_weights/my_check_may14.pth.tar")

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

def eval_for_roc(unet_pred, watershed_threshold):
    with torch.no_grad():

        for x, y,_ in tqdm(val_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            preds = model(x)
            loss = criterion(preds, y)
            preds = torch.sigmoid(preds)
            preds = (preds > unet_pred).float()
            

            pred = np.squeeze(preds, (0,1)).cpu()
            pred_np = pred.unsqueeze(-1).cpu().detach().numpy().astype(np.uint8)
            # INSTANCE SEGMENTATION
            rgb_np_pred = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2RGB)
            segments_in_img = get_instances(rgb_np_pred, watershed_threshold)
            combined_segments = torch.from_numpy(np.sum(segments_in_img, axis=0))

            scores = torch.sigmoid(combined_segments).flatten()
            y_scores.extend(scores)

            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(combined_segments.flatten())


    plot_roc_curve(y_true, y_pred, unet_pred, watershed_threshold)
    plot_precision_recall_curve(y_true, y_scores, unet_pred, watershed_threshold)



unet_preds = [0.25, 0.5, 0.75]
watershed_thresholds = [0.005, 0.01, 0.05, 0.1, 0.5]

for unet_pred in unet_preds:
    for watershed_thresh in watershed_thresholds:
        eval_for_roc(unet_pred, watershed_thresh)


