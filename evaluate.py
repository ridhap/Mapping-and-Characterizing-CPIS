import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_metrics(metrics):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Confusion matrix
    cm = [[metrics['TP'], metrics['FP']], [metrics['FN'], metrics['TN']]]
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=cm[i][j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('/home/ridha/Documents/Mapping-and-Characterizing-Center-Pivot-Irrigation-in-the-US/confusionMatrix_f1.png')
    plt.show()

    # Metrics
    print("Sensitivity (Recall):", metrics['Sensitivity'])
    print("Specificity:", metrics['Specificity'])
    print("PPV (Precision):", metrics['PPV'])
    print("NPV:", metrics['NPV'])

def check_accuracy(loader, model, device, loss_fn):
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
            TP += ((preds == 1) & (y == 1)).sum()
            FP += ((preds == 1) & (y == 0)).sum()
            FN += ((preds == 0) & (y == 1)).sum()
            TN += ((preds == 0) & (y == 0)).sum()
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    diceScore = diceScore.append(dice_score / len(loader))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)

    # print(f"Sensitivity (Recall): {sensitivity}")
    # print(f"Specificity: {specificity}")
    # print(f"PPV (Precision): {PPV}")
    # print(f"NPV: {NPV}")
    # print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    # Calculate additional metrics
    accuracy = num_correct / num_pixels
    dice = dice_score / len(loader)
    avg_loss = total_loss / len(loader)

    model.train()
    metrics = {
        'TP': TP.item(),
        'FP': FP.item(),
        'FN': FN.item(),
        'TN': TN.item(),
        'Sensitivity': sensitivity.item(),
        'Specificity': specificity.item(),
        'PPV': PPV.item(),
        'NPV': NPV.item(),
        'Accuracy': accuracy.item(),
        'Dice': dice.item()
    }
    plot_metrics(metrics)

    return avg_loss  # Return average validation loss for plotting
