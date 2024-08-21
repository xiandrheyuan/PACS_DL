import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from dataLoader import Dataset
from model import MaskRcnnResnet50Fpn
from util_ import utils, engine
import copy
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np
from model import alexnet_binary_classifier,vgg16_binary_classifier
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

def main():
    LR = 5E-4
    BATCH_SIZE = 16
    MAX_EPOCH = 30
    K_FOLDS = 10
    WEIGHT_DECAY = 5e-5
    dataset = Dataset('datasets/', 'datasets/annotations.json', transform=train_transforms)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = [sample[1] for sample in dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn, num_workers=8)

        model = alexnet_binary_classifier(num_classes=2).to(device)
        # model = vgg16_binary_classifier(num_classes=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR,weight_decay=WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_val_iou = 0

        for epoch in range(MAX_EPOCH):
            engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            val_iou = engine.evaluate(model, val_loader, device=device)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = f'net/Fold_{fold}_Epoch_{epoch}_IOU_{val_iou:.4f}.pth'
                torch.save(best_model_wts, model_path)
                print(f'New best model for fold {fold} saved to {model_path}')

        print(f'Best validation IOU for fold {fold}: {best_val_iou:.4f}')

if __name__ == "__main__":
    main()
