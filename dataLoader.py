import os
import numpy as np
import torch
import cv2
import os
import numpy as np
import torch
from PIL import Image
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load images from Normalimg folder
        normal_img_path = os.path.join(root_dir, 'Normalimg')
        for img_name in os.listdir(normal_img_path):
            self.image_paths.append(os.path.join(normal_img_path, img_name))
            self.labels.append(0)  # Label 0 for Normalimg

        # Load images from Lesionsimg folder
        lesions_img_path = os.path.join(root_dir, 'Lesionsimg')
        for img_name in os.listdir(lesions_img_path):
            self.image_paths.append(os.path.join(lesions_img_path, img_name))
            self.labels.append(1)  # Label 1 for Lesionsimg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is RGB

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label