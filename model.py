from torchvision import models
import torch.nn as nn

def vgg16_binary_classifier(pretrained=True,num_classes = 2):
    """
    Create a VGG-16 model adapted for binary classification.
    
    Parameters:
    pretrained (bool): If True, initializes model with weights pre-trained on ImageNet.
    
    Returns:
    model (torch.nn.Module): VGG-16 model adapted for binary classification.
    """
    # Load a pre-trained VGG-16 model
    model = models.vgg16(pretrained=pretrained)
    
    # Modify the classifier for binary classification
    num_features = model.classifier[6].in_features  # Get the number of inputs for the last layer
    model.classifier[6] = nn.Linear(num_features, num_classes)  # Replace it with a new layer for 2 classes
    
    return model

def alexnet_binary_classifier(pretrained=True,num_classes = 2):
    """
    Create an AlexNet model adapted for binary classification.
    
    Parameters:
    pretrained (bool): If True, initializes model with weights pre-trained on ImageNet.
    
    Returns:
    model (torch.nn.Module): AlexNet model adapted for binary classification.
    """
    # Load a pre-trained AlexNet model
    model = models.alexnet(pretrained=pretrained)
    
    # Modify the classifier for binary classification
    num_features = model.classifier[6].in_features  # Get the number of inputs for the last layer
    model.classifier[6] = nn.Linear(num_features, num_classes)  # Replace it with a new layer for 2 classes
    
    return model
