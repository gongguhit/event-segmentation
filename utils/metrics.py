import torch
import torch.nn.functional as F
import numpy as np

def compute_iou(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) for segmentation.
    
    Args:
        pred: Predicted segmentation (B, C, H, W) tensor
        target: Target segmentation (B, H, W) tensor
        smooth: Small constant to avoid division by zero
        
    Returns:
        IoU score
    """
    # Convert predictions to class indices
    pred = torch.argmax(pred, dim=1)
    
    # Convert to one-hot encoding
    num_classes = pred.max().item() + 1
    pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate intersection and union
    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
    union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
    
    # Calculate IoU for each class
    iou = (intersection + smooth) / (union + smooth)
    
    # Average IoU across all classes (excluding background if needed)
    mean_iou = iou[:, 1:].mean() if num_classes > 1 else iou.mean()
    
    return mean_iou.item()

def compute_accuracy(pred, target):
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predicted segmentation (B, C, H, W) tensor
        target: Target segmentation (B, H, W) tensor
        
    Returns:
        Accuracy score
    """
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

def compute_precision_recall(pred, target):
    """
    Compute precision and recall.
    
    Args:
        pred: Predicted segmentation (B, C, H, W) tensor
        target: Target segmentation (B, H, W) tensor
        
    Returns:
        Precision and recall scores
    """
    pred = torch.argmax(pred, dim=1)
    
    # For binary segmentation (class 1 is foreground)
    if pred.max().item() <= 1:
        true_positive = ((pred == 1) & (target == 1)).float().sum()
        false_positive = ((pred == 1) & (target == 0)).float().sum()
        false_negative = ((pred == 0) & (target == 1)).float().sum()
    else:
        # Multi-class: calculate for all non-background classes
        true_positive = ((pred > 0) & (target > 0) & (pred == target)).float().sum()
        false_positive = ((pred > 0) & ((target == 0) | (pred != target))).float().sum()
        false_negative = ((pred == 0) & (target > 0)).float().sum()
    
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    
    return precision.item(), recall.item()

def compute_metrics(preds, targets):
    """
    Compute evaluation metrics for segmentation.
    
    Args:
        preds: Predicted segmentation masks (batch_size, height, width)
        targets: Ground truth segmentation masks (batch_size, height, width)
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary masks if needed
    if preds.dim() > 3:
        preds = torch.argmax(preds, dim=1)
    
    # Compute intersection and union
    intersection = torch.logical_and(preds, targets).sum().float()
    union = torch.logical_or(preds, targets).sum().float()
    
    # Compute true positives, false positives, false negatives
    tp = intersection
    fp = (preds == 1).sum().float() - tp
    fn = (targets == 1).sum().float() - tp
    
    # Compute metrics
    iou = intersection / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    } 