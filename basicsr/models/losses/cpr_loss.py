import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastivePromptLoss(nn.Module):
    """
    Contrastive Prompt Regularization (CPR) Loss
    
    Enforces functional alignment between prompts and restoration tasks
    by penalizing outputs generated with mismatched prompts.
    
    Args:
        vgg_layer (str): VGG layer for perceptual similarity (e.g., 'relu2_2')
        alpha (float): Weight for CPR loss component
        num_negative_samples (int): Number of negative samples to use
    """
    
    def __init__(self, vgg_layer='relu2_2', alpha=0.01, num_negative_samples=4):
        super(ContrastivePromptLoss, self).__init__()
        self.alpha = alpha
        self.num_negative_samples = num_negative_samples
        
        # Use simple feature extractor instead of full VGG
        # to avoid additional dependencies
        self.feature_extractor = self._build_feature_extractor()
        
    def _build_feature_extractor(self):
        """Simple CNN feature extractor as proxy for VGG"""
        layers = []
        in_channels = 3
        out_channels = [64, 128, 256]
        
        for out_ch in out_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_ch
        
        return nn.Sequential(*layers)
    
    def extract_features(self, x):
        """Extract perceptual features"""
        # Normalize to [0,1] if needed
        if x.min() < 0:
            x = (x + 1) / 2
        
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features
    
    def forward(self, output_pos, outputs_neg, gt):
        """
        Compute CPR loss
        
        Args:
            output_pos: Restored image with correct prompt [B, C, H, W]
            outputs_neg: List of restored images with incorrect prompts
            gt: Ground truth image [B, C, H, W]
            
        Returns:
            loss_cpr: Contrastive prompt regularization loss
        """
        # Extract features
        feat_gt = self.extract_features(gt)
        feat_pos = self.extract_features(output_pos)
        
        # Positive loss: minimize distance to GT
        loss_pos = F.mse_loss(feat_pos, feat_gt)
        
        # Negative loss: maximize distance to GT for wrong prompts
        loss_neg = 0
        for output_neg in outputs_neg:
            feat_neg = self.extract_features(output_neg)
            loss_neg += F.mse_loss(feat_neg, feat_gt)
        
        loss_neg = loss_neg / len(outputs_neg)
        
        # CPR: push positive close, negative far
        loss_cpr = loss_pos - loss_neg
        
        return loss_cpr


class SimplifiedCPRLoss(nn.Module):
    """
    Simplified CPR Loss using pixel-space differences
    Faster alternative when VGG is not available
    """
    
    def __init__(self, alpha=0.01, num_negative_samples=4):
        super(SimplifiedCPRLoss, self).__init__()
        self.alpha = alpha
        self.num_negative_samples = num_negative_samples
        
    def forward(self, output_pos, outputs_neg, gt):
        """
        Compute simplified CPR loss in pixel space
        
        Args:
            output_pos: Restored image with correct prompt [B, C, H, W]
            outputs_neg: List of restored images with incorrect prompts
            gt: Ground truth image [B, C, H, W]
            
        Returns:
            loss_cpr: Simplified CPR loss
        """
        # Positive loss: L1 distance to GT
        loss_pos = F.l1_loss(output_pos, gt)
        
        # Negative loss: average L1 distance for wrong prompts
        loss_neg = 0
        for output_neg in outputs_neg:
            loss_neg += F.l1_loss(output_neg, gt)
        
        loss_neg = loss_neg / len(outputs_neg)
        
        # CPR: encourage correct prompt, discourage wrong prompts
        loss_cpr = loss_pos - loss_neg
        
        return loss_cpr


def sample_negative_tasks(current_task, all_tasks, num_samples=4):
    """
    Sample negative task types for CPR
    
    Args:
        current_task: Current degradation task label [B] or int
        all_tasks: List of all possible task labels (e.g., [0,1,2,3,4])
        num_samples: Number of negative samples
        
    Returns:
        negative_tasks: List of negative task labels
    """
    import random
    
    if isinstance(current_task, torch.Tensor):
        current_task = current_task[0].item()
    
    # Get all tasks except current
    negative_pool = [t for t in all_tasks if t != current_task]
    
    # Sample without replacement
    num_samples = min(num_samples, len(negative_pool))
    negative_tasks = random.sample(negative_pool, num_samples)
    
    return negative_tasks