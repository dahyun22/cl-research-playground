"""
Model Module: MNIST MLP and CIFAR10 CNN
=======================================
Defines neural network architectures with parameter tracking capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class MNIST_MLP(nn.Module):
    """
    Small Multi-Layer Perceptron for MNIST.
    
    Architecture:
      Input: 784 (28x28 flattened)
      Hidden layers: [256, 256] with ReLU
      Output: 2 (binary classification)
    
    For task-incremental learning, the final layer outputs 2 logits
    corresponding to the two classes in each binary task.
    """
    
    def __init__(self, input_size=784, hidden_dims=None, num_classes=2):
        """
        Args:
            input_size (int): Input feature dimension (784 for MNIST)
            hidden_dims (list): Dimensions of hidden layers
            num_classes (int): Number of output classes (2 for binary tasks)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build network
        layers = []
        in_dim = input_size
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(("fc{}".format(i), nn.Linear(in_dim, hidden_dim)))
            layers.append(("relu{}".format(i), nn.ReLU()))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(("output", nn.Linear(in_dim, num_classes)))
        
        self.network = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, 784)
        
        Returns:
            torch.Tensor: Logits (batch_size, 2)
        """
        # Ensure x is flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def get_parameters(self):
        """
        Get all model parameters as a dictionary.
        Used for tracking weight changes across tasks.
        
        Returns:
            dict: Parameter name -> parameter value (detached)
        """
        return {name: param.data.clone().detach() 
                for name, param in self.named_parameters()}
    
    def compute_weight_drift(self, prev_params):
        """
        Compute L2 norm of parameter changes from previous checkpoint.

        Conceptually: measures how much the model's weights have changed
        after learning a new task. High drift suggests significant adaptation.

        Args:
            prev_params (dict): Previous parameter snapshot

        Returns:
            dict: Layer name -> L2 drift value
            float: Overall L2 drift across all parameters
        """
        drift_dict = {}
        total_drift = 0.0

        for name, param in self.named_parameters():
            if name in prev_params:
                diff = param.data - prev_params[name]
                layer_drift = torch.norm(diff, p=2).item()
                drift_dict[name] = layer_drift
                total_drift += layer_drift ** 2

        total_drift = total_drift ** 0.5
        return drift_dict, total_drift

    def get_hat_layer_sizes(self):
        """Sizes of hidden layers that receive HAT attention masks."""
        return list(self.hidden_dims)

    def forward_hat(self, x, masks):
        """
        Forward pass with per-layer HAT attention masks applied after each ReLU.

        masks[i] gates hidden layer i's output element-wise. The output layer
        is never masked (task head is always fully active).

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size)
            masks (list[Tensor]): Soft or hard attention mask per hidden layer,
                                  masks[i] has shape (hidden_dims[i],)

        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for i in range(len(self.hidden_dims)):
            x = getattr(self.network, f"fc{i}")(x)
            x = getattr(self.network, f"relu{i}")(x)
            x = x * masks[i]

        x = self.network.output(x)
        return x

    def get_hat_gradient_mask_info(self):
        """
        Weight-to-mask-layer mapping for HAT gradient masking.

        Returns a list of (param_name, pre_mask_idx, post_mask_idx) tuples:
          - param_name   : key in named_parameters()
          - pre_mask_idx : cumulative_mask index for the input side  (-1 = none)
          - post_mask_idx: cumulative_mask index for the output side (-1 = none)

        Used by HATTrainer._clip_weight_grads() to zero gradients that would
        change weights connected to neurons already claimed by previous tasks.
        """
        n = len(self.hidden_dims)
        mappings = []
        for i in range(n):
            pre_idx = i - 1   # -1 for first layer (raw input is not masked)
            post_idx = i
            mappings.append((f"network.fc{i}.weight", pre_idx, post_idx))
            mappings.append((f"network.fc{i}.bias",   -1,      post_idx))
        # Output layer: input side is masked by last hidden mask, output side is not.
        mappings.append(("network.output.weight", n - 1, -1))
        mappings.append(("network.output.bias",   -1,    -1))
        return mappings


class MLP_Co2L(nn.Module):
    """
    MLP for Co2L (Contrastive Continual Learning).

    Architecture:
        backbone  : 784 → [256 → ReLU] × N      (shared feature extractor)
        classifier: feat_dim → num_classes        (task logits)
        proj_head : feat_dim → proj_dim (128)     (contrastive projection)

    forward() returns (features, projections):
        features    – backbone output, shape (B, feat_dim).
                      Pass to self.classifier(features) for logits, or
                      compare against the frozen previous-model's features
                      for AsymDistillLoss.
        projections – L2-normalised proj_head output, shape (B, proj_dim).
                      Used directly in SupConLoss.
    """

    def __init__(self, input_size=784, hidden_dims=None, num_classes=2, proj_dim=128):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        # Backbone: same hidden layers as MNIST_MLP, without the output layer
        layers = []
        in_dim = input_size
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f"fc{i}", nn.Linear(in_dim, hidden_dim)))
            layers.append((f"relu{i}", nn.ReLU()))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.feat_dim = in_dim  # 256 with default hidden_dims

        # Classification head (same role as MNIST_MLP's "output" layer)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # Projection head for SupConLoss
        self.proj_head = nn.Linear(self.feat_dim, proj_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input (B, 784) or (B, 1, 28, 28)

        Returns:
            features    (torch.Tensor): (B, feat_dim) – backbone representation
            projections (torch.Tensor): (B, proj_dim) – L2-normalised projection
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        features = self.backbone(x)                              # (B, 256)
        projections = F.normalize(self.proj_head(features), dim=1)  # (B, 128)
        return features, projections

    def get_parameters(self):
        return {name: param.data.clone().detach()
                for name, param in self.named_parameters()}

    def compute_weight_drift(self, prev_params):
        drift_dict = {}
        total_drift = 0.0
        for name, param in self.named_parameters():
            if name in prev_params:
                diff = param.data - prev_params[name]
                layer_drift = torch.norm(diff, p=2).item()
                drift_dict[name] = layer_drift
                total_drift += layer_drift ** 2
        return drift_dict, total_drift ** 0.5


class CNN_Co2L(nn.Module):
    """
    CNN for Co2L (Contrastive Continual Learning) on CIFAR-10.

    Architecture:
        backbone  : Conv(3→32)→ReLU→Pool → Conv(32→64)→ReLU→Pool
                    → Flatten → Linear(4096→512) → ReLU   (feat_dim=512)
        classifier: feat_dim → num_classes                 (task logits)
        proj_head : feat_dim → proj_dim (128)              (contrastive projection)

    Mirrors MLP_Co2L structure for CIFAR-10 inputs (3, 32, 32).
    forward() returns (features, projections).
    """

    def __init__(self, num_classes=2, proj_dim=128):
        super().__init__()

        self.num_classes = num_classes
        self.proj_dim = proj_dim

        # Backbone: same conv blocks as CIFAR10_CNN, without the final fc2
        self.backbone = nn.Sequential(OrderedDict([
            ("conv1",  nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            ("relu1",  nn.ReLU()),
            ("pool1",  nn.MaxPool2d(kernel_size=2, stride=2)),
            # After pool1: (B, 32, 16, 16)
            ("conv2",  nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("relu2",  nn.ReLU()),
            ("pool2",  nn.MaxPool2d(kernel_size=2, stride=2)),
            # After pool2: (B, 64, 8, 8)
            ("flatten", nn.Flatten()),
            ("fc1",    nn.Linear(64 * 8 * 8, 512)),
            ("relu3",  nn.ReLU()),
        ]))
        self.feat_dim = 512

        # Classification head (same role as CIFAR10_CNN's fc2)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # Projection head for SupConLoss
        self.proj_head = nn.Linear(self.feat_dim, proj_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input (B, 3, 32, 32)

        Returns:
            features    (torch.Tensor): (B, 512) – backbone representation
            projections (torch.Tensor): (B, proj_dim) – L2-normalised projection
        """
        features = self.backbone(x)                                   # (B, 512)
        projections = F.normalize(self.proj_head(features), dim=1)   # (B, 128)
        return features, projections

    def get_parameters(self):
        return {name: param.data.clone().detach()
                for name, param in self.named_parameters()}

    def compute_weight_drift(self, prev_params):
        drift_dict = {}
        total_drift = 0.0
        for name, param in self.named_parameters():
            if name in prev_params:
                diff = param.data - prev_params[name]
                layer_drift = torch.norm(diff, p=2).item()
                drift_dict[name] = layer_drift
                total_drift += layer_drift ** 2
        return drift_dict, total_drift ** 0.5


class CIFAR10_CNN(nn.Module):
    """
    Small Convolutional Neural Network for CIFAR-10.
    
    Architecture:
      Conv layers: 2 blocks with [32, 64] filters, ReLU, MaxPool
      FC layers: [512, 2]
      Output: 2 (binary classification)
    
    For task-incremental learning, the final layer outputs 2 logits
    corresponding to the two classes in each binary task.
    """
    
    def __init__(self, num_classes=2):
        """
        Args:
            num_classes (int): Number of output classes (2 for binary tasks)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks
        # Input: (batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: (batch, 32, 16, 16)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: (batch, 64, 8, 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 32, 32)
        
        Returns:
            torch.Tensor: Logits (batch_size, 2)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
    
    def get_parameters(self):
        """
        Get all model parameters as a dictionary.
        Used for tracking weight changes across tasks.
        
        Returns:
            dict: Parameter name -> parameter value (detached)
        """
        return {name: param.data.clone().detach() 
                for name, param in self.named_parameters()}
    
    def compute_weight_drift(self, prev_params):
        """
        Compute L2 norm of parameter changes from previous checkpoint.

        Conceptually: measures how much the model's weights have changed
        after learning a new task. High drift suggests significant adaptation.
        Lower drift indicates the regularizer (e.g., EWC) is constraining changes.

        Args:
            prev_params (dict): Previous parameter snapshot

        Returns:
            dict: Layer name -> L2 drift value
            float: Overall L2 drift across all parameters
        """
        drift_dict = {}
        total_drift = 0.0

        for name, param in self.named_parameters():
            if name in prev_params:
                diff = param.data - prev_params[name]
                layer_drift = torch.norm(diff, p=2).item()
                drift_dict[name] = layer_drift
                total_drift += layer_drift ** 2

        total_drift = total_drift ** 0.5
        return drift_dict, total_drift

    def get_hat_layer_sizes(self):
        """Sizes of masked layers: conv1 output channels, conv2 output channels, fc1 neurons."""
        return [32, 64, 512]

    def forward_hat(self, x, masks):
        """
        Forward pass with per-layer HAT attention masks.

        Masks are applied channel-wise after each conv block and neuron-wise
        after fc1. The output layer (fc2) is never masked.

          masks[0]: (32,)  – applied after conv1 → relu1 → pool1  (channel mask)
          masks[1]: (64,)  – applied after conv2 → relu2 → pool2  (channel mask)
          masks[2]: (512,) – applied after fc1   → relu3           (neuron mask)

        Args:
            x (torch.Tensor): Input (batch_size, 3, 32, 32)
            masks (list[Tensor]): [mask_conv1, mask_conv2, mask_fc1]

        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        # Conv block 1 + channel mask
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x * masks[0].view(1, -1, 1, 1)

        # Conv block 2 + channel mask
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x * masks[1].view(1, -1, 1, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1 + neuron mask
        x = self.relu3(self.fc1(x))
        x = x * masks[2]

        # Output (no mask)
        x = self.fc2(x)
        return x

    def get_hat_gradient_mask_info(self):
        """
        Weight-to-mask-layer mapping for HAT gradient masking.

        Returns a list of (param_name, pre_mask_idx, post_mask_idx) tuples.
        For fc1, the pre_mask is mask[1] (64-dim); HATTrainer expands it to
        4096 via repeat_interleave(64) to match the flattened conv2 output
        (64 channels × 8 × 8 spatial positions).
        """
        return [
            # param_name         pre_idx  post_idx
            ("conv1.weight",      -1,      0),   # (32,  3, 3, 3): out-channels → mask[0]
            ("conv1.bias",        -1,      0),
            ("conv2.weight",       0,      1),   # (64, 32, 3, 3): in-ch→mask[0], out-ch→mask[1]
            ("conv2.bias",        -1,      1),
            ("fc1.weight",         1,      2),   # (512, 4096): pre expanded ×64, out→mask[2]
            ("fc1.bias",          -1,      2),
            ("fc2.weight",         2,     -1),   # (2, 512): in-neurons → mask[2]
            ("fc2.bias",          -1,     -1),
        ]
