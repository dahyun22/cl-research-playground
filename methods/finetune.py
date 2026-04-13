"""
Fine-tuning Baseline Trainer
==============================
Simple baseline: standard SGD/Adam training on current task only.
No memory overhead, no regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class FinetuneTrainer:
    """
    Fine-tuning baseline trainer.
    
    Conceptual approach:
      - Train model on each new task using standard cross-entropy loss
      - No mechanism to prevent catastrophic forgetting
      - Serves as baseline for comparison
      - Expected to show high forgetting on previous tasks
    """
    
    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5):
        """
        Args:
            model: Neural network model (e.g., MNIST_MLP or CIFAR10_CNN)
            device (str): Device for training ("cpu" or "cuda")
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task.
        
        Args:
            train_loader: DataLoader for current task training data
            verbose (bool): Print training progress
        
        Returns:
            float: Final training loss
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, test_loader):
        """
        Evaluate model accuracy on test data.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            float: Accuracy (0-1)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / max(total, 1)
        return accuracy
