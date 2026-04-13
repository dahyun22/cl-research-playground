"""
Elastic Weight Consolidation (EWC) Trainer
===========================================
Protects important parameters for previous tasks using Fisher Information Matrix.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class EWCTrainer:
    """
    EWC (Elastic Weight Consolidation) trainer.
    
    Conceptual approach:
      - After learning each task, compute Fisher Information Matrix (diagonal approx.)
      - Fisher indicates which parameters are most important for that task
      - Add regularization term to new task losses to prevent large changes to
        important previous parameters
      - Expected to show reduced forgetting compared to fine-tuning baseline
    
    Key equation:
      L_ewc = L_task + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2
      
      where:
        L_task = cross-entropy loss on current task
        F_i = Fisher Information diagonal element (importance)
        theta_i = current parameter value
        theta_i* = optimal parameter after previous task
        lambda = regularization strength
    """
    
    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5, ewc_lambda=5000):
        """
        Args:
            model: Neural network model
            device (str): Device for training
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
            ewc_lambda (float): EWC regularization strength
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.ewc_lambda = ewc_lambda
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Store Fisher and optimal parameters for each task
        self.fisher_dict = {}  # {param_name: fisher_values}
        self.optimal_params = {}  # {param_name: optimal_values}
    
    def compute_fisher(self, data_loader):
        """
        Compute diagonal Fisher Information Matrix for current task.
        
        Fisher Information captures task-relevant parameter importance:
        - Large F_i means parameter i significantly affects task performance
        - EWC will constrain changes to parameters with large Fisher values
        
        Method: F_i ≈ (∂log p / ∂θ_i)^2 averaged over data
        
        Args:
            data_loader: DataLoader for current task
        
        Returns:
            dict: Parameter name -> Fisher diagonal values
        """
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        
        for x, y, _ in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            # Compute gradients
            loss.backward(retain_graph=True)
            
            # Fisher elements are squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Average over batches
        num_batches = len(data_loader)
        for name in fisher:
            fisher[name] /= max(num_batches, 1)
        
        return fisher
    
    def consolidate_task(self, data_loader):
        """
        After training on a task, consolidate Fisher and optimal parameters.
        Accumulates Fisher from all previous tasks.
        
        Args:
            data_loader: DataLoader for completed task
        """
        # Compute Fisher for this task
        new_fisher = self.compute_fisher(data_loader)
        
        # Accumulate Fisher (sum over all tasks seen so far)
        for name, param in self.model.named_parameters():
            if name not in self.fisher_dict:
                self.fisher_dict[name] = new_fisher[name].clone()
            else:
                self.fisher_dict[name] += new_fisher[name]
            
            # Store current parameters as optimal for this task
            self.optimal_params[name] = param.data.clone().detach()
    
    def ewc_loss(self):
        """
        Compute EWC penalty term.
        
        This regularization term is added to the task loss to prevent
        important previous parameters from changing too much.
        
        Returns:
            torch.Tensor: Scalar EWC loss
        """
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                # Only apply if we have Fisher values from previous tasks
                fisher_values = self.fisher_dict[name]
                optimal_values = self.optimal_params[name]
                
                # (param - optimal)^2 weighted by Fisher importance
                diff = param - optimal_values
                loss += (fisher_values * diff ** 2).sum()
        
        return (self.ewc_lambda / 2) * loss
    
    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task with EWC regularization.
        
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
                task_loss = self.criterion(logits, y)
                
                # Add EWC regularization if we have previous tasks
                if self.fisher_dict:
                    ewc_reg = self.ewc_loss()
                    total_task_loss = task_loss + ewc_reg
                else:
                    total_task_loss = task_loss
                
                total_task_loss.backward()
                self.optimizer.step()
                
                total_loss += total_task_loss.item()
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
