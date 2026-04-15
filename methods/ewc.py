"""
Elastic Weight Consolidation (EWC) Trainer
===========================================
Protects important parameters for previous tasks using Fisher Information Matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Store (fisher_dict, optimal_params) per task for correct multi-task EWC
        self.ewc_tasks = []  # list of (fisher_dict, optimal_params) tuples
    
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
        num_samples = 0

        for x, y, _ in data_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Compute per-sample squared gradients (correct Fisher diagonal)
            # F_i = E[(∂log p(y|x,θ)/∂θ_i)^2], not (E[∂log p/∂θ_i])^2
            #
            # We sample y ~ p(y|x, θ) instead of using the true label.
            # This computes the *true* Fisher (expectation under the model's
            # own distribution), rather than the empirical Fisher (expectation
            # under the data distribution).  On easy tasks the model becomes
            # very confident (p ≈ 0.999), so gradients w.r.t. the true label
            # collapse to near-zero at convergence, making the empirical Fisher
            # useless as an importance measure.  Sampling from the model keeps
            # gradient magnitudes meaningful regardless of confidence level.
            for i in range(x.size(0)):
                self.model.zero_grad()
                logit = self.model(x[i:i+1])
                prob = F.softmax(logit, dim=1)
                sampled_y = torch.multinomial(prob[0], num_samples=1).squeeze()
                log_prob = F.log_softmax(logit, dim=1)[0, sampled_y]
                log_prob.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2

            num_samples += x.size(0)

        # Average over all samples
        for name in fisher:
            fisher[name] /= max(num_samples, 1)

        return fisher
    
    def consolidate_task(self, data_loader):
        """
        After training on a task, consolidate Fisher and optimal parameters.
        Stores a separate (Fisher, anchor) pair per task so each task's
        penalty uses the correct anchor point.

        Args:
            data_loader: DataLoader for completed task
        """
        # Compute Fisher for this task
        new_fisher = self.compute_fisher(data_loader)

        # Store optimal params at this task's solution as anchor
        optimal_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }

        self.ewc_tasks.append((new_fisher, optimal_params))
    
    def ewc_loss(self):
        """
        Compute EWC penalty term summed over all previous tasks.

        Each task contributes its own F_t * (θ - θ*_t)^2 term, so the
        anchor point is always the parameters that were optimal for *that*
        specific task.

        Returns:
            torch.Tensor: Scalar EWC loss
        """
        loss = 0.0

        for fisher_dict, optimal_params in self.ewc_tasks:
            for name, param in self.model.named_parameters():
                if name in fisher_dict:
                    fisher_values = fisher_dict[name]
                    optimal_values = optimal_params[name]
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
                if self.ewc_tasks:
                    ewc_reg = self.ewc_loss()
                    total_task_loss = task_loss + ewc_reg
                else:
                    ewc_reg = torch.tensor(0.0)
                    total_task_loss = task_loss

                total_task_loss.backward()
                self.optimizer.step()

                total_loss += total_task_loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} "
                      f"(task={task_loss.item():.4f}, ewc={ewc_reg.item():.4f})")
        
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
