"""
Experience Replay with DER++ Trainer
====================================
Replays previously learned data to mitigate catastrophic forgetting.
DER++ uses both standard replay and dark experience replay (logit distillation).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer with reservoir sampling.
    
    Maintains a fixed-size buffer of representative samples from previous tasks.
    Uses reservoir sampling to ensure class balance across tasks.
    
    Conceptual role:
      - Store (image, label, logits) triples from learned tasks
      - When learning new task, mix in replay samples to remind model of previous tasks
      - Prevents catastrophic forgetting by maintaining some gradient signal
        from old tasks during new task training
    """
    
    def __init__(self, buffer_size=200, num_classes_per_task=2):
        """
        Args:
            buffer_size (int): Maximum number of samples to store
            num_classes_per_task (int): Number of classes per task (usually 2)
        """
        self.buffer_size = buffer_size
        self.num_classes_per_task = num_classes_per_task
        
        # Storage
        self.images = []  # List of image tensors
        self.labels = []  # List of labels
        self.logits = []  # List of logits (for DER++, dark experience replay)
        
        self.counter = 0  # For reservoir sampling
    
    def add_task_data(self, data_loader, model, device):
        """
        Add samples from a completed task to the replay buffer.
        Uses reservoir sampling to maintain class balance.
        
        Reservoir sampling: ensures that new samples have equal probability
        of being added to buffer, preventing recent tasks from dominating.
        
        Args:
            data_loader: DataLoader for task data
            model: Neural network model (to compute logits)
            device (str): Device for computation
        """
        model.eval()
        
        with torch.no_grad():
            for x, y, _ in data_loader:
                x, y = x.to(device), y.to(device)
                
                # Compute logits (for dark experience replay)
                logits = model(x)
                
                # Store each sample with reservoir sampling
                for i in range(x.size(0)):
                    sample_x = x[i:i+1].cpu()
                    sample_y = y[i:i+1].cpu()
                    sample_logits = logits[i:i+1].cpu().detach()
                    
                    if len(self.images) < self.buffer_size:
                        # Buffer not full, add sample
                        self.images.append(sample_x)
                        self.labels.append(sample_y)
                        self.logits.append(sample_logits)
                    else:
                        # Buffer full, use reservoir sampling with random replacement
                        self.counter += 1
                        # Probability of keeping this sample = buffer_size / counter
                        if np.random.rand() < self.buffer_size / self.counter:
                            # Replace random sample in buffer
                            idx = np.random.randint(self.buffer_size)
                            self.images[idx] = sample_x
                            self.labels[idx] = sample_y
                            self.logits[idx] = sample_logits
    
    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.
        
        Args:
            batch_size (int): Number of samples to draw
        
        Returns:
            tuple: (images, labels, logits) or None if buffer empty
        """
        if len(self.images) == 0:
            return None
        
        # Randomly sample without replacement
        sample_size = min(batch_size, len(self.images))
        indices = np.random.choice(len(self.images), size=sample_size, replace=False)
        
        images = torch.cat([self.images[i] for i in indices], dim=0)
        labels = torch.cat([self.labels[i] for i in indices], dim=0)
        logits = torch.cat([self.logits[i] for i in indices], dim=0)
        
        return images, labels, logits
    
    def get_buffer_size(self):
        """Get current number of samples in buffer."""
        return len(self.images)


class DERppTrainer:
    """
    DER++ (Dark Experience Replay++) trainer.
    
    Conceptual approach:
      - Maintain replay buffer of samples from previous tasks
      - For new task: mix new task data with replay buffer data
      - Two regularization terms for replay samples:
        1. Standard cross-entropy on replayed samples
        2. Dark experience replay: MSE loss on model's current logits vs stored logits
      - Expected to show strong resistance to forgetting
    
    DER++ loss components:
      L_new = CE(f(x_new), y_new)
      L_replay = alpha * MSE(f(x_replay), logits_replay) + beta * CE(f(x_replay), y_replay)
      L_total = L_new + L_replay
      
      where:
        alpha = 0.1 (dark experience replay weight)
        beta = 0.5  (standard replay weight)
    """
    
    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 buffer_size=200, alpha=0.1, beta=0.5):
        """
        Args:
            model: Neural network model
            device (str): Device for training
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
            buffer_size (int): Maximum size of replay buffer
            alpha (float): Weight for dark experience replay (logit MSE)
            beta (float): Weight for standard experience replay (CE)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, num_classes_per_task=2)
    
    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task using DER++ loss.
        
        Mixes new task training with experience replay to prevent forgetting.
        
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
                
                # New task loss
                logits = self.model(x)
                loss_new = self.criterion_ce(logits, y)
                
                total_loss_batch = loss_new
                
                # Add replay buffer loss if available
                if self.replay_buffer.get_buffer_size() > 0:
                    replay_batch_size = min(x.size(0), self.replay_buffer.get_buffer_size())
                    replay_data = self.replay_buffer.sample(replay_batch_size)
                    
                    if replay_data is not None:
                        x_replay, y_replay, logits_replay = replay_data
                        x_replay = x_replay.to(self.device)
                        y_replay = y_replay.to(self.device)
                        logits_replay = logits_replay.to(self.device)
                        
                        # Model logits on replay samples
                        logits_replay_pred = self.model(x_replay)
                        
                        # Dark experience replay (logit-based distillation)
                        loss_der = self.criterion_mse(logits_replay_pred, logits_replay)
                        
                        # Standard experience replay (cross-entropy)
                        loss_replay_ce = self.criterion_ce(logits_replay_pred, y_replay)
                        
                        # Combine losses
                        loss_replay = self.alpha * loss_der + self.beta * loss_replay_ce
                        total_loss_batch += loss_replay
                
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} " +
                      f"(Buffer size: {self.replay_buffer.get_buffer_size()})")
        
        return avg_loss
    
    def update_buffer(self, train_loader):
        """
        Add current task data to replay buffer after training.
        
        Called after finishing training on each task to populate
        the buffer with representative samples.
        
        Args:
            train_loader: DataLoader for current task training data
        """
        self.replay_buffer.add_task_data(train_loader, self.model, self.device)
    
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
