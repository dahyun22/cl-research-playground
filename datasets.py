"""
Dataset Module: Split MNIST and Split CIFAR-10
===============================================
Implements task-incremental learning setup with binary classification tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


class SplitMNIST:
    """
    Split MNIST dataset for continual learning.
    
    Splits MNIST into 5 binary classification tasks:
      Task 0: digits {0, 1}
      Task 1: digits {2, 3}
      Task 2: digits {4, 5}
      Task 3: digits {6, 7}
      Task 4: digits {8, 9}
    
    Each task is binary classification (2 output units).
    """
    
    def __init__(self, data_root="./data", batch_size=32):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_tasks = 5
        self.task_labels = [
            (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
        ]
        self.input_size = 28 * 28  # 784
        self.num_classes_per_task = 2
        
        # Load full MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_data = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        self.test_data = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform
        )
    
    def get_task_data(self, task_id, split="train"):
        """
        Get dataloader for a specific task.
        
        Args:
            task_id (int): Task index (0-4)
            split (str): "train" or "test"
        
        Returns:
            DataLoader: Loader for the task
        """
        assert 0 <= task_id < self.num_tasks
        
        dataset = self.train_data if split == "train" else self.test_data
        class1, class2 = self.task_labels[task_id]
        
        # Filter data for this task's classes
        mask = (dataset.targets == class1) | (dataset.targets == class2)
        indices = torch.where(mask)[0]
        
        # Create subset and relabel (0->0, class2->1)
        task_dataset = TaskDataset(
            dataset=dataset,
            indices=indices,
            task_id=task_id,
            class_map={class1: 0, class2: 1},
            flatten=True
        )
        
        shuffle = (split == "train")
        return DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def get_all_test_data(self, task_ids):
        """
        Get test data for multiple tasks combined.
        Useful for evaluating all tasks after learning a new one.
        
        Args:
            task_ids (list): List of task indices
        
        Returns:
            DataLoader: Combined test loader
        """
        all_indices = []
        for task_id in task_ids:
            class1, class2 = self.task_labels[task_id]
            mask = (self.test_data.targets == class1) | (self.test_data.targets == class2)
            indices = torch.where(mask)[0]
            all_indices.append((task_id, indices))
        
        combined_dataset = CombinedTaskDataset(
            base_dataset=self.test_data,
            task_indices=all_indices,
            task_labels=self.task_labels,
            flatten=True
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )


class SplitCIFAR10:
    """
    Split CIFAR-10 dataset for continual learning.
    
    Splits CIFAR-10 into 5 binary classification tasks:
      Task 0: {airplane, automobile}
      Task 1: {bird, cat}
      Task 2: {deer, dog}
      Task 3: {frog, horse}
      Task 4: {ship, truck}
    
    Each task is binary classification (2 output units).
    """
    
    def __init__(self, data_root="./data", batch_size=32):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_tasks = 5
        
        # CIFAR-10 class names: 0=plane, 1=auto, 2=bird, 3=cat, 4=deer,
        #                        5=dog, 6=frog, 7=horse, 8=ship, 9=truck
        self.task_labels = [
            (0, 1),   # airplane, automobile
            (2, 3),   # bird, cat
            (4, 5),   # deer, dog
            (6, 7),   # frog, horse
            (8, 9)    # ship, truck
        ]
        
        self.input_size = 32 * 32 * 3  # 3072
        self.num_classes_per_task = 2
        
        # Normalize CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        
        self.train_data = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        self.test_data = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform
        )
    
    def get_task_data(self, task_id, split="train"):
        """
        Get dataloader for a specific task.
        
        Args:
            task_id (int): Task index (0-4)
            split (str): "train" or "test"
        
        Returns:
            DataLoader: Loader for the task
        """
        assert 0 <= task_id < self.num_tasks
        
        dataset = self.train_data if split == "train" else self.test_data
        class1, class2 = self.task_labels[task_id]
        
        # Filter data for this task's classes
        targets = torch.tensor(dataset.targets)
        mask = (targets == class1) | (targets == class2)
        indices = torch.where(mask)[0]
        
        # Create subset and relabel
        task_dataset = TaskDataset(
            dataset=dataset,
            indices=indices,
            task_id=task_id,
            class_map={class1: 0, class2: 1},
            flatten=False  # Keep as images (3, 32, 32)
        )
        
        shuffle = (split == "train")
        return DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def get_all_test_data(self, task_ids):
        """
        Get test data for multiple tasks combined.
        Useful for evaluating all tasks after learning a new one.
        
        Args:
            task_ids (list): List of task indices
        
        Returns:
            DataLoader: Combined test loader
        """
        all_indices = []
        targets = torch.tensor(self.test_data.targets)
        
        for task_id in task_ids:
            class1, class2 = self.task_labels[task_id]
            mask = (targets == class1) | (targets == class2)
            indices = torch.where(mask)[0]
            all_indices.append((task_id, indices))
        
        combined_dataset = CombinedTaskDataset(
            base_dataset=self.test_data,
            task_indices=all_indices,
            task_labels=self.task_labels,
            flatten=False
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )


class TaskDataset(Dataset):
    """
    Wrapper for a single task's data with optional flattening.
    """
    
    def __init__(self, dataset, indices, task_id, class_map, flatten=False):
        """
        Args:
            dataset: Base dataset (MNIST or CIFAR10)
            indices (torch.Tensor): Indices for this task
            task_id (int): Task identifier
            class_map (dict): Mapping from original labels to binary labels
            flatten (bool): Whether to flatten images
        """
        self.dataset = dataset
        self.indices = indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices
        self.task_id = task_id
        self.class_map = class_map
        self.flatten = flatten
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, y = self.dataset[original_idx]
        
        # Relabel to binary
        y = self.class_map[y]
        
        # Flatten if needed
        if self.flatten:
            x = x.view(-1)  # Flatten to 1D
        
        return x, y, self.task_id


class CombinedTaskDataset(Dataset):
    """
    Wrapper for multiple tasks combined (for evaluation).
    """
    
    def __init__(self, base_dataset, task_indices, task_labels, flatten=False):
        """
        Args:
            base_dataset: Base dataset (MNIST or CIFAR10)
            task_indices (list): List of (task_id, indices) tuples
            task_labels (list): List of (class1, class2) tuples per task
            flatten (bool): Whether to flatten images
        """
        self.base_dataset = base_dataset
        self.task_indices = task_indices
        self.task_labels = task_labels
        self.flatten = flatten
        
        # Build mapping of global index to (task_id, local_idx)
        self.global_to_task = []
        for task_id, indices in task_indices:
            for local_idx in range(len(indices)):
                self.global_to_task.append((task_id, indices, local_idx))
    
    def __len__(self):
        return sum(len(indices) for _, indices in self.task_indices)
    
    def __getitem__(self, idx):
        task_id, indices, local_idx = self.global_to_task[idx]
        original_idx = indices[local_idx]
        
        x, original_y = self.base_dataset[original_idx]
        
        # Relabel to binary for task
        class1, class2 = self.task_labels[task_id]
        class_map = {class1: 0, class2: 1}
        y = class_map[original_y]
        
        # Flatten if needed
        if self.flatten:
            x = x.view(-1)
        
        return x, y, task_id
