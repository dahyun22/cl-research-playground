"""
Main Training Loop
==================
Orchestrates task-incremental learning for all three methods.
"""

import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from methods.finetune import FinetuneTrainer
from methods.ewc import EWCTrainer
from methods.ewc_separate_head import EWCSeparateHeadTrainer
from methods.derpp import DERppTrainer
from methods.hat import HATTrainer
from methods.co2l import Co2LTrainer
from methods.gem import GEMTrainer
from methods.lwf import LwFTrainer
from methods.si import SITrainer
from models import MNIST_MLP, CIFAR10_CNN, MLP_Co2L, CNN_Co2L

# Maps standard model class → Co2L variant with backbone/proj_head split
_CO2L_MODEL_MAP = {
    MNIST_MLP:   MLP_Co2L,
    CIFAR10_CNN: CNN_Co2L,
}


class TaskIncrementalLearner:
    """
    Task-incremental learning orchestrator.
    
    Manages training loop across multiple tasks for different methods.
    Tracks accuracy matrix (task-wise performance), weight drift, and BWT.
    """
    
    def __init__(self, model_class, num_tasks, dataset, device="cpu",
                 learning_rate=0.001, epochs=5):
        """
        Args:
            model_class: Model class (MNIST_MLP or CIFAR10_CNN)
            num_tasks (int): Total number of tasks
            dataset: Dataset object (SplitMNIST or SplitCIFAR10)
            device (str): Device for training
            learning_rate (float): Learning rate
            epochs (int): Epochs per task
        """
        self.model_class = model_class
        self.num_tasks = num_tasks
        self.dataset = dataset
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def train_all_tasks(self, method_name="finetune", verbose=True):
        """
        Train model on all tasks sequentially using specified method.
        
        Conceptual flow:
          For each task t:
            1. Get training data for task t
            2. Train model on task t with specified method
            3. Evaluate on all tasks 0..t (to measure forgetting)
            4. Store parameter changes and accuracy
            5. Record to replay buffer (for DER++)
            6. Consolidate Fisher/EWC weights (for EWC)
        
        Args:
            method_name (str): "finetune", "ewc", or "derpp"
            verbose (bool): Print progress
        
        Returns:
            dict: Results containing accuracy_matrix, weight_drift, avg_accuracy, bwt
        """
        assert method_name in ["finetune", "ewc", "ewc_sh", "derpp", "hat", "co2l", "gem", "lwf", "si"]
        
        # Initialize results storage
        results = {
            "accuracy_matrix": [],  # List of lists: accuracy_matrix[i][j]
            "weight_drift": {},  # {layer_name: [drift_per_task]}
            "avg_accuracy": [],
        }
        
        # Initialize model — Co2L uses the Co2L variant of the dataset's model class
        if method_name == "co2l":
            co2l_cls = _CO2L_MODEL_MAP.get(self.model_class)
            if co2l_cls is None:
                raise ValueError(
                    f"Co2L has no registered variant for {self.model_class.__name__}. "
                    f"Supported: {[c.__name__ for c in _CO2L_MODEL_MAP]}"
                )
            model = co2l_cls().to(self.device)
        else:
            model = self.model_class().to(self.device)
        
        # Initialize trainer based on method
        if method_name == "finetune":
            trainer = FinetuneTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs
            )
        elif method_name == "ewc":
            trainer = EWCTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                ewc_lambda=1000000
            )
        elif method_name == "ewc_sh":
            trainer = EWCSeparateHeadTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                ewc_lambda=5000, num_tasks=self.num_tasks
            )
        elif method_name == "derpp":
            trainer = DERppTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                buffer_size=200, alpha=0.1, beta=0.5
            )
        elif method_name == "hat":
            trainer = HATTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                num_tasks=self.num_tasks, hat_lambda=0.75, s_max=400
            )
        elif method_name == "co2l":
            trainer = Co2LTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                distill_lambda=1.0, temperature=0.1
            )
        elif method_name == "gem":
            trainer = GEMTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                n_memories=256, margin=0.0
            )
        elif method_name == "lwf":
            trainer = LwFTrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                lwf_lambda=1.0, temperature=2.0
            )
        elif method_name == "si":
            trainer = SITrainer(
                model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                si_lambda=1.0, xi=0.1
            )
        
        # Track parameter snapshots for weight drift
        prev_params = model.get_parameters()
        
        # Train on each task sequentially
        task_iterator = tqdm(range(self.num_tasks), desc=f"[{method_name.upper()}]") if verbose else range(self.num_tasks)
        
        for task_id in task_iterator:
            if verbose:
                task_iterator.set_description(f"[{method_name.upper()}] Task {task_id+1}/{self.num_tasks}")
            
            # Get training data for this task
            train_loader = self.dataset.get_task_data(task_id, split="train")
            
            # LwF: snapshot current model as distillation reference before
            # training begins, so old_model reflects knowledge of tasks 0..t-1
            if method_name == "lwf":
                if verbose:
                    print(f"  Recording soft labels (LwF)...")
                trainer.record_soft_labels(train_loader)

            # Train on current task
            if verbose:
                print(f"\n  Training on task {task_id} ({method_name})")
            trainer.train_task(train_loader, verbose=False)
            
            # Evaluate on all tasks seen so far
            accuracies_this_row = []
            for eval_task_id in range(task_id + 1):
                test_loader = self.dataset.get_task_data(eval_task_id, split="test")
                # HAT uses task-specific hard masks at evaluation time.
                if hasattr(trainer, "set_eval_task"):
                    trainer.set_eval_task(eval_task_id)
                accuracy = trainer.evaluate(test_loader)
                accuracies_this_row.append(accuracy)
            
            results["accuracy_matrix"].append(accuracies_this_row)
            
            # Compute weight drift
            cost_dict, total_drift = model.compute_weight_drift(prev_params)
            for layer_name, drift_val in cost_dict.items():
                if layer_name not in results["weight_drift"]:
                    results["weight_drift"][layer_name] = []
                results["weight_drift"][layer_name].append(drift_val)
            
            # Average accuracy up to this task
            avg_acc = np.mean(accuracies_this_row)
            results["avg_accuracy"].append(avg_acc)
            
            if verbose:
                print(f"  Avg Accuracy (all tasks so far): {avg_acc:.4f}")
                print(f"  Total Weight Drift: {total_drift:.4f}")
            
            # Update previous parameters for next iteration
            prev_params = model.get_parameters()
            
            # Method-specific post-training steps
            if method_name in ("ewc", "ewc_sh"):
                # Consolidate Fisher for this task
                if verbose:
                    print(f"  Computing Fisher Information...")
                trainer.consolidate_task(train_loader)

            elif method_name == "derpp":
                # Update replay buffer with this task's data
                if verbose:
                    print(f"  Updating replay buffer...")
                trainer.update_buffer(train_loader)

            elif method_name == "hat":
                # Binarise current task mask and merge into cumulative mask
                if verbose:
                    print(f"  Consolidating HAT mask...")
                trainer.consolidate_task()

            elif method_name == "co2l":
                # Freeze current model snapshot as distillation reference
                if verbose:
                    print(f"  Consolidating Co2L prev model...")
                trainer.consolidate_task()

            elif method_name == "gem":
                # Store episodic memory for this task
                if verbose:
                    print(f"  Storing GEM episodic memory...")
                trainer.consolidate_task(train_loader)

            elif method_name == "si":
                # Convert accumulated ω into importance Ω; no data pass needed
                if verbose:
                    print(f"  Consolidating SI importance...")
                trainer.consolidate_task()
        
        # Compute Backward Transfer (BWT)
        bwt = self._compute_bwt(results["accuracy_matrix"])
        results["bwt"] = bwt
        
        if verbose:
            print(f"\n[{method_name.upper()}] Final BWT: {bwt:.4f}")
        
        return results
    
    def _compute_bwt(self, accuracy_matrix):
        """
        Compute Backward Transfer metric.
        
        Conceptual meaning:
          Measures how much learning new tasks hurt performance on old tasks.
          BWT = average change in old task accuracy after learning new tasks.
          Negative BWT indicates forgetting, positive indicates transfer.
        
        Formula:
          BWT = (1/(T-1)) * sum_j=0^{T-2} (A[T-1][j] - A[j][j])
          where A[i][j] = accuracy on task j after learning task i
        
        Args:
            accuracy_matrix (list): (num_tasks x num_tasks) accuracy matrix
        
        Returns:
            float: BWT score
        """
        if len(accuracy_matrix) <= 1:
            return 0.0
        
        T = len(accuracy_matrix)
        bwt_sum = 0.0
        
        # A[T-1][j] - A[j][j] for j in 0..T-2
        for j in range(T - 1):
            final_accuracy = accuracy_matrix[-1][j]  # Accuracy on task j after learning all tasks
            initial_accuracy = accuracy_matrix[j][j]  # Accuracy on task j when first learned
            bwt_sum += final_accuracy - initial_accuracy
        
        bwt = bwt_sum / (T - 1)
        return bwt
