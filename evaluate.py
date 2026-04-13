"""
Evaluation Module
=================
Contains utilities for computing continual learning metrics.
(Primary metrics are computed in TaskIncrementalLearner in train.py)
"""

import numpy as np


def compute_accuracy_matrix_stats(accuracy_matrix):
    """
    Compute statistics from accuracy matrix.
    
    Args:
        accuracy_matrix (list): (num_tasks x num_tasks) matrix
                                where [i][j] = accuracy on task j after task i
    
    Returns:
        dict: Statistics including diagonal (initial), final, average
    """
    matrix_arr = np.array(accuracy_matrix)
    
    stats = {
        "diagonal": np.diag(matrix_arr),  # Performance when task first learned
        "final_row": matrix_arr[-1],      # Final performance on all tasks
        "average_diagonal": np.mean(np.diag(matrix_arr)),
        "average_final": np.mean(matrix_arr[-1]),
    }
    
    return stats


def compute_forgetting(accuracy_matrix):
    """
    Compute forgetting measure for each task.
    
    Forgetting on task j = max(A[0][j], ..., A[j-1][j]) - A[-1][j]
    (How much performance dropped from best to final)
    
    Args:
        accuracy_matrix (list): (num_tasks x num_tasks) matrix
    
    Returns:
        list: Forgetting values per task
    """
    forgetting = []
    
    for j in range(len(accuracy_matrix)):
        max_acc = max([accuracy_matrix[i][j] for i in range(j + 1)])
        final_acc = accuracy_matrix[-1][j]
        forgetting.append(max_acc - final_acc)
    
    return forgetting


def compute_forward_transfer(accuracy_matrix):
    """
    Compute forward transfer: impact of previous tasks on learning new tasks.
    
    FWT is typically computed from a different experimental setup,
    but we can approximate it from our results.
    
    Args:
        accuracy_matrix (list): (num_tasks x num_tasks) matrix
    
    Returns:
        float: Average forward transfer estimate
    """
    # Forward transfer = average accuracy on new tasks during learning
    # (higher = previous tasks helped learn new ones)
    if len(accuracy_matrix) <= 1:
        return 0.0
    
    # Extract off-diagonal accuracies (performance before task was trained)
    off_diag_accs = []
    for i in range(len(accuracy_matrix)):
        for j in range(i + 1, len(accuracy_matrix)):
            # accuracy_matrix[i][j] = accuracy on task j after learning task i (i < j)
            if j < len(accuracy_matrix[i]):
                # We don't have this in our setup, so return placeholder
                pass
    
    # In our setup, we only evaluate on tasks already learned
    # So forward transfer is not directly measurable
    return 0.0
