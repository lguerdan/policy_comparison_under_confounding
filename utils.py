import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def generate_positive_semidefinite_matrix(dim, seed=None):
    """
    Generates a random positive semidefinite matrix.
    
    Parameters:
        dim (int): The dimension of the square matrix.
        seed (int, optional): A seed for the random number generator to make results reproducible.
        
    Returns:
        numpy.ndarray: A dim x dim positive semidefinite matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random matrix A
    A = np.random.randn(dim, dim)
    
    # Multiply A by its transpose to get a symmetric and positive semidefinite matrix
    matrix = np.dot(A, A.T)
    
    # Adding a small value to the diagonal elements to ensure the matrix is positive definite
    matrix += np.eye(dim) * 1e-8
    
    return matrix

def k_fold_split_and_complement(data_dict, k):
    # Assuming all arrays have the same first dimension
    n_elements = next(iter(data_dict.values())).shape[0]

    # Create an array of indices
    indices = np.arange(n_elements)

    # Optionally shuffle the indices for randomness
    np.random.shuffle(indices)

    # Calculate the size of each fold
    fold_size = n_elements // k
    sizes = [fold_size] * k
    for i in range(n_elements % k):
        sizes[i] += 1

    # Split indices into k groups and create k fold dicts
    folds = []
    out_folds = []
    current = 0
    for size in sizes:
        fold_indices = indices[current:current + size]
        out_fold_indices = np.delete(indices, np.arange(current, current + size))

        fold_dict = {key: val[fold_indices] for key, val in data_dict.items()}
        out_fold_dict = {key: val[out_fold_indices] for key, val in data_dict.items()}

        folds.append(fold_dict)
        out_folds.append(out_fold_dict)

        current += size

    return folds, out_folds


def average_numeric_dataframes(dfs):
    if not dfs:
        raise ValueError("List of DataFrames is empty")

    # Identify numeric and non-numeric columns
    numeric_cols = dfs[0].select_dtypes(include='number').columns
    non_numeric_cols = dfs[0].select_dtypes(exclude='number').columns

    # Average numeric columns
    avg_numeric = sum(df[numeric_cols] for df in dfs) / len(dfs)

    # Extract non-numeric columns from the first dataframe
    non_numeric = dfs[0][non_numeric_cols]

    # Combine averaged numeric columns with non-numeric columns
    avg_df = pd.concat([avg_numeric, non_numeric], axis=1)

    return avg_df