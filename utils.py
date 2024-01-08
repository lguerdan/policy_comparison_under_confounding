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
