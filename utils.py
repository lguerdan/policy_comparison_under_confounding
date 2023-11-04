import numpy as np
import pandas as pd


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_v_stats(D, A, Y):
    
    v10_true = ((A==1) & (D==0) & (Y==1)).mean()
    v00_true = ((A==0) & (D==0) & (Y==1)).mean()
    w10_true = ((A==1) & (D==0) & (Y==0)).mean()
    w00_true = ((A==0) & (D==0) & (Y==0)).mean()
    
    
    v11 = ((D==1) & (A==1) & (Y==1)).mean()
    v01 = ((D==1) & (A==0) & (Y==1)).mean()
    
    w11 = ((D==1) & (A==1) & (Y==0)).mean()
    w01 = ((D==1) & (A==0) & (Y==0)).mean()
    
    return {
        'v11': v11,
        'v01': v01,
        'w11': w11,
        'w01': w01,
        'v10_true': v10_true,
        'v00_true': v00_true,
        'w10_true': w10_true,
        'w00_true': w00_true
    }
