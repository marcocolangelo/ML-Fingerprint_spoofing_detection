import numpy as np

# Carica il tuo dataset
def z_norm(D):

    # Calcola la media e la deviazione standard dei dati
    mean = np.mean(D, axis=0)
    std = np.std(D, axis=0)
    
    # Normalizza i dati
    normalized_data = (D - mean) / std
    
    return normalized_data

