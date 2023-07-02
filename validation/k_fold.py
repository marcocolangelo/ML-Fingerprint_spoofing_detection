import numpy as np
def k_fold_cross_validation(X, y, K, model, metric):
    """
    Esegue la validazione incrociata K-Fold su un set di dati.
    
    Parametri:
    X: array-like, shape (n_samples, n_features)
        Matrice delle feature.
    y: array-like, shape (n_samples,)
        Vettore dei target.
    K: int
        Numero di fold.
    model: oggetto con metodi fit e predict
        Modello di machine learning da valutare.
    metric: callable
        Funzione per calcolare una metrica di prestazione tra y_true e y_pred.
    
    Restituisce:
    scores: list of float
        Lista dei punteggi di prestazione per ogni iterazione della validazione incrociata.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // K
    scores = []
    
    for i in range(K):
        # Suddividi il set di dati in set di addestramento e test
        start = i * fold_size
        end = start + fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        
        # Addestra il modello sui dati di addestramento
        model.fit(X_train, y_train)
        
        # Valuta le prestazioni del modello sui dati di test
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        scores.append(score)
    
    return scores