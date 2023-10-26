import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mcol(v):
    return v.reshape((v.size, 1))

def loadFile(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Spoofed fingerprint': 0,
        'Authentic fingerprint': 1,
    }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = mcol(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

# m is the number of dimensions
def plotSingle(D, L, m):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    print(m)
    for i in range(m):
        plt.figure()
        plt.xlabel("Feature " + str(i))
        plt.ylabel("Number of elements")
        plt.hist(D0[i, :], bins=60,density=True, alpha=0.7, label="Spoofed fingerprint")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Authentic fingerprint")
        plt.legend()
        plt.show()
        
def plotTot(D, L, m):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.figure()
    plt.xlabel("Feature " )
    plt.ylabel("Number of elements")
    plt.hist(D0[:, :], density=True, alpha=0.7, label="Spoofed fingerprint")
    plt.hist(D1[:, :], density=True, alpha=0.7, label="Authentic fingerprint")
    plt.legend()
    plt.show()

# m is the number of dimensions
def plotCross(D, L,m):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            plt.figure()
            plt.xlabel("Feature " + str(i))
            plt.ylabel("Feature " + str(j))
            plt.scatter(D0[i, :], D0[j, :], label="Spoofed fingerprint")
            plt.scatter(D1[i, :], D1[j, :], label="Authentic fingerprint")
            plt.legend()
            plt.show()

def stats(D, L):
    mean = D.mean(1)
    DC = D - mean.reshape((D.shape[0], 1))
    
    return DC


# def pearson_correlation(x, y):
#     # Calcola la media di x e y
#     mean_x = np.mean(x)
#     mean_y = np.mean(y)

#     # Calcola le differenze tra x e la media di x, e tra y e la media di y
#     diff_x = x - mean_x
#     diff_y = y - mean_y

#     # Calcola il prodotto delle differenze
#     diff_prod = diff_x * diff_y

#     # Calcola la somma dei quadrati delle differenze
#     sum_diff_squares = np.sqrt(np.sum(diff_x**2) * np.sum(diff_y**2))

#     # Calcola la correlazione di Pearson
#     correlation = np.sum(diff_prod) / sum_diff_squares

#     return correlation

# def plot_pearson_correlation(data, labels, target_class):
#     target_data = data[:,labels == target_class]
#     non_target_data = data[:,labels != target_class]

#     num_features = data.shape[0]
#     correlations_target = np.zeros((num_features, num_features))
#     correlations_non_target = np.zeros((num_features, num_features))

#     for i in range(num_features):
#         for j in range(num_features):
#             correlations_target[i, j] = pearson_correlation(target_data[:, i], target_data[:, j])
#             correlations_non_target[i, j] = pearson_correlation(non_target_data[:, i], non_target_data[:, j])

#     fig, axs = plt.subplots(1, 2, figsize=(10, 6))
#     im1= axs[0].matshow(correlations_target, cmap='coolwarm', vmin=-1, vmax=1)
#     axs[0].set_title('Target Class')
#     im2= axs[1].matshow(correlations_non_target, cmap='coolwarm', vmin=-1, vmax=1)
#     axs[1].set_title('Non-Target Class')
#     fig.colorbar(im1, ax=axs[0])
#     fig.colorbar(im2, ax=axs[1])
#     fig.suptitle('Pearson Correlation Coefficient')
#     plt.show()
    
def plot_features(DTR, LTR, m=2, appendToTitle=''):
     plot_correlations(DTR, "heatmap_" + appendToTitle)
     plot_correlations(DTR[:, LTR == 0], "heatmap_male_" + appendToTitle, cmap="Blues")
     plot_correlations(DTR[:, LTR == 1], "heatmap_female_" + appendToTitle, cmap="Reds")
    # matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
# def plot_heatmap(D, L):

#     D0 = D[:, L==0]
#     D1 = D[:, L==1]
    

#     hFea = {
#         0: 'Feature_0',
#         1: 'Feature_1',
#         2: 'Feature_2',
#         3: 'Feature_3',
#         4: 'Feature_4',
#         5: 'Feature_5',
#         6: 'Feature_6',
#         7: 'Feature_7',
#         8: 'Feature_8',
#         9: 'Feature_9',
#         }
#     # calculate Pearson correlation matrix
#     corr_matrix = np.corrcoef(D1)
    
#     fig, ax = plt.subplots(figsize=(12, 12))
#     plt.imshow(corr_matrix, cmap='seismic')
#     plt.colorbar()
    
#     # set tick labels for x and y axes
#     ax.set_xticks(np.arange(len(corr_matrix)))
#     ax.set_yticks(np.arange(len(corr_matrix)))
#     ax.set_xticklabels(np.arange(len(corr_matrix)))
#     ax.set_yticklabels(np.arange(len(corr_matrix)))
    
#     plt.title('Pearson Correlation Heatmap - \'Authentic fingerprints\' training set')    
        
        
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('heatmap_training_set_authentic.png',dpi=300)
#     plt.show()    

def compute_correlation(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def plot_correlations(DTR, title, cmap="Greys"):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    heatmap = sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    # fig.savefig("./images/" + title + ".svg")

# def Pearson_corr(D,L):

#     # Suddividere la matrice dei dati in base alle etichette di classe
#     X_target = D[:,L == 1]
#     X_non_target = D[:,L == 0]
    
#     # Calcolare il coefficiente di correlazione di Pearson per la classe target
#     corr_target = np.corrcoef(X_target, rowvar=False)
    
#     # Calcolare il coefficiente di correlazione di Pearson per la classe non target
#     corr_non_target = np.corrcoef(X_non_target, rowvar=False)
    
#     # Creare un grafico a matrice per la classe target
#     plt.matshow(corr_target, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title('Coefficienti di correlazione di Pearson - Classe target')
#     plt.colorbar()
#     plt.show()
    
#     # Creare un grafico a matrice per la classe non target
#     plt.matshow(corr_non_target, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title('Coefficienti di correlazione di Pearson - Classe non target')
#     plt.colorbar()
#     plt.show()





# # Esempio di utilizzo dei metodi:

# # Carica i dati dal file
# D, L = loadFile("nome_del_file.csv")

# # Plot delle features singolarmente
# plotSingle(D, L)

# # Plot delle features incrociate
# plotCross(D, L)

# # Calcolo delle statistiche
# DC = stats(D, L)
