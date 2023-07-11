from load import *
from features_analysis.feature_analysis import *
from features_analysis.PCA import *
from features_analysis.LDA import *
from Gaussian_model.new_MVG_model import *
from Gaussian_model.MVG_density import *
from Gaussian_model.class_MVG import *
from evaluation_functions.evaluation import *
from validation.k_fold import *
from logistic_regression.logreg import logRegClass,quadLogRegClass
from svm.svm import SVMClass
import numpy as np
import matplotlib.pyplot as plt
C_values = [1.000000000000000082e-05,
1.000000000000000048e-04,
1.000000000000000021e-03,
1.000000000000000021e-02,
1.000000000000000056e-01,
1.000000000000000000e+00,
1.000000000000000000e+01,
1.000000000000000000e+02
]
svm_pca6 =[1, 0.7463319672131148, 0.7967418032786885, 0.6436270491803279, 0.4895901639344262, 0.48116803278688525, 0.47897540983606557, 0.5139344262295082]
svm_pca8 = [1, 0.7416598360655737, 0.7804508196721311, 0.6342213114754098, 0.48331967213114757, 0.4874385245901639, 0.4899180327868853, 0.4979918032786885]
svm_pcaNone = [1.0, 0.7426024590163934, 0.8005122950819672, 0.6307786885245902, 0.5027049180327869, 0.4783606557377049, 0.47993852459016395, 0.4911475409836066]
plt.semilogx(C_values,svm_pca6, label = "PCA 6")
plt.semilogx(C_values,svm_pca8, label = "PCA 8")
plt.semilogx(C_values,svm_pcaNone, label = "No PCA")
    
plt.xlabel("C")
plt.ylabel("DCF_min")
plt.legend()
plt.title()
plt.show()