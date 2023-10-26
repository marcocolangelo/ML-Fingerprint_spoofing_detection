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

x = np.array([2, 4, 8])  # Valori di x

# Larghezza delle barre
bar_width = 0.15

# Posizione dei gruppi di barre sull'asse x
bar_positions = np.arange(len(x))

# Dati dei quattro modelli

# model_pca6_val_ff_tn1 = np.array([0.27676229508196726, 0.25647540983606554, 0.2655327868852459])  # Dati del modello 1
# model_pcaNone_val_ff_tn1 = np.array([0.2830327868852459, 0.25209016393442624, 0.2633401639344263])  # Dati del modello 2
# model_pca6_eval_ff_tn1 = np.array([0.2365497737556561, 0.22133107088989443, 0.22468514328808445])  # Dati del modello 3
# model_pcaNone_eval_ff_tn1 = np.array([0.24493589743589744, 0.22012254901960784, 0.2279242081447964]) # Dati del modello 4
#----
# model_pca6_val_ff_tn2 = np.array([0.2752254098360656,0.25553278688524594,0.25772540983606557])  # Dati del modello 1
# model_pcaNone_val_ff_tn2 = np.array([0.27961065573770494,0.2602049180327869,0.2646311475409836 ])  # Dati del modello 2
# model_pca6_eval_ff_tn2 = np.array([0.2323623680241327,0.20769607843137255,0.22857088989441932])  # Dati del modello 3
# model_pcaNone_eval_ff_tn2 = np.array([0.24514328808446456,0.22937217194570136,0.23707013574660635]) # Dati del modello 4

#VALID diag-full
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.2705532786885246 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.24834016393442623 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2518032786885246 znorm: False

#EVAL diag-full
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.23216440422322776 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.22079939668174964 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.23183069381598795 znorm: False

#VALID diag-diag
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.31301229508196726 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.24834016393442623 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.24959016393442623 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.34209016393442626 znorm: False

# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.27676229508196726 znorm: False

# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.23522540983606557 znorm: False

#EVAL diag-diag
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.26730957767722474 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.22084087481146306 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.22010180995475112 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.2968306938159879 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.2194871794871795 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2182051282051282 znorm: False

#VALID diag-tied
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.32522540983606557 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.2658401639344262 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.2514754098360656 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.3214754098360656 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.2686885245901639 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2539549180327869 znorm: False

#EVAL diag-tied
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.2719871794871795 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.2250603318250377 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.21809200603318252 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.2745701357466063 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.22760180995475113 znorm: False
# GMM min_DCF mode_target=diag e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2057164404223228 znorm: False


#DIAG - FULL
model_pca6_val_df_tn2 = np.array([0.2655327868852459,0.25772540983606557,0.2496106557377049])  # Dati del modello 1
model_pcaNone_val_df_tn2 = np.array([0.2705532786885246,0.24834016393442623, 0.2518032786885246])  # Dati del modello 2
model_pca6_eval_df_tn2 = np.array([0.22909125188536955,0.21532051282051284,0.21589366515837105])  # Dati del modello 3
model_pcaNone_eval_df_tn2 = np.array([0.23216440422322776,0.22079939668174964,0.23183069381598795]) # Dati del modello 4

#DIAG - DIAG
model_pca6_val_dd_tn2 = np.array([0.31301229508196726,0.24834016393442623,0.24959016393442623 ])  # Dati del modello 1
model_pcaNone_val_dd_tn2 = np.array([0.34209016393442626,0.27676229508196726,0.23522540983606557 ])  # Dati del modello 2
model_pca6_eval_dd_tn2 = np.array([0.26730957767722474,0.22084087481146306,0.22010180995475112])  # Dati del modello 3
model_pcaNone_eval_dd_tn2 = np.array([0.2968306938159879,0.2194871794871795,0.2182051282051282]) # Dati del modello 4

#DIAG - TIED
model_pca6_val_dt_tn2 = np.array([0.32522540983606557,0.2658401639344262,0.2514754098360656])  # Dati del modello 1
model_pcaNone_val_dt_tn2 = np.array([0.3214754098360656,0.2686885245901639,0.2539549180327869 ])  # Dati del modello 2
model_pca6_eval_dt_tn2 = np.array([0.2719871794871795,0.2250603318250377,0.21809200603318252])  # Dati del modello 3
model_pcaNone_eval_dt_tn2 = np.array([0.2745701357466063,0.22760180995475113,0.2057164404223228]) # Dati del modello 4



# # Creazione del grafico  DIAG-FULL
# fig, ax = plt.subplots()
# bars1 = ax.bar(bar_positions, model_pca6_val_df_tn2, bar_width, color='red',label='PCA6 (VAL)')
# bars2 = ax.bar(bar_positions + bar_width,model_pcaNone_val_df_tn2, bar_width,color='orange', label='PCA None (VAL)')
# bars3 = ax.bar(bar_positions + 2*bar_width, model_pca6_eval_df_tn2, bar_width,color='red',hatch="/", label='PCA6 (EVAL)')
# bars4 = ax.bar(bar_positions + 3*bar_width, model_pcaNone_eval_df_tn2, bar_width, color='orange',hatch="/",label='PCA None (EVAL)')

# # Aggiunta di etichette, titolo e legenda
# ax.set_xlabel('components_not_target')
# ax.set_ylabel('DCF_min')
# ax.set_title('Target : Diag - Non Target : Full  - n_target = 2')
# ax.set_xticks(bar_positions + bar_width)
# ax.set_xticklabels(x)
# ax.legend()

# # Mostrare il grafico
# plt.show()

# # Creazione del grafico DIAG-DIAG
# fig, ax = plt.subplots()
# bars1 = ax.bar(bar_positions, model_pca6_val_dd_tn2, bar_width, color='red',label='PCA6 (VAL)')
# bars2 = ax.bar(bar_positions + bar_width,model_pcaNone_val_dd_tn2, bar_width,color='orange', label='PCA None (VAL)')
# bars3 = ax.bar(bar_positions + 2*bar_width, model_pca6_eval_dd_tn2, bar_width,color='red',hatch="/", label='PCA6 (EVAL)')
# bars4 = ax.bar(bar_positions + 3*bar_width, model_pcaNone_eval_dd_tn2, bar_width, color='orange',hatch="/",label='PCA None (EVAL)')

# # Aggiunta di etichette, titolo e legenda
# ax.set_xlabel('components_not_target')
# ax.set_ylabel('DCF_min')
# ax.set_title('Target : Diag - Non Target : Diag  - n_target = 2')
# ax.set_xticks(bar_positions + bar_width)
# ax.set_xticklabels(x)
# ax.legend()

# # Mostrare il grafico
# plt.show()

# Creazione del grafico DIAG-DIAG
fig, ax = plt.subplots()
bars1 = ax.bar(bar_positions, model_pca6_val_dt_tn2, bar_width, color='red',label='PCA6 (VAL)')
bars2 = ax.bar(bar_positions + bar_width,model_pcaNone_val_dt_tn2, bar_width,color='orange', label='PCA None (VAL)')
bars3 = ax.bar(bar_positions + 2*bar_width, model_pca6_eval_dt_tn2, bar_width,color='red',hatch="/", label='PCA6 (EVAL)')
bars4 = ax.bar(bar_positions + 3*bar_width, model_pcaNone_eval_dt_tn2, bar_width, color='orange',hatch="/",label='PCA None (EVAL)')

# Aggiunta di etichette, titolo e legenda
ax.set_xlabel('components_not_target')
ax.set_ylabel('DCF_min')
ax.set_title('Target : Diag - Non Target : Tied  - n_target = 2')
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(x)
ax.legend()

# Mostrare il grafico
plt.show()




# lr_pca6_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 6, piT = {pi_tilde} pi={pi} zscore={zscore}",lr_pca6))
# lr_pca6_glob_zscore.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 6, piT = {pi_tilde} pi={pi} zscore={zscore}",lr_pca6_zscore))
# #lr_pca7_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 7, piT = {piT} pi={pi}",lr_pca7))
# #lr_pca8_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 8, piT = {piT} pi={pi}",lr_pca8))
# #lr_pca9_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 9, piT = {piT} pi={pi}",lr_pca9))
# lr_pcaNone_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = No, piT = {pi_tilde} pi={pi} zscore={zscore}"",lr_pcaNone",lr_pcaNone))

# lamb  =np.logspace(-4, 2, num=7)

# lr_pcaNone = [0.2909445701357466, 0.2853092006033182, 0.2764027149321267, 0.2864762443438914, 0.28230957767722475, 0.2687160633484163, 0.29662141779788836]
# lr_pca6 = [0.25190422322775263, 0.2505184766214178, 0.25084087481146305, 0.2540082956259427, 0.25702865761689286, 0.26502828054298644, 0.2934445701357466]
# lr_pca6_zscore = [0.27620663650075417, 0.27039404223227753, 0.269643665158371, 0.28359162895927603, 0.3204769984917044, 0.3496436651583711, 0.35650829562594266]
# lr_pcaNone_zscore = [0.2836632730015083, 0.26483031674208146, 0.24457013574660635, 0.25703808446455506, 0.28994532428355957, 0.3271021870286576, 0.3367269984917044]

# val_lr_pca6 = [0.2718032786885246, 0.2718032786885246, 0.26305327868852457, 0.28616803278688524, 0.3027049180327869, 0.3280327868852459, 0.3560860655737705]
# val_lr_pca6_zscore = [0.26524590163934425, 0.26430327868852455, 0.2814344262295082, 0.32237704918032783, 0.34616803278688524, 0.3458606557377049, 0.3458606557377049]
# val_lr_pcaNone = [0.3230327868852459, 0.3114549180327869, 0.2889549180327869, 0.3046106557377049, 0.3105327868852459, 0.3176844262295082, 0.34987704918032786]

# plt.semilogx(lamb,val_lr_pca6, label = "PCA 6 (VAL)",linestyle='--',color = "blue")
# plt.semilogx(lamb,val_lr_pca6_zscore, label = "PCA 6 - ZSCORE (VAL)",linestyle='--',color = "orange")
# plt.semilogx(lamb,val_lr_pcaNone, label = "PCA None (VAL)",linestyle='--',color = "green")
# plt.semilogx(lamb,lr_pca6, label = "PCA 6 (EVAL)",color = "blue")
# plt.semilogx(lamb,lr_pca6_zscore, label = "PCA 6 - ZSCORE (EVAL)",color = "orange")
# plt.semilogx(lamb,lr_pcaNone, label = "PCA None (EVAL)",color = "green")
# # plt.semilogx(lamb,lr_pcaNone_zscore, label = "PCA None - ZSCORE (EVAL)")
# # plt.semilogx(lamb,lr_pca7, label = "PCA 7")
# # plt.semilogx(lamb,lr_pca8, label = "PCA 8")
# # plt.semilogx(lamb,lr_pca9, label = "PCA 9")
# # plt.semilogx(lamb,lr_pcaNone, label = "No PCA")
    
# plt.xlabel("Lambda")
# plt.ylabel("DCF_min")
# plt.legend()
# # if piT == 0.1:
# #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_min"
# # if piT == 0.33:
# #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_033"
# # if piT == 0.5:
# #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_medium"
# # if piT == 0.9:
# #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_max"
# titolo = "Quad Log Reg - EVAL and VAL confr"    
# plt.title(titolo)
# #plt.savefig(path)
# plt.show()
