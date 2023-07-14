import numpy as np
import matplotlib.pyplot as plt

#this function compute the confusion_matrix confronting predictions on the classes and actual labels
def confusion_matrix(pred,LTE):
    nclasses = int(np.max(LTE))+1
    matrix = np.zeros((nclasses,nclasses))
    
    
    for i in range(len(pred)):
        matrix[pred[i],LTE[i]] += 1
        
    return matrix

#compute the posterior_probabilities adding Cfn and Cfp contributions
def binary_posterior_prob(llr,prior,Cfn,Cfp):
    new_llr = np.zeros(llr.shape);
    for i in range(len(llr)):
        #I just applied the llr formula in case of Cost addition
        #because of log properties I can rewrite it as sum between log_likelihood and prior and C log contribution
        new_llr[i] = llr[i] + np.log(prior*Cfn/((1-prior)*Cfp))
        
    return new_llr

#compute the un-normalized DCF
def binary_DCFu(prior,Cfn,Cfp,cm):
    FNR = cm[0,1]/(cm[0,1]+cm[1,1])
    FPR = cm[1,0]/(cm[1,0]+cm[0,0])
    
    DCFu = prior*Cfn*FNR + (1-prior)*Cfp*FPR
    
    return DCFu


#here we can plot the ROC plot which shows how the FPR and TPR change according to the current threshold value
def ROC_plot(thresholds,post_prob,LTE):
    FNR=[]
    FPR=[]
    for t in thresholds:
        #I consider t (and not prior) as threshold value for split up the dataset in the two classes' set
        pred = [1 if x >= t else 0 for x in post_prob]
        #I compute the confusion_matrix starting from these new results
        cm = confusion_matrix(pred, LTE)
            
        FNR.append(cm[0,1]/(cm[0,1]+cm[1,1]))
        FPR.append(cm[1,0]/(cm[1,0]+cm[0,0]))
        
    # plt.figure()
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    TPR=1-np.array(FNR)
    # plt.plot(FPR,TPR,scalex=False,scaley=False)
    # plt.show()
    
    return FPR,TPR
    
#compute Bayes error plot
def Bayes_plot(llr,LTE):
    effPriorLogOdds = np.linspace(-3, 3,21)
    DCF_effPrior = {}
    DCF_effPrior_min = {}
    print("In bayes_plot")
    #I try to compute DCF using several possible effPriorLogOdd values
    for p in effPriorLogOdds:
       
        #I compute the effPrior pi_tilde from the current effPriorLogOdds
        effPrior = 1/(1+np.exp(-p))

        #computation of the post_prob using pi_tilde as thresholds (not the optimal choice)
        post_prob = binary_posterior_prob(llr,effPrior,1,1)
        pred = [1 if x >=0 else 0 for x in post_prob]
        cm = confusion_matrix(pred, LTE)
        
        #computation of the not optimal DCF value bound to the specific pi_tilde choice
        dummy = min(effPrior,(1-effPrior))
        #now that we habe an effPrior which inglobes Cfn and Cfp contributions we can still recycle the old DCF formula but 
         #passing (effPrior,1,1) as parameters
        DCF_effPrior[p] = (binary_DCFu(effPrior, 1, 1, cm)/dummy)
        
     
        #loop over all the possible thresholds values to find the optimal t
        thresholds = np.sort(post_prob)
        tmp_DCF = []
        for t in thresholds:
            #I consider t (so not prior neither 0!!!) as threshold value for splitting up the dataset in the two classes' set
            pred = [1 if x >=t else 0 for x in post_prob]
            #I compute the confusion_matrix starting from these new results
            cm = confusion_matrix(pred, LTE)
            #I save the normalized DCU computed for each t value
            tmp_DCF.append((binary_DCFu(effPrior, 1, 1, cm)/dummy))
        
        #computation of the min DCF bound to the specific t and the specific pi_tilde
        DCF_effPrior_min[p] = (np.min(tmp_DCF))

        
    #log(pi/(1-pi)) on the x-axis    
        
    plt.plot(effPriorLogOdds, DCF_effPrior.values(), label='DCF', color='r')
    plt.plot(effPriorLogOdds, DCF_effPrior_min.values(), label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()
    
    return DCF_effPrior,DCF_effPrior_min

#should return DCF_norm NOT CALIBRATED
def DCF_norm_impl(llr,label,prior,Cfp,Cfn):
      #optimal bayes decision for inf-par binary problem 
        # infpar_llr = np.load("Data\commedia_llr_infpar_eps1.npy")
        # infpar_label =np.load("Data\commedia_labels_infpar_eps1.npy")
        # prior,Cfp,Cfn = (0.5,1,1)
        post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
        pred = [1 if x > 0 else 0 for x in post_prob]
        cm = confusion_matrix(pred, label)
        #observe that when prior increase = class1 predicted more frequently by the classifier
        #when Cfn increases classifiers will make more FP errors and less FN errors -> the opposite for the opposite case
        
      #binary task evaluation
        #DCFu doesn't allow us to comparing different systems so it's only the first step to compute DCF
        DCFu = binary_DCFu(prior,Cfn,Cfp,cm)
        #let's compute DCF now, normalizing DCFu with a dummy system DCFu
        dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
        #print("dummy DCF: "+str(dummy_DCFu))
        #print("DCFu : "+str(DCFu));
        DCF_norm = DCFu/dummy_DCFu
        
        return DCF_norm
    
def DCF_min_impl(llr,label,prior,Cfp,Cfn):
    post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
    thresholds = np.sort(post_prob)
    DCF_tresh = []
    dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
    #print("shape di post prob in DCF_min: "+str(post_prob.shape))
    
    #iteration over all the possible threshold values
    for t in thresholds:
        #I consider t (and not prior) as threshold value for split up the dataset in the two classes' set
        
        pred = [1 if x >= t else 0 for x in post_prob]
        #I compute the confusion_matrix starting from these new results
        cm = confusion_matrix(pred, label)
        #I save the normalized DCU computed for each t value
        DCF_tresh.append(binary_DCFu(prior, Cfn, Cfp, cm)/dummy_DCFu)
        
    #I choose the minimum DCF value and the relative threshold value
    min_DCF = min(DCF_tresh)
    t_min = thresholds[np.argmin(DCF_tresh)]
    
    #Observe how much loss due to poor calibration (the difference is clear if you compare DCF_norm with min_DCF)
    
    return min_DCF,t_min,thresholds
    

# if __name__ == "__main__":
    
#  #confusion matrix for the iris dataset 
#     D,L = lb5.load_iris()
#     (DTR,LTR),(DTE,LTE) = lb5.split_db_2to1(D, L)
#     pred_MVG = lb5.MVG_approach(DTR,LTR,DTE)
#     pred_Tied = lb5.TCG_approach(DTR,LTR,DTE)
    
#     MVG_cm = confusion_matrix(pred_MVG,LTE)
#     TCG_cm = confusion_matrix(pred_Tied,LTE)
    
#  #confusion matrix for the entire commedia dataset
#     commedia_ll = np.load("Data\commedia_ll_eps1.npy")
#     commedia_labels = np.load("Data\commedia_labels_eps1.npy")
#     commedia_pred = np.argmax(commedia_ll, axis=0)
#     commedia_cm = confusion_matrix(commedia_pred, commedia_labels)
    
#  #optimal bayes decision for inf-par binary problem 
#     infpar_llr = np.load("Data\commedia_llr_infpar_eps1.npy")
#     infpar_label =np.load("Data\commedia_labels_infpar_eps1.npy")
#     prior,Cfp,Cfn = (0.5,1,1)
#     infpar_post_prob = binary_posterior_prob(infpar_llr,prior,Cfn,Cfp)
#     infpar_pred = [1 if x > 0 else 0 for x in infpar_post_prob]
#     infpar_cm = confusion_matrix(infpar_pred, infpar_label)
#     #observe that when prior increase = class1 predicted more frequently by the classifier
#     #when Cfn increases classifiers will make more FP errors and less FN errors -> the opposite for the opposite case
    
#  #binary task evaluation
#     #DCFu doesn't allow us to comparing different systems so it's only the first step to compute DCF
#     DCFu = binary_DCFu(prior,Cfn,Cfp,infpar_cm)
#     #let's compute DCF now, normalizing DCFu with a dummy system DCFu
#     dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
#     DCF_norm = DCFu/dummy_DCFu
    
#     #minimum detection costs, we use the llr scores as possible threshold values to iterate over
    # thresholds = np.sort(infpar_post_prob)
    # DCF_tresh = []
    
    
    # #iteration over all the possible threshold values
    # for t in thresholds:
    #     #I consider t (and not prior) as threshold value for split up the dataset in the two classes' set
    #     pred = [1 if x >= t else 0 for x in infpar_post_prob]
    #     #I compute the confusion_matrix starting from these new results
    #     cm = confusion_matrix(pred, infpar_label)
    #     #I save the normalized DCU computed for each t value
    #     DCF_tresh.append(binary_DCFu(prior, Cfn, Cfp, cm)/dummy_DCFu)
        
    # #I choose the minimum DCF value and the relative threshold value
    # min_DCF = min(DCF_tresh)
    # t_min = thresholds[np.argmin(DCF_tresh)]
    
    # #Observe how much loss due to poor calibration (the difference is clear if you compare DCF_norm with min_DCF)
    # misc_loss = DCF_norm - min_DCF  
    
#     #ROC to see how much FPR and TPR change according to the chosen threshold value 
#     ROC_plot(thresholds,infpar_post_prob,infpar_label)
    
#     #bayes error plots = plotting the normalized costs as a function of an effective prior pi_tilde
#     #how much the costs change according to different (pi,Cfp,Cfn) ?
    
#     Bayes_DCF,Bayes_DCF_min = Bayes_plot(infpar_llr,infpar_label)
        

