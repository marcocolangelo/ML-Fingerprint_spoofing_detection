import Gaussian_model.lab04_part2 as lb04
import numpy as np
import scipy as sc


def MVG_model(D,L):
    c0 = []
    c1 = []
    
    means = []
    S_matrices = []
   
    print(L)
    for i in range(D.shape[1]):
        if L[i] == 0:
            print("entra in L[i]==0")
            c0.append(D[:,i])
        else :
            print("entra in L[i]==1")
            c1.append(D[:,i])
       


    c0 = (np.array(c0)).T
    c1 = (np.array(c1)).T        
    
    print(c0)
    
    c0_cent = lb04.centerData(c0)
    c1_cent = lb04.centerData(c1)
   
    
    #you can find optimizations for this part in Lab03
    
    S_matrices.append(lb04.createCenteredCov(c0_cent)) 
    S_matrices.append(lb04.createCenteredCov(c1_cent))
            
    
    means.append(lb04.vcol(c0.mean(1)))
    means.append(lb04.vcol(c1.mean(1)))
   
    
    
    return means,S_matrices,(c0.shape[1],c1.shape[1])

def TCG_model(D,L):

    S_matrix = 0
    means,S_matrices,cN = MVG_model(D, L)
    
    cN = np.array(cN)
    
    S_matrices = np.array(S_matrices)
    
    D_cent = lb04.centerData(D)
    
    for i in range(cN.shape[0]):
        
        S_matrix += cN[i]*S_matrices[i]  
    
    S_matrix /=D.shape[1]
    
    return means,S_matrix
    

def loglikelihoods(DTE,means,S_matrices):
    ll0 = []
    ll1 = []
    
    
    for i in range(DTE.shape[1]):
            ll0.append(lb04.loglikelihood(DTE[:,i:i+1] , means[0], S_matrices[0]))
        
            ll1.append(lb04.loglikelihood(DTE[:,i:i+1], means[1], S_matrices[1]))
            
              
    
    return np.array((ll0, ll1))


def posterior_prob(SJoint):
    
    # Calcola le densità marginali sommando le probabilità congiunte su tutte le classi
    SMarginal = lb04.vrow(SJoint.sum(axis=0))
    
    # Calcola le probabilità posteriori di classe dividendo le probabilità congiunte per le densità marginali
    SPost = SJoint / SMarginal
    
    # Calcola l'array delle etichette previste utilizzando il metodo argmax con la parola chiave axis
    pred = np.argmax(SPost, axis=0)
          
    return pred

def log_post_prob(log_SJoint):
        

    log_SMarginal = lb04.vrow(sc.special.logsumexp(log_SJoint,axis=0))
    #print(np.abs(log_SMarginal - log_SMarginal_sol).max())
    
    #print(log_SMarginal.shape)
    #print(log_SJoint.shape)
    log_SPost = log_SJoint - log_SMarginal
    #log_SPost_sol = np.load('logPosterior_MVG.npy')
    
    log_pred = np.argmax(log_SPost,axis=0)
        
    return log_pred
    

def evaluation(pred,LTE) : 
    
    
    mask = (pred==LTE)
    
    mask= np.array(mask,dtype=bool)
    
    corr = np.count_nonzero(mask)
    tot = LTE.shape[0]
    
    
    acc = float(corr)/tot
    
    
    
    return acc,tot-corr

def MVG_approach(D,L,DTE,Pc):
    #remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes
    
    #using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means,S_matrices,_ = MVG_model(D,L) #3 means and 3 S_matrices -> 1 for each class (3 classes)
    
    #we create a NxNc matrix with the log-likelihoods elements
    #each row represents a class and each column represents a sample
    #so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    #Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc

    log_sm_joint = log_score_matrix + np.log(Pc)

    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    #simple function to evaluate the accuracy of our model
    #acc,_ = evaluation(pred,LTE)  
    #acc_2,_=evaluation(log_pred,LTE)
    #inacc = 1-acc
    
    return log_pred



def NB_approach(D,L,DTE,Pc):
    #remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes
    
    #using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means,S_matrices,_ = MVG_model(D,L) #3 means and 3 S_matrices -> 1 for each class (3 classes)
    
    for i in range(np.array(S_matrices).shape[0]):
        S_matrices[i] = S_matrices[i]*np.eye(S_matrices[i].shape[0],S_matrices[i].shape[1])
    
    #we create a NxNc matrix with the log-likelihoods elements
    #each row represents a class and each column represents a sample
    #so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    #Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc

    log_sm_joint = log_score_matrix + np.log(Pc)

    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterior probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    #simple function to evaluate the accuracy of our model
    # acc,_ = evaluation(pred,LTE)  
    # acc_2,_=evaluation(log_pred,LTE)
    # inacc = 1-acc
    
    return log_pred

def TCG_approach(D,L,DTE,Pc):
    
    
    means,S_matrix = TCG_model(D,L) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
    
    #to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix 
    S_matrices = [S_matrix,S_matrix,S_matrix]
    
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    #Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc

    
    log_sm_joint = log_score_matrix + np.log(Pc)

    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    
    
    #simple function to evaluate the accuracy of our model
    # acc,_ = evaluation(pred,LTE)  
    # acc_2,_=evaluation(log_pred,LTE)
    # inacc = 1-acc

    return log_pred

def TCNBG_approach(D,L,DTE,Pc):
    means,S_matrix = TCG_model(D,L) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
    
    S_matrix = S_matrix * np.eye(S_matrix.shape[0],S_matrix.shape[1])
    #to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix 
    S_matrices = [S_matrix,S_matrix,S_matrix]
    
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    #Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc
   
    
    log_sm_joint = log_score_matrix + np.log(Pc)
   
    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    
    #simple function to evaluate the accuracy of our model
    #acc,_ = evaluation(pred,LTE)  
    #acc_2,_=evaluation(log_pred,LTE)
   # inacc = 1-acc

    return log_pred
    

def LOO(D,L):
    
        MVG_pred = []
        NB_pred=np.zeros(D.shape[1])
        TCG_pred=np.zeros(D.shape[1])
        TCNBG_pred=np.zeros(D.shape[1])
    
        for i in range(D.shape[1]):
             #K-fold Leave One Out method adopted to slit up training set into training set and validation 
             DTE = D[:,i:i+1]  #1 sample of the dataset used for testing and the other for testing
             
             #it deletes a single sample at time
             DTR = np.delete(D,i,axis=1)
             LTR = np.delete(L,i)
             LTE = L[i:i+1]
             
             pred_LOO_MVG = MVG_approach(DTR,LTR,DTE)[0]
             MVG_pred.append(pred_LOO_MVG)
             
             
             pred_LOO_NB = NB_approach(DTR,LTR,DTE)
             NB_pred[i] = pred_LOO_NB
             
             pred_LOO_TCG = TCG_approach(DTR, LTR, DTE)
             TCG_pred[i] = pred_LOO_TCG
             
             pred_LOO_TCNBG = TCNBG_approach(DTR,LTR, DTE)
             TCNBG_pred[i] = pred_LOO_TCNBG
        
        
        return np.array(MVG_pred),np.array(NB_pred),np.array(TCG_pred),np.array(TCNBG_pred)
    
    
# if __name__ == "__main__":
   
    
#     pred_MVG = MVG_approach(DTR,LTR,DTE)
    
#     #for so few data we can apply the same code as used for MVG approach and make the S_matrices diagonal using a np.eye()
#     pred_Naive_Bayes = NB_approach(DTR,LTR,DTE)
    
#     pred_Tied_cov_Gauss = TCG_approach(DTR,LTR,DTE)
    
#     #accuracy evaluation system
#     print(evaluation(pred_MVG,LTE)[0]*100)
#     print(evaluation(pred_Naive_Bayes,LTE)[0]*100)
#     print(evaluation(pred_Tied_cov_Gauss,LTE)[0]*100)
    
    
    
#     #let's try with Leave One Out EVALUATION system (Leave One Out variant)
    
#     #NOT SURE THIS IS THE RIGHT SOLUTION EXPECIALLY ABOUT WHAT KIND OF DATASET WE NEED TO USE!!!!!
#     MVG_pred,NB_pred,TCG_pred,TCNBG_pred = LOO(DTR,LTR)
    
#     MVG_acc,MVG_err = evaluation(MVG_pred, L)
    
#     NB_acc,NB_err = evaluation(NB_pred,L)
    
#     TCG_acc,TCG_err = evaluation(TCG_pred, L)
    
#     TCNBG_acc,TCNBG_err = evaluation(TCNBG_pred, L)
    
#     print("LLO_MVG_err_ratio: ",(1-MVG_acc)*100)
#     print("LLO_NB_err_ratio: ",(1-NB_acc)*100)
#     print("LLO_TCG_err_ratio: ",(1-TCG_acc)*100)
#     print("LLO_TCNBG_err_ratio: ",(1-TCNBG_acc)*100)

    
    
    
   
    
    
    
    




