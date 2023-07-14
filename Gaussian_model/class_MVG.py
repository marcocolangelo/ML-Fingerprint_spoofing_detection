from . import new_MVG_model as m
import numpy as np

class GaussClass:
    
    def __init__(self,mode,prior,Cfp,Cfn):
       
        self.means = 0
        self.eff_prior=0
        self.S_matrices = 0
        self.ll0 = 0
        self.ll1 = 0
        self.mode = mode
        self.eff_prior = (prior*Cfn)/(prior*Cfn + (1-prior)*Cfp)
        
        
    def name(self):
        return self.mode
        
    def train(self,DTR,LTR):
        if self.mode == "MVG":
            means,S_matrices,_ = m.MVG_model(DTR,LTR)
        elif self.mode == "NB":
            means,S_matrices,_ = m.MVG_model(DTR,LTR) #3 means and 3 S_matrices -> 1 for each class (3 classes)
            for i in range(np.array(S_matrices).shape[0]):
                S_matrices[i] = S_matrices[i]*np.eye(S_matrices[i].shape[0],S_matrices[i].shape[1])
        elif self.mode == "TCG":
            means,S_matrix = m.TCG_model(DTR,LTR) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
            #to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix 
            S_matrices = [S_matrix,S_matrix,S_matrix]
        elif self.mode == "TCGNB":
            means,S_matrix = m.TCG_model(DTR,LTR) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
            S_matrix = S_matrix * np.eye(S_matrix.shape[0],S_matrix.shape[1])
            #to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix 
            S_matrices = [S_matrix,S_matrix,S_matrix]
        else:
            print(f"Model variant {self.mode} not supported!")
           
        self.means = means
        self.S_matrices = S_matrices
        
    def compute_scores(self,DTE):
        
       llr = m.loglikelihoods(DTE,self.means,self.S_matrices, [1-self.eff_prior,self.eff_prior])
        
       return llr
    
    # def predict(self,Pc):
        
    #     log_sm_joint = np.array((self.ll0, self.ll1)) + np.log(Pc)
    #     #log_sm_joint_sol = np.load('logSJoint_MVG.npy')
        
    #     #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #     #be careful! These functions below return prediction labels yet! The Posterio probability 
    #     #computation is made inside the functions!
    #     log_pred = m.log_post_prob(log_sm_joint)
        
    #     return log_pred
        
    
    