import numpy
import scipy.optimize

def mcol(v):
    return v.reshape((v.size,1))
def mRow(v):
    return v.reshape((1,v.size))

class SVMClass:
    
    def __init__(self,K,C,piT):
        self.K = K
        self.C = C
        self.piT = piT
        self.Z = 0
        self.w_hat_star = 0
        self.f = 0
    
    def compute_lagrangian_wrapper(self,H):
        def compute_lagrangian(alpha):
            alpha = alpha.reshape(-1, 1)
            Ld_alpha = 0.5 * alpha.T @ H @ alpha - numpy.sum(alpha)
            gradient = H @ alpha - 1
            return Ld_alpha.item(), gradient.flatten()
        return compute_lagrangian
    
    def compute_H(self,D_,LTR):
        Z = numpy.where(LTR == 0, -1, 1).reshape(-1, 1)
        Gb = D_.T @ D_
        Hc = Z @ Z.T * Gb
        return Z, Hc
    
    def primal_solution(self,alpha,Z,x):
        return alpha*Z*x.T
    
    def compute_primal_objective(self,w_star,C,Z,D_):
        w_star = mcol(w_star)
        Z = mRow(Z)
        fun1= 0.5 * (w_star*w_star).sum()   
        fun2 = Z* numpy.dot(w_star.T, D_)
        fun3 = 1- fun2
        zeros = numpy.zeros(fun3.shape)
        sommatoria = numpy.maximum(zeros, fun3)
        fun4= numpy.sum(sommatoria)
        fun5= C*fun4
        ris = fun1 +fun5
        return ris
    
    def train(self,DTR,LTR):
        nf = DTR[:,LTR==0].shape[1]
        nt = DTR[:,LTR==1].shape[1]
        emp_prior_f = nf/ DTR.shape[1]
        emp_prior_t = nt/ DTR.shape[1]
        Cf = self.C * self.piT / emp_prior_f
        Ct = self.C * self.piT / emp_prior_t
        
        K_row = numpy.ones((1, DTR.shape[1])) * self.K
        D_ = numpy.vstack((DTR, K_row))
        #print(D_)
        self.Z,H_=self.compute_H(D_,LTR)
        compute_lag=self.compute_lagrangian_wrapper(H_)
        bound_list=[(-1,-1)]*DTR.shape[1]
        
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bound_list[i] = (0,Cf)
            else:
                bound_list[i] = (0,Ct)
        
        
        #factr=1 requires more iteration but returns more accurate results
        (alfa,self.f,d)=scipy.optimize.fmin_l_bfgs_b(compute_lag,x0=numpy.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
        w_hat_star = (mcol(alfa)* self.Z * D_.T).sum(axis=0)
        self.w_hat_star = w_hat_star
        
    def compute_scores(self,DTE):
        K_row = numpy.ones((1, DTE.shape[1])) * self.K
        D_ = numpy.vstack((DTE, K_row))
        score = numpy.dot(self.w_hat_star, D_)
        
        predicted_labels = numpy.where(score > 0, 1, 0)
        error = (predicted_labels != DTE).mean()
        #accuracy,error=compute_accuracy_error(predicted_labels,mRow(LTE))
        primal_obj=self.compute_primal_objective(self.w_hat_star,self.C,self.Z,D_)
        dual_gap=primal_obj+self.f
        print("K:",self.K)
        print("C:",self.C)
        print("primal loss: ",primal_obj)
        print("dual loss: ", -self.f)
        print("dual gap: ",dual_gap)
        print(f'Error rate: {error * 100:.1f}%')


# if __name__=='__main__':
     
#      D, L = load_iris_binary()
#      (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
      # for K in [1,10]:
      #   for C in [0.1,1.0,10.0]:
      #       K_row = numpy.ones((1, DTR.shape[1])) * K
      #       D_ = numpy.vstack((DTR, K_row))
      #       #print(D_)
      #       Z,H_=compute_H(D_,LTR)
      #       compute_lag=compute_lagrangian_wrapper(H_)
      #       bound_list=[(0,C)]*LTR.size
      #       #factr=1 requires more iteration but returns more accurate results
      #       (alfa,f,d)=scipy.optimize.fmin_l_bfgs_b(compute_lag,x0=numpy.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
      #       w_hat_star = (mcol(alfa)* Z * D_.T).sum(axis=0)
      #       w_star=w_hat_star[:-1]
      #       b_star=w_hat_star[-1]
            
      #       K_row2 = numpy.ones((1, DTE.shape[1])) * K
      #       D_2 = numpy.vstack((DTE, K_row2))
      #       score = w_hat_star @ D_2
            
      #       predicted_labels = numpy.where(score > 0, 1, 0)
      #       error = (predicted_labels != LTE).mean()
      #       #accuracy,error=compute_accuracy_error(predicted_labels,mRow(LTE))
      #       primal_obj=compute_primal_objective(w_hat_star,C,Z,D_)
      #       dual_gap=primal_obj+f
      #       print("K:",K)
      #       print("C:",C)
      #       print("primal loss: ",primal_obj)
      #       print("dual loss: ", -f)
      #       print("dual gap: ",dual_gap)
      #       print(f'Error rate: {error * 100:.1f}%')
             


             

     
     