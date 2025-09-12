from sklearn.model_selection import KFold
from math import sqrt
import numpy as np
import mlflow
#Class
class LinearRegression(object):
    kfold = KFold(n_splits=5)

    #init function
    def __init__(self, regularization, lr, method, momentum, init_theta, num_epochs=500, bs=50, cv= kfold):
    # def __init__(self, regularization, lr, method, num_epochs=500, bs=50, cv= kfold):
       self.lr = lr
       self.regularization = regularization
       self.momentum = momentum
       self.init_theta = init_theta
       self.num_epochs = num_epochs
       self.bs = bs
       self.method = method
       self.cv = cv
       self.prev_step = 0 # Initialize velocity for momentum as None; will set it later
    
    #mse
    def mse(self,ytrue,ypred):
        # MSE = (ytruee-ypred)**2.sum()/
        return ((ytrue-ypred)**2).sum()/ytrue.shape[0]
    
    #r2 score function
    def r2_score(self, ytrue, ypred):
        return 1 - ((((ypred-ytrue)**2).sum())/(((ytrue-ytrue.mean())**2).sum()))
    
    #fit Function
    def fit(self, X_train, y_train):

        #create a list of keeping kfold scores
        self.kfold_scores = []

        #variable to know our loss is not improving anymore
        # if the new loss !< old loss, then we stop the training process. (0.01 -> tolerance)
        self.val_loss_old = np.infty

        #cross validation
        for fold, (train_idx,val_idx) in enumerate(self.cv.split(X_train)):
        
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]# Cross validaiton ends here!!


            # print("X_cross_train",X_cross_train.shape, "y_cross_train",y_cross_train.shape, "X_cross_val",X_cross_val.shape, "y_cross_val",y_cross_val.shape)


            # Initializeing theta 
            # method 1: zero
            if self.init_theta =='zero':
                self.theta = np.zeros(X_cross_train.shape[1]) #initialize theta into 0

            # Xavier mthod for initialization
            if self.init_theta == 'xavier':
                
                #m = number of samples
                m = X_cross_train.shape[1]

                #calculate the range of the weights
                lower, upper = -(1.0/np.sqrt(m)),(1.0/np.sqrt(m))
                # print(lower,upper)

                numbers = np.random.uniform(lower,upper,size=m)
                scaled = lower + numbers*(upper-lower)
                self.theta = scaled
                # print(scaled)



            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                for epoch in range(self.num_epochs):

                    #Shuffle the data a little bit so that order of data doesn't affect the learning
                    perm = np.random.permutation(X_cross_train.shape[0])

                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]

                    if self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0],self.bs):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.bs, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.bs]
                            train_loss = self._train(X_method_train,y_method_train) # train_loss is theta here.

                    elif self.method =='sto':
                        for batch_idx in range(0, X_cross_train.shape[0],1):
                            X_method_train = X_cross_train[batch_idx].reshape(1,-1)
                            y_method_train = y_cross_train[batch_idx].reshape(1, )
                            train_loss = self._train(X_method_train,y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train,y_method_train)

                    # Training error for each epoch into mlflow
                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                    yhat_val = self.predict(X_cross_val)

                    # Validation loss for each epoch into mlflow
                    val_loss_new = self.mse(y_cross_val,yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)

                    # r2 score for each epoch into mlflow
                    val_r2 = self.r2_score(y_cross_val,yhat_val)
                    mlflow.log_metric(key="val_r2", value=val_r2, step=epoch)

                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        print(f"Early Stopping at Epoch {epoch}")
                        break

                    self.val_loss_old = val_loss_new

                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: MSE: {val_loss_new}")
                print(f"Fold {fold}:R2: {val_r2}")


    #train
    def _train(self,X,y):
        #1. predict
        yhat = self.predict(X)

        #2. calculate the grad
        grad = (1/X.shape[0]) * X.T @ (yhat - y) 
        if self.regularization:
            grad += self.regularization.derivation(self.theta)

        self.update(grad)
        
        return self.mse(y,yhat)
    
    def update(self,grad):
        step = self.lr * grad
        self.theta = self.theta - step + self.momentum * self.prev_step
        self.prev_step = step
        return

    # predict
    def predict(self,X):
        #shape of return value(predicted value) -> (m,n) @ (n,) =(m,)
        return X @ self.theta  # @ is matrix multiplication

    # get theta
    def _coef(self):
        return self.theta[1:]
    
    # get intercept or bias
    def _intercept(self):
        return self.theta[0]
    
     # Feature Importance 
    def feature_importance(self):
        return self.theta[1:]
    
class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class NormalPenalty:

    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): 
        return 0
        
    def derivation(self, theta):
        return 0
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio=0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
    #  def __init__(self, regularization, lr, method, momentum, init_theta, mlflow_params, num_epochs=500, bs=50, cv= kfold):
class Lasso(LinearRegression):
    
    def __init__(self, lr, method, momentum, init_theta, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method,momentum, init_theta)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, momentum, init_theta,l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, momentum, init_theta)

class Normal(LinearRegression):
    
    def __init__(self, method, lr, momentum, init_theta,l):
        self.regularization = NormalPenalty(l)
        super().__init__(self.regularization, lr, method, momentum, init_theta)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, l, momentum, init_theta, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method,momentum, init_theta)




