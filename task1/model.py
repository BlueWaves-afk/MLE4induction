'''
Author: Tom Mathew
Date: 26th March, 2025

'''
import pandas as pd
import numpy as np

'''
Dataprocessing pipeline;
-data segregation, select usefull features, using correlation matrix
-normalization, to a standard range, only to be done to feature values(X), as we want to predict y
-shuffling
-data splitting, into train and test
'''
class DataProcessing: #writing a dataprocessing class similar to commonly written regression models.
    def __init__(self,split_fraction=0.80):
        self.split_fraction = split_fraction
        self.no_features = 0
        self.no_examples = 0
        self.X=None
        self.y=None
        self.minmax = None

    def train_test_split(self):
        train_volume = int(self.split_fraction*(self.no_examples))
        
        return self.X[0:train_volume,:], self.X[train_volume:,:], self.y[0:train_volume],self.y[train_volume:]
    
    def get_minmax(self):#getting min and max value for each feature
        for i in range(0,self.no_features):
            self.min_max[i,0] = np.min(self.X[:,i])
            self.min_max[i,1] = np.max(self.X[:,i])

    def normalize(self,x): #get numpy array as input
        for i in range(0,x.shape[1]):
            x[:,i] = (x[:,i]- self.min_max[i,0])/(self.min_max[i,1]-self.min_max[i,0])
        return x
    
    def shuffle(self,df): #shuffle pandas dataframe, only first axis(the rows) are shuffled
        return df.sample(frac=1)
    
    def fit(self, csv_file='Boston House Prices.csv'):
        df = pd.read_csv(csv_file, sep=',')
        df = self.shuffle(df)
        self.X = df.iloc[:,0:-1].to_numpy()
        self.y = df.iloc[:,-1].to_numpy()
        self.no_examples = self.X.shape[0]
        self.no_features = self.X.shape[1]
        self.min_max = np.zeros([self.no_features,2])#in each row of min_max first element is min, second is max
        self.get_minmax()
        self.X = self.normalize(self.X)

        return self

'''
Multivariate LinearRegressionModel;
-works on the formula of straight line, as we are fitting a straight line to estimate median value
-y = mx +c, where c is bias term, and m is weights
-y, m and x are vector terms, in expanded format; for 1 example; y = c*(1)+ m1x1 + m2x2 + ... +mnxn
-taking loss function as MSE, our GOAL is to minimize the loss, so we should aim to get gradiant = 0
-then we apply gradient descent, where using our gradient we slowly(with small learning rate to avoid large fluctuations)
-increment or push the loss towards minimum value
'''
class RegressionModel:
    def __init__(self,lr=0.05,n_iters = 200):
        self.lr = lr
        self.iters = n_iters
        self.no_features= 0
        self.no_examples = 0
        self.weights = None #stores weights for each feature[no_features+1, 1], extra coloumn to account for bias term
    def loss(self, y , pred): #implementing MSE as loss function
        return (1/self.no_examples)*np.sum((y-pred)**2) 
        
    def fit(self, X, y):
        self.no_examples,self.no_features = X.shape
 
        X = np.concatenate((np.ones((self.no_examples,1)),X),axis=1) #add extra coloumn of ones at beginning to account for bias term
        self.weights = np.random.rand(self.no_features+1,1)
        print(self.weights.shape)
        for i in range(self.iters): #performing gradient descent
            
            pred = np.matmul(X,self.weights) # X= [examples, feature+1], weights = [features+1, 1] = [examples, 1]
            
            error = pred-y[:,np.newaxis]
            
            gradient = (1/self.no_examples)*(np.matmul(X.T,error))
            
            self.weights = self.weights - (self.lr*gradient)
            print(f'iteration {i+1}, loss: {self.loss(y,pred)}')
        return self.weights
        
    def predict(self, x):
        x = np.squeeze(x)[np.newaxis,:]
        
        
        x = np.concatenate((np.ones(x.shape[0])[np.newaxis,:],x),axis=1)
        
        return np.dot(x,self.weights)
    
dp = DataProcessing(split_fraction=0.90)
dp.fit(csv_file='Boston House Prices.csv')
X_train, X_test, y_train, y_test = dp.train_test_split()

model = RegressionModel(lr=0.001,n_iters=1000)
model.fit(X_train, y_train)

print(model.predict(X_test[0,:]))
print(y_test[0])
