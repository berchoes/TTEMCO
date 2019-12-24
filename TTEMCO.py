import pandas as pd 
import numpy as np
import random 
from matplotlib import pyplot as plt
import math
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


"""I couldn't use sklearn test-split library on my algorithm because it didn't create arrays from the
 split data. So I manually and randomly split the data on excel.It is split as %20 - %80."""

ds = pd.read_excel("cwurData.xlsx")
dt = pd.read_excel("test.xlsx") 


x = ds["world_rank"]
y = ds["quality_of_education"]
xTest = dt["world_rank"]
yTest = dt["quality_of_education"]



plt.plot(x,y,"k.") 
plt.plot(xTest,yTest,"k.")



# coefficients are assigned randomly between 0 and 1, as a matrix degree+1 rows and 1 column.
def coefficient(degree):  
    w = np.random.rand(degree+1,1)
    return w

def normalization(train):
    values = train.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized 

def y_pred(w,feature,deg):  # the equation of w0 + w1x + w2x^2 + ....                        
    h = np.ones((feature.shape[0],1)) # creating a unit matrix with 400 rows and 1 column.
    w = w.reshape(1,deg+1) 
    for i in range(0,feature.shape[0]):
        xarr = np.ones(deg+1)
        for j in range(0,deg+1):
            xarr[j] = pow(feature[i],j)
        xarr = xarr.reshape(deg+1,1)
        h[i] = float(np.matmul(w,xarr))
    h = h.reshape(feature.shape[0])
#     plt.plot(normalization(x),y,"k.") #   run with these 2 lines if you want to see
#     plt.plot(feature,h,"r-")          #   the change in every step.
    return h


def gradient_descent(w,lrate,epoch,h,feature,y,deg):
    feature = feature.flatten()
    for i in range(0,epoch):
        w[0] = w[0]- (lrate/feature.shape[0])* sum(h-y)
        for j in range(1,deg+1):
            w[j]=w[j]-(lrate/feature.shape[0])*sum((h-y)*pow(feature,j))
        h = y_pred(w,feature,deg)
#       RMSE= math.sqrt((1/feature.shape[0])* sum(np.square(h-y)))
        RMSE = math.sqrt(mean_squared_error(y,h))
#       print("RMSE: {} ".format(RMSE))
    w = w.reshape(1,deg+1)
    print("RMSE: {} ".format(RMSE))
    print("updated coefficients: {}" .format(w))
    plt.plot(normalization(x),y,"k.")
    plt.plot(feature,h,"r-")
    testRMSE(w,deg)


def testRMSE(w,deg):#takes the updated coefficents from gradient_descent method.Calculates the test RMSE.
    h=y_pred(w,normalization(xTest),deg)
    testRMSE = math.sqrt(mean_squared_error(yTest,h))
    print("Test RMSE: {} ".format(testRMSE))


def poly_regression(x,y,deg,lrate,epoch):
    w = coefficient(deg)
    print("coefficients: {}".format(w.reshape(deg+1)))
    h = y_pred(w,x,deg)
    gradient_descent(w,lrate,epoch,h,x,y,deg) 


    

