# -*- coding: utf-8 -*-
import numpy as np

#python
# 1D hampel filter
# 
# purpose
#  Outlier detection and remove

def hampel(x,k,method="center",thr=3):
    #Input
    # x       input data
    # k       half window size (full 2*k+1)
    # mode    about both ends
    #         str {‘center’, 'same','ignore',‘nan’}, optional
    #
    #           center  set center of window at target value
    #           same    always same window size
    #           ignore  set original data
    #           nan     set non
    #           
    # thr     threshold (defaut 3), optional
    #Output
    # newX    filtered data
    # omadIdx indices of outliers
    arraySize=len(x)
    idx=np.arange(arraySize)
    newX=x.copy()
    omadIdx=np.zeros_like(x)
    for i in range(arraySize):
        mask1=np.where( idx>= (idx[i]-k) ,True, False)
        mask2=np.where( idx<= (idx[i]+k) ,True, False)
        kernel= np.logical_and(mask1,mask2)
        if method=="same":
            if i<(k):
                kernel=np.zeros_like(x).astype(bool)
                kernel[:(2*k+1)]=True
            elif i>= (len(x)-k):
                kernel=np.zeros_like(x).astype(bool)
                kernel[-(2*k+1):]=True
        #print (kernel.astype(int))
        #print (x[kernel])
        med0=np.median(x[kernel])
        #print (med0)
        s0=1.4826*np.median(np.abs(x[kernel]-med0))
        if np.abs(x[i]-med0)>thr*s0:
             omadIdx[i]=1
             newX[i]=med0
    
    if method=="nan":
        newX[:k]=np.nan
        newX[-k:]=np.nan
        omadIdx[:k]=0
        omadIdx[-k:]=0
    elif method=="ignore":
        newX[:k]=x[:k]
        newX[-k:]=x[-k:]
        omadIdx[:k]=0
        omadIdx[-k:]=0        
        
    return newX,omadIdx.astype(bool)


if __name__ == "__main__":
    np.random.seed(2)
    t=np.arange(1024)
    x=np.arange(1024)
    z=np.random.normal(0,1,1024)
    x[0]=z[0]
    for i in range(1,1024,1):
        x[i] = 0.4*x[i-1] + 0.8*x[i-1]*z[i-1]+z[i]

    #ret=hampel(x,20,"same")
    ret=hampel(x,20,"center")        
    #ret=hampel(x,20)
    print (ret[0])#filtered data
    print (ret[1])#indices of outliers

    
