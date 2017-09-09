# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyHampel
import pyper

if __name__ == "__main__":
    # create test data
    np.random.seed(2)
    t=np.arange(1024)
    x=np.arange(1024)
    z=np.random.normal(0,1,1024)
    x[0]=z[0]
    for i in range(1,1024,1):
        x[i] = 0.4*x[i-1] + 0.8*x[i-1]*z[i-1]+z[i]
        
    halfWindowSize=20
    ret={} 
    ret[0]=pyHampel.hampel(x,halfWindowSize,method="center")
    #set the center of window at target value
    ret[1]=ret[0]# just copy
    ret[2]=pyHampel.hampel(x,halfWindowSize,method="same")#same window
    ret[3]=pyHampel.hampel(x,halfWindowSize,method="ignore")# ignore in end
    ret[4]=pyHampel.hampel(x,halfWindowSize,method="nan")#set nan in end
    print (ret[0][0])#filtered data
    print (ret[0][1])#indices of outliers
    
    #compare pyhampel with R
    r=pyper.R()
    r("library(pracma);")
    r.assign("x",x)
    r("omad <- hampel(x, k=20);")
    retR=r.get("omad")
    
    
    fig,ax=plt.subplots(5,1,figsize=(8,8))
    ax[0].plot(t,x,"b",label="original data")
    for i in range(5):
        ax[i].plot(t,retR["y"],"orange",label="R")
        ax[i].plot(t,ret[i][0],"r.",label="python")
        ax[i].legend(loc="lower center")
    ax[0].set_title("overall")
    ax[1].set_title("center")
    ax[2].set_title("same")
    ax[3].set_title("ignore")
    ax[4].set_title("nan")
    ax[4].set_ylim(-8,8)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()
    diff = np.abs(retR["y"]-ret[0][0])
    #There is a difference at both end
    print ("center mode",np.sum(diff))
    #There is a same except both ends
    print ("center mode except both ends",np.sum(diff[20:-20]))

    #There is a same in ignore mode.
    diff = np.abs(retR["y"]-ret[3][0])
    print ("ignore mode",np.sum(diff))  
    
    
