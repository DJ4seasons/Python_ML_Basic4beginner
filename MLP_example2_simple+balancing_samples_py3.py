"""
Test Multi-layer Perceptron (Feed-forward) Neural Network

Input: OLR data (NOAA CDR Monthly; box mean)
Target: 3-month forecast of El Nino or La Nina

By Daeho Jin
2021.01.31
---

Test over-sampling

"""


import sys
import os.path
import numpy as np
from datetime import datetime
import MLP_functions as fns
import matplotlib.pyplot as plt

def main():
    ### Parameters
    tgt_dates= (datetime(1979,1,1),datetime(2020,12,31))
    tgt_dates_str= ['{}{:02d}'.format(tgt.year, tgt.month) for tgt in tgt_dates]
    nyr, mon_per_yr= tgt_dates[1].year-tgt_dates[0].year+1, 12
    nmons= nyr*mon_per_yr
    indir = './Data/'

    ### Read OLR
    infn_olr= indir+'olr-monthly_v02r07_{}_{}.nc'.format(*tgt_dates_str)
    vars = ['time','lat','lon','olr']
    olr_data= fns.read_nc_data(infn_olr,vars)
    ## Check temporal dimension
    t0= olr_data['time'][0]
    if tgt_dates[0].year != t0.year or tgt_dates[0].month != t0.month or nmons!=olr_data['time'].shape[0]:
        print('Temporal dimension is inconsistent')
        sys.exit()
    else:
        print('OLR data is read')

    ## OLR data is degraded for box mean, from 15S to 15N, every 30deg longitude
    lon0,dlon,nlon= olr_data['lon'][0],olr_data['lon'][1]-olr_data['lon'][0],olr_data['lon'].shape[0]
    lat0,dlat,nlat= olr_data['lat'][0],olr_data['lat'][1]-olr_data['lat'][0],olr_data['lat'].shape[0]
    latidx= [fns.lat_deg2y(lat,lat0,dlat) for lat in [-15,15]]
    nlat2= latidx[1]-latidx[0]
    nlon_30deg= int(np.rint(30/(dlon)))
    olr= olr_data['olr'][:,latidx[0]:latidx[1],:]
    olr= olr.reshape([nmons,nlat2,nlon//nlon_30deg,nlon_30deg]).mean(axis=(1,3)).reshape([nyr,mon_per_yr,-1])
    print('After box-averaging, ',olr.shape)

    ### Read Nino3.4 values
    infn_nn34= indir+'nino3.4.txt'
    nn34= fns.read_nn34_text(infn_nn34,tgt_dates)
    if nn34.shape!=(nyr,mon_per_yr):
        print('Temporal dimension is inconsistent')
        sys.exit()
    else:
        print('Nino3.4 data is read', nn34.shape)

    ### Remove seanal cycle
    olr_mm= olr.mean(axis=0)
    olr= olr-olr_mm[None,:]
    olr= olr.reshape([-1,olr.shape[-1]])

    nn34_mm= nn34.mean(axis=0)
    nn34= nn34-nn34_mm[None,:]
    nn34= nn34.reshape(-1)
    print('Seasonal cycle is removed')
    #from scipy import stats
    #print(stats.describe(olr))
    #print(stats.describe(nn34))

    ### Matching time for 3-mon prediciton of nino3.4
    olr, nn34 = olr[:-3,:], nn34[3:]

    ###------- ML part -------###
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report,confusion_matrix

    ### Input data scaling
    #scaler= StandardScaler() #MinMaxScaler(feature_range=(-1,1)) #
    #X = scaler.fit_transform(olr)
    X= olr/np.std(olr,ddof=1)

    ### Label encoding
    one_std= 0.6 #np.std(nn34,ddof=1)
    y = np.digitize(nn34,[-one_std,one_std]) #-1
    nn_label=['La Ni\u00F1a','Neutral','El Ni\u00F1o']

    print("\nX data range: {:.3f} to {:.3f}".format(X.min(), X.max()))
    print("y data distribution")
    for lab,val in zip(nn_label,np.unique(y)):
        print("{:3d}={:10s}: {:5d}".format(val,lab,(y==val).sum()))

    ### Train-Test split: recent 5 years for test, and no shuffle for test recent years
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 60/y.shape[0],shuffle=False)

    ### Balancing the population of labels
    def generate_new_samples(X1,y1,tgt_size):
        """Simple duplication to increase samples"""
        sample_size= y1.shape[0]
        size2add= tgt_size- sample_size
        idx2copy= np.random.choice(sample_size,size2add)  ## replacement=True in default
        return X1[idx2copy,:],y1[idx2copy]

    tgt_size= (y_train==0).sum()
    ## La Nina case
    idx= y_train==0
    X_add1, y_add1= generate_new_samples(X_train[idx,:],y_train[idx],tgt_size)
    ## El Nino case
    idx= y_train==2
    X_add2, y_add2= generate_new_samples(X_train[idx],y_train[idx],tgt_size)
    ## Concatenate
    print("\nBefore balancing")
    print(X_train.shape, y_train.shape)
    X_train, y_train= np.concatenate((X_add1,X_add2,X_train),axis=0), np.concatenate((y_add1,y_add2,y_train))
    print("After balancing")
    print(X_train.shape, y_train.shape)
    print("y data distribution")
    for lab,val in zip(nn_label,np.unique(y_train)):
        print("{:3d}={:10s}: {:5d}".format(val,lab,(y_train==val).sum()))

    ### Shuffle train data before start
    #rg = np.random.default_rng(seed=1) ## Numpy version>=1.17
    rg= np.random.RandomState(seed=1)  ## Numpy version<1.17
    shf_idx= np.arange(y_train.shape[0],dtype=int)
    rg.shuffle(shf_idx)
    X_train, y_train= X_train[shf_idx,:], y_train[shf_idx]
    print('\nShuffling of train data is done.')

    ### MLP (Default: activation='relu', solver='adam',alpha=0.0001,learning_rate='constant')
    mlp= MLPClassifier(hidden_layer_sizes= (10,),random_state=1, max_iter=4999)

    mlp.fit(X_train, y_train)
    print("\nConverged at {} iterations\n".format(mlp.n_iter_))
    plt.plot(mlp.loss_curve_)
    plt.show()

    ## Training score
    y_pred0= mlp.predict(X_train)
    print("\nTraining Score")
    print(classification_report(y_train,y_pred0))
    print("Confusion matrix, without normalization")
    print(confusion_matrix(y_train,y_pred0).T)

    ## Test Score
    print("\nTest Score")
    y_pred= mlp.predict(X_test)
    print(classification_report(y_test,y_pred))
    cm=confusion_matrix(y_test,y_pred).T
    fns.plot_confusion_matrix(cm,labels=nn_label)
    plt.show()

    return

if __name__ == "__main__":
    main()
