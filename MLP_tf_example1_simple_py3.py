"""
Test Multi-layer Perceptron (Feed-forward) Neural Network

Input: OLR data (NOAA CDR Monthly; box mean)
Target: 3-month forecast of El Nino or La Nina

By Daeho Jin
2021.01.31
---

Using Tensorflow+Keras

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
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    from sklearn.model_selection import train_test_split
    #from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report,confusion_matrix
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    tf.compat.v1.enable_eager_execution()

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

    ### Train-Test split: 5 years for test, and no shuffle for testing recent years
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 60/y.shape[0],shuffle=False)

    ### Shuffle train data before start
    #rg = np.random.default_rng(seed=1) ## Numpy version>=1.17
    rg= np.random.RandomState(seed=1)  ## Numpy version<1.17
    shf_idx= np.arange(y_train.shape[0],dtype=int)
    rg.shuffle(shf_idx)
    X_train, y_train= X_train[shf_idx,:], y_train[shf_idx]
    print('Shuffling of train data is done.')

    ### MLP in Tensorflow+Keras
    #model= tf.keras.Sequential([
    #    layers.Dense(15, activation='relu'),
    #    layers.Dense(len(nn_label), activation='relu')        ])
    model= models.Sequential()
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(len(nn_label), activation='relu'))

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

    usualCallback = EarlyStopping(monitor='loss',patience = 25) #monitor='sparse_categorical_accuracy')
    #overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 25)

    history=model.fit(X_train, y_train, epochs=1200, callbacks=[usualCallback,])
    model.summary()
    fns.plot_loss(history.history)
    plt.show()
    #for key in history.history.keys():
    #    print(key,history.history[key])

    ## Prediction with softmax
    prob_model= models.Sequential([model, layers.Softmax()])
    y_pred0= np.argmax(prob_model(X_train),axis=1)
    y_pred= np.argmax(prob_model(X_test),axis=1)

    ## Training score
    print("\nTraining Score")
    print(classification_report(y_train,y_pred0))
    print("Confusion matrix, without normalization")
    print(confusion_matrix(y_train,y_pred0).T)

    ## Test Score
    print("\nTest Score")
    print(classification_report(y_test,y_pred))
    cm=confusion_matrix(y_test,y_pred).T
    fns.plot_confusion_matrix(cm,labels=nn_label)
    plt.show()

    return

if __name__ == "__main__":
    main()
