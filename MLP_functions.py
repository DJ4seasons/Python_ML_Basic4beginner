"""
Functions to be used for exampel codes

By Daeho Jin
2021.01.31
"""

import sys
import os.path
import numpy as np

from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta

from math import ceil
def lon_deg2x(lon,lon0,dlon,nlon=-1):
    if nlon==-1: nlon= int(360/dlon)
    x=ceil((lon-lon0)/dlon)
    if x>=nlon: x-=nlon
    return x
lat_deg2y = lambda lat,lat0,dlat: ceil((lat-lat0)/dlat)

def open_netcdf(fname):
    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()

    fid=Dataset(fname,'r')
    print("Open:",fname)
    return fid

def read_nc_data(infn,var_names):
    fid= open_netcdf(infn)

    data={}  ## Empty dictionary
    for vv in var_names:
        if vv=='time':
            tt= fid.variables[vv]
            tinfo= tt.units.split()
            if tinfo[0]=='days':
                t0= [int(val) for val in tinfo[2].split('-')]
                times=[datetime(*t0)+timedelta(days=float(dt)) for dt in tt[:]]
            else:
                sys.exit('Now only support days, but {}'.format(tinfo))
            data[vv]=np.asarray(times)
        else:
            data[vv]= fid.variables[vv][:]  ## Save as numpy array
        out_txt="Name= {}, Dims= {}".format(vv,data[vv].shape)
        if vv[:3].lower()=='lon' or vv[:3].lower()=='lat' or vv[:3].lower()=='tim':
            out_txt+=', from {} to {}'.format(data[vv][0],data[vv][-1])
        print(out_txt)
    return data

def read_nn34_text(infn,tgt_dates):
    tgt_iyr, tgt_eyr= tgt_dates[0].year, tgt_dates[1].year

    with open(infn,'r') as f:
        lines=[]
        for line in f:
            ww= line.strip().split()
            if len(ww)>0:
                lines.append(ww)

    undef= float(lines[-3][0])
    iyr,eyr= map(int,lines[0])
    outdata=[]
    if iyr>tgt_iyr:
        for yy in range(tgt_iyr,iyr):
            outdata.append([undef,]*12)
    for wws in lines[1:-3]:
        lyy= int(wws[0])
        if lyy>=tgt_iyr and lyy<=tgt_eyr:
            outdata.append([float(val) for val in wws[1:13]])
    if tgt_eyr>eyr:
        for yy in range(eyr+1,tgt_eyr+1,1):
            outdata.append([undef,]*12)

    return np.asarray(outdata)



import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, labels=[],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = labels #[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           xlabel='True label',
           ylabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),fontsize=12,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_loss(data_dict, title=None):
    """
    This function plots the model learning log.
    """
    if not title:
        title = 'Model Learning Log'

    for key in data_dict.keys():
        if key=='loss':
            data1= data_dict[key]
            key_name1= key
        else:
            data2= data_dict[key]
            key_name2= key

    fig, ax = plt.subplots()
    im = ax.plot(data1,color='b')
    ax2= ax.twinx()
    im2= ax2.plot(data2,color='orange')
    # We want to show all ticks...
    ax.set(title=title,
           xlabel='Epochs',
           ylabel=key_name1)
    ax.set_ylabel(key_name1.title(),color='b',fontsize=11)
    ax2.set_ylabel(key_name2.title(),color='orange',fontsize=11)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    fig.tight_layout()
    return   #ax
