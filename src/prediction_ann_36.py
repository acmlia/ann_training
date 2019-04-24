#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:04:36 2019

@author: rainfall
"""

from __future__ import absolute_import, division, print_function

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from src.meteoro_skills import CategoricalScores
from src.meteoro_skills import ContinuousScores
from keras.models import load_model

import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras import layers
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_yaml

print('TF version '+tf.__version__)

# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# ------------------------------------------------------------------------------

class Prediction:
    """
    This module is intended to automate the TensorFlow Neural Network training.
    """
    PCA = PCA()
    seed = 0
    run_prefix = ''
    tver = ''
    vernick = ''
    file = ''
    path = ''
    fig_title = ''
    path_fig = ''
    mod_out_pth = ''
    mod_out_name = ''
    ymlv = ''
    ymlp = ''
    ymlf = ''


    def __init__(self, random_seed=0,
                 run_prefix='',
                 version='',
                 version_nickname='',
                 file_csv='',
                 path_csv='',
                 fig_title='',
                 figure_path='',
                 model_out_path='',
                 model_out_name='',
                 yaml_version='',
                 yaml_path='',
                 yaml_file=''):

        self.seed=random_seed
        self.run_prefix=run_prefix
        self.tver=version
        self.vernick=version_nickname
        self.file=file_csv
        self.path=path_csv
        self.path_fig=figure_path
        self.fig_title=run_prefix+version+version_nickname
        self.mod_out_pth=model_out_path
        self.mod_out_name=model_out_name
        self.ymlv=yaml_version
        self.ymlp=yaml_path
        self.ymlf=yaml_file
    # -------------------------------------------------------------------------
    # DROP DATA OUTSIDE INTERVAL
    # -------------------------------------------------------------------------
       
    @staticmethod
    def keep_interval(keepfrom: 0.0, keepto: 1.0, dataframe, target_col: str):
        keepinterval = np.where((dataframe[target_col] >= keepfrom) &
                                (dataframe[target_col] <= keepto))
        result = dataframe.iloc[keepinterval]
        return result

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------ 

    def PredictScreening(self):

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------ 
        ## load YAML and create model
#        yaml_file = open(self.ymlp+'screening_'+self.ymlv+'.yaml', 'r')
#        loaded_model_yaml = yaml_file.read()
#        yaml_file.close()
#        loaded_model = model_from_yaml(loaded_model_yaml)
#        # load weights into new model
#        loaded_model.load_weights(self.ymlp+'screening_'+self.ymlv+'.h5')
#        print("Loaded models yaml and h5 from disk!")
#        loaded_model = keras.models.load_model(self.ymlp+self.ymlf)
#        loaded_model.summary()
        loaded_model = joblib.load('screening_TESTE.pkl')
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        # Fix random seed for reproducibility:
        np.random.seed(self.seed)

        # Load dataset:
        df = pd.read_csv(os.path.join(self.path, self.file), sep=',', decimal='.')
        x = df.loc[:,['36V', '89V', '166V', '190V']]
        y = df.loc[:,['TagRain']]
        y_true = np.ravel(y)

        x_arr = np.asanyarray(x)

        # Scaling the input paramaters:
#       scaler_min_max = MinMaxScaler()
        norm_sc = Normalizer()
        x_normalized= norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
#        x_train, x_test, y_train, y_test = train_test_split(x_normalized,
#                                                            y_arr, test_size=0.10,
#                                                            random_state=101)

        # Doing prediction from the test dataset:
        y_pred = loaded_model.predict_classes(x_normalized)
        y_pred = np.ravel(y_pred)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # Appplying meteorological skills to verify the performance of the model, in this case, categorical scores:

        skills = CategoricalScores()
        val_accuracy, val_bias, val_pod, val_pofd, val_far, val_csi, val_ph, val_ets, val_hss, val_hkd, val_num_pixels = skills.metrics(y_true, y_pred)
        
        #converting to text file
        print("converting arrays to text files")
        my_scores = {'val_accuracy': val_accuracy,
                     'val_bias': val_bias,
                     'val_pod': val_pod,
                     'val_pofd': val_pofd,
                     'val_far': val_far,
                     'val_csi': val_csi,
                     'val_ph': val_ph,
                     'val_ets': val_ets,
                     'val_hss': val_hss,
                     'val_hkd': val_hkd,
                     'val_num_pixels': val_num_pixels}

        with open('cateorical_scores_TESTE.txt', 'w') as myfile:
             myfile.write(str(my_scores))
        print("Text file saved!")
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        df['SCR'] = ""
        df['SCR'] = y_pred
        filename=self.file[22:58]
        filename = 'validation_SCR_TESTE'+filename+'.csv'
        df.to_csv(os.path.join(self.path, filename), index=False, sep=",", decimal='.')

        return df

    def PredictRetrieval(self):
        #------------------------------------------------------------------------------ 
        #load YAML and create model
        yaml_file = open(self.ymlp+'final_'+self.ymlv+'.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(self.ymlp+'final_'+self.ymlv+'.h5')
        print("Loaded models yaml and h5 from disk!")
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
#       Fix random seed for reproducibility:
        np.random.seed(self.seed)
# ------------------------------------------------------------------------------
        df_orig = pd.read_csv(os.path.join(self.path, self.file), sep=',', decimal='.')

        df_input = df_orig.loc[:, ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                                   '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH', 'sfcprcp']]

        # ------------------------------------------------------------------------------
        #df_input = self.keep_interval(0.2, 60, df_input, 'sfcprcp')
        y_true = df_input.pop('sfcprcp')

        colunas = ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                   '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH']
        # ------------------------------------------------------------------------------
        scaler = StandardScaler()

        x_norm = scaler.fit_transform(df_input)
        df_x_norm = pd.DataFrame(x_norm[:], index = df_input.index,columns=colunas)
        ancillary = df_x_norm.loc[:, ['183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH']]

        # ------------------------------------------------------------------------------
        # Choosing the number of components:

        TB1 = df_x_norm.loc[:, ['10V', '10H', '18V', '18H']]
        TB2 = df_x_norm.loc[:, ['36V', '36H', '89V', '89H', '166V', '166H']]

        # ------------------------------------------------------------------------------
        # Verifying the number of components that most contribute:
        pca = self.PCA
        pca1 = pca.fit(TB1)
        TB1_pca = pca1.transform(TB1)
        # ------------------------------------------------------------------------------
        pca2 = pca.fit(TB2)
        TB2_pca = pca2.transform(TB2)
        # ------------------------------------------------------------------------------
        # JOIN THE TREATED VARIABLES IN ONE SINGLE DATASET AGAIN:

        PCA1 = pd.DataFrame(TB1_pca,index = df_input.index,
                            columns=['pca1_1', 'pca1_2', 'pca1_3', 'pca1_4'])
        PCA2 = pd.DataFrame(TB2_pca, index = df_input.index,
                            columns=['pca2_1', 'pca2_2', 'pca2_3', 'pca2_4', 'pca2_5', 'pca2_6'])

        dataset = PCA1.join(PCA2, how='right')
        dataset = dataset.join(ancillary, how='right')
        sfcprcp = pd.DataFrame(y_true, df_input.index, columns=['sfcprcp'])
        dataset = dataset.join(sfcprcp, how='right') 
        dataset = dataset.dropna()

        SCR_pixels = np.where((df_orig['SCR01'] == 1))
        dataset = dataset.iloc[SCR_pixels]
        dataset_index=dataset.index.values
        
        #SCR = dataset.pop('SCR01')
        y_true = dataset.pop('sfcprcp')
        #SCR = dataset.pop('SCR01')

        x_norm = dataset.values
        y_pred = loaded_model.predict(x_norm).flatten()

        # ------------------------------------------------------------------------------
        # Appplying meteorological skills to verify the performance of the model, in this case, categorical scores:

        skills = ContinuousScores()
        val_y_pred_mean, val_y_true_mean, val_mae, val_rmse, val_std, val_fseperc, val_fse, val_corr, val_num_pixels = skills.metrics(y_true, y_pred)
        
        #converting to text file
        print("converting arrays to text files")
        my_scores = {'val_y_pred_mean': val_y_pred_mean,
                     'val_y_true_mean': val_y_true_mean,
                     'val_mae': val_mae,
                     'val_rmse': val_rmse,
                     'val_std': val_std,
                     'val_fseperc': val_fseperc,
                     'val_fse': val_fse,
                     'val_corr': val_corr,
                     'val_num_pixels': val_num_pixels}

        with open(self.path+'continuous_scores_SCR01_'+self.tver+'_'+self.ymlv+'.txt', 'w') as myfile:
             myfile.write(str(my_scores))
        print("Text file saved!")

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        df_final = df_orig.iloc[dataset_index]
        df_final['y_true'] = y_true.values
        df_final['y_pred'] = y_pred
        #filename=self.file[21:58]
        filename = 'retrieval_SCR01_'+self.tver+'_'+self.ymlv+'.csv'
        df_final.to_csv(os.path.join(self.path, filename), index=False, sep=",", decimal='.')

        return df_final
