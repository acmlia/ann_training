# ------------------------------------------------------------------------------
# Loading the libraries to be used:
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from collections import Counter
#from meteoro_skills import CategoricalScores
#from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
#from imblearn.combine import SMOTEENN, SMOTETomek

# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


class TuningScreeningPrecipitation:
    

    def  create_model(self, neurons=1):
        '''
         Fucntion to create the instance and configuration of the keras
         model(Sequential and Dense).
        '''
        # Create the Keras model:
        model = Sequential()
        model.add(Dense(neurons, input_dim=10, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(neurons, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'],)
        return model

    def run_TuningScreeningPrecipitation(self):
        '''
        Fucntion to create the instance and configuration of the KerasRegressor
        model(keras.wrappers.scikit_learn).
        '''

        # Fix random seed for reproducibility:
        seed = 7
        np.random.seed(seed)

        # Load dataset:
        #path = '/home/david/DATA/'
        path = '/home/david/DATA/'
        file = 'train_data_11m_outliers.csv'
        df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
        x, y= df.loc[:,['36V', '89V','89VH','166V','166VH','186V','190V','SI','PCT36','PCT89']], df.loc[:,['TagRain2']]
        
        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        y_arr = np.ravel(y_arr)
        
        # Applying the Imabalanced Learn Solution: SMOTEENN
        print('Original dataset shape %s' % Counter(y_arr))
        #sm = SMOTEENN(random_state=42)
        #x_res, y_res = sm.fit_resample(x_arr, y_arr)
        #print('Resampled dataset shape %s' % Counter(y_res))

        # Scaling the input paramaters:
#       scaler_min_max = MinMaxScaler()
#       x_scaled = scaler_min_max.fit_transform(x)
        norm_sc = Normalizer()
        x_normalized= norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_normalized,
                                                            y_arr, test_size=0.20,
                                                            random_state=101)

#        # Inserting the modelcheckpoint:
#        checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc',
#                                      save_best_only=True, mode='max')
#        callbacks_list = [checkpoint]

        # Create the instance for KerasRegressor:
        model = KerasClassifier(build_fn=self.create_model, batch_size=10,
                                epochs=100, verbose=0)

        # Define the grid search parameters:
        neurons = [20, 32]
        param_grid = dict(neurons=neurons)
        grid_model = GridSearchCV(estimator=model, param_grid=param_grid,
                                  cv=10, n_jobs=-1)
        grid_result = grid_model.fit(x_train, y_train)
        return grid_result
        #modelo=model(grid_result)

        # Summarize results:
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        # list all data in history
        #history = model.fit(...)
        #print(grid_result.callbacks.History())

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Saving a model
if __name__ == '__main__':
    _start_time = time.time()

    tic()

    training_model = TuningScreeningPrecipitation()
    grid_result = training_model.run_TuningScreeningPrecipitation()
    joblib.dump(grid_result, 'model_trained_screening_precipitation_A7.pkl')
    #loaded_model = joblib.load('model_trained_screening_precipitation_A7.pkl')

    # Saving model to YAML:

    # serialize model to YAML
#    model_yaml = grid_result.to_yaml()
#    with open('final_screening'+'.yaml', 'w') as yaml_file:
#        yaml_file.write(model_yaml)
#    # serialize weights to HDF5
#    model.save_weights('final_screening'+'.h5')
#    print("Saved model to disk")

    tac()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Loading the input dataset to be used to make "unseen" predictions:
# Fix random seed for reproducibility:
#    seed = 7
#    np.random.seed(seed)
##
##    # Load dataset:
#    
#    path_csv='/media/DATA/tmp/git-repositories/validation/NOV_2014/orbita_4299/'
#    file_csv='regional_all_input_SCR01_orbita_4299_30112014.csv'
#    df_pred = pd.read_csv(os.path.join(path_csv, file_csv), sep=',', decimal='.')
#    x, y= df_pred.loc[:,['36V', '89V', '166V', '190V']], df_pred.loc[:,['TagRain2']]
#
#
##    path = '/media/DATA/tmp/git-repositories/imbalanced-learn/'
##    file = 'yearly_br_TAG_class1_reduced.csv'    
##    df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
#    
#    x_arr = np.asanyarray(x)
#    y_arr = np.asanyarray(y)
#    y_true = np.ravel(y_arr)
#
#    # Scaling the input paramaters:
#    #       scaler_min_max = MinMaxScaler()
#    #       x_scaled = scaler_min_max.fit_transform(x)
#    norm_sc = Normalizer()
#    x_norm= norm_sc.fit_transform(x_arr)
#
#    # Split the dataset in test and train samples:
#    #x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_arr, test_size=0.10, random_state=101)
#
#    # Doing prediction from the test dataset:
#    y_pred = loaded_model.predict(x_norm)
#    y_pred = np.ravel(y_pred)
#    
#    num_pxls_true = np.where((df['TagRain1'] == 1))
#    num_pxls_true = len(num_pxls_true[0])
#    num_pxls_prd_scr = np.where((y_pred == 1))
#    num_pxls_prd_scr = len(num_pxls_prd_scr[0])
#
#
#     # ------------------------------------------------------------------------------
##    # ------------------------------------------------------------------------------
##    # Appplying meteorological skills to verify the performance of the model, in this case, categorical scores:
##
#    skills = CategoricalScores()
#    val_accuracy, val_bias, val_pod, val_pofd, val_far, val_csi, val_ph, val_ets, val_hss, val_hkd, val_num_pixels= skills.metrics(y_true, y_pred)
#    
#    print("converting arrays to text files")
#    my_scores = {'val_accuracy': val_accuracy,
#                     'val_bias': val_bias,
#                     'val_pod': val_pod,
#                     'val_pofd': val_pofd,
#                     'val_far': val_far,
#                     'val_csi': val_csi,
#                     'val_ph': val_ph,
#                     'val_ets': val_ets,
#                     'val_hss': val_hss,
#                     'val_hkd': val_hkd,
#                     'val_tot_num_pixels': val_num_pixels,
#                     'val_num_pxls_true': num_pxls_true,
#                     'val_num_pxls_prd_scr': num_pxls_prd_scr}
#
#    with open('categorical_scores_SCR01_orbita_4299_ann_14_db_11m.txt', 'w') as myfile:
#        myfile.write(str(my_scores))
#        print("Text file saved!")
#    # ------------------------------------------------------------------------------
#    # ------------------------------------------------------------------------------
#    df_pred['SCR02'] = ""
#    df_pred['SCR02'] = y_pred
#    file_name = "regional_all_input_SCR02_orbita_4299_30112014.csv"
#    df_pred.to_csv(os.path.join(path_csv, file_name), index=False, sep=",", decimal='.')
#df_pred.to_csv(os.path.join(path_csv, file_name), index=False, sep=",", decimal='.')
