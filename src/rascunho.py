#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 06:22:35 2019

@author: rainfall
"""

 ## load YAML and create model
        yaml_file = open(ymlp+'final_screening_'+ymlv+'.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(ymlp+'final_screening_'+ymlv+'.h5')
        print("Loaded models yaml and h5 from disk!")
#        loaded_model = keras.models.load_model(ymlp+ymlf)
#        loaded_model.summary()
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        # Fix random seed for reproducibility:
        np.random.seed(7)

        # Load dataset:
        df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
        x = df.loc[:,['SI', '89V', '166V', '190V']]
        y_true = df.loc[:,['TagRain']]

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

        with open('categorical_scores_screening_'+ymlv+'.txt', 'w') as myfile:
             myfile.write(str(my_scores))
        print("Text file saved!")
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        df['SCR'] = ""
        df['SCR'] = y_pred
        filename=file[22:58]
        filename = 'validation_SCR_'+filename+'.csv'
        df.to_csv(os.path.join(path, filename), index=False, sep=",", decimal='.')