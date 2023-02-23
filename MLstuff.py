# -*- coding: utf-8 -*-
"""
Created on 2022/06/07

@author: a072108
"""
import os
import numpy as np
import pandas as pd
from math import sqrt
import timeit
from datetime import datetime

# Pre-processing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Feature selection
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import make_scorer

# ML
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import get_scorer
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier

# Réseaux de neurones
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

# Librairies perso
from DisplotData import bcolors
import DisplotML

def filter_df_with_excluded_vals(df0, imput_col, excluded_vals):
    """
    Mise à l'écart des valeurs d'imputation interdites d'entraînement.
    """
    print(f"excluded_vals = {excluded_vals}")
    print(f"imput_col = {imput_col}")
    print(f"df0.shape = {df0.shape}")
    if not excluded_vals: 
        excluded_vals = []
    filtre = (df0[imput_col].isin(excluded_vals))
    df_excl = df0.loc[filtre]
    df_work = df0.loc[~filtre]
    return df_work, df_excl


def separate_df_known_unknown(df0,col):
    """
    Séparation du dataset en deux : colonne à imputer connue / inconnue
    """
    known = (df0[col].notna())
    unknown = ~known
    df_known   = df0.loc[known].copy()
    df_unknown = df0.loc[unknown].copy()
    return df_known, df_unknown


def resample_data(X, y, oversampling_ratio=None, undersampling_ratio=None):
    """Oversample et undersample les données."""
    steps = []
    if oversampling_ratio:
        over = SMOTE(sampling_strategy=oversampling_ratio, random_state=18011975) # oversampling_ratio = ce que doit représenter la classe après coup
        steps.append(('o', over))
    if undersampling_ratio:
        under = RandomUnderSampler(sampling_strategy=undersampling_ratio, random_state=18011975) # oversampling_ratio = ce que doit représenter la classe après coup
        steps.append(('u', under))
    if oversampling_ratio or undersampling_ratio:
        pipeline = imbPipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y


def make_preprocessor_X(#  X, 
                        categ_cols, 
                        num_cols, 
                        categ_encoder=None,
                        imputer=None, imputer_params=None,
                        over_sampler=None, over_sampler_params=None,
                        under_sampler=None, under_sampler_params=None,
                        scaler=None):
    """
    Encodage des colonnes catégorielles et mise à l'échelle des colonnes numériques
    """
    # Numerical pipeline
    if imputer:
        # steps_num = [("imputer", imputer(**imputer_params))] # TODO
        steps_num = [("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))] # strategy="median"
    else:
        steps_num = []
    steps_num.append(("scaler", scaler()))
    numeric_transformer = Pipeline(
        steps=steps_num 
    )

    # Categorical pipeline
    steps_cat = []
    if categ_encoder:
        steps_cat.append(("categorical", categ_encoder, categ_cols))

    steps_all = [("numerical", numeric_transformer, num_cols)]
    if categ_encoder:
        steps_all.extend(steps_cat)

    preprocessor = ColumnTransformer(
        steps_all,
        verbose_feature_names_out=False,
    )
    return preprocessor

def preprocess_y(y, encode_y):
    """Encodage de la colonne à imputer/régresser/classifier"""
    if encode_y:
        encoder = LabelEncoder()
        encoder.fit(y)
        y_enc = pd.DataFrame(encoder.transform(y))
    else:
        encoder = None
        y_enc = y.copy()    
    return y_enc, encoder

def data_augmentation_mult(X_train, y_train, data_augm):
    if data_augm:
        # TODO
        pass
    return X_train, y_train

def initialize_keras_model(n_num_cols, scoring=None):
    # Documentation de la bibliothèque Keras: https://keras.io/guides/sequential_model/
    model = models.Sequential()
    # On indique à notre modèle la dimension des données d'entrées qui correspond au nombre de colonnes de X_train
    model.add(keras.Input(shape=(n_num_cols))) 
    # Première couche de notre réseau de neurones
    model.add(layers.Dense(n_num_cols, input_dim=2, activation='sigmoid'))
    # Seconde couche car nous sommes dans un problème possiblement non-linéaire
    model.add(layers.Dense(1, activation=''))
    # Ici, nous pouvons ajouter des paramètres à notre modèle. Il faut juste retenir que "accuracy" permet d'avoir
    # la précision de notre modèle et est particulièrement indiqué pour les problèmes de classification. 
    if not scoring: scoring = 'mean_squared_error'
    model.compile(loss=scoring,
                  optimizer='adam')
    return model


def initialize_pytorch_model(X, y):

    return model

def rename_keys_with_prefix_suffix(dico, prefix='', suffix=''):
    new_dico = {}
    for oldkey in dico.keys():
        newkey = prefix+oldkey+suffix
        # if newkey!=oldkey:
        new_dico[newkey] = dico[oldkey]
            # del dico[oldkey]
    return new_dico


def launch_algo(X_train, y_train, X_test, y_test, 
                categ_cols, num_cols, target_col,
                algo_info, scores, problem_kind,
                categ_encoder=None,
                scaler=None,
                oversampling=False, undersampling=False,
                predict_proba=True, class_to_predict=1,
                custom_score=None, threshold=0.5, alpha=None,
                encode_y=False,
                display_results_text=0):
    """
    Lance un algorithme sur un train/test set et renvoie le score spécifié.
    """
    algo_res = {}

    algo_surname = algo_info['surname']
    algo = algo_info['class']
    param_grid = algo_info['param_grid']

    if not custom_score is None:
        main_score = custom_score
    else :
        main_score = scores[0]

    ch = f"\nAlgorithm {algo_surname} :\n{'-'*(len(algo_surname)+12)}\n"
    print(bcolors.BOLD+bcolors.OKBLUE+ch+bcolors.ENDC)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ch = now + '\n' + ch
    os.write(1, ch.encode())
    
    # Top chrono
    start_time = timeit.default_timer()

    steps = []
    # Encodage des colonnes catégorielles et mise à l'échelle des colonnes numériques
    preprocessor_X = make_preprocessor_X(#X_train, 
                                         categ_cols, num_cols, 
                                         categ_encoder=categ_encoder, 
                                         scaler=scaler)
    steps.append(('preprocessor', preprocessor_X))

    # Encodage de la colonne à régresser
    y_train_enc, encoder_y = preprocess_y(y_train, encode_y)

    # Partie test
    # X_test_enc = preprocessor_X.transform(X_test)
    if encode_y:
        y_test_enc = encoder_y.transform(y_test)
    else:
        y_test_enc = y_test
    X_train_orig = X_train
    X_test_orig = X_test

    # Over et under sampling pipeline
    if oversampling:
        if oversampling == 'SMOTE':
            steps.append(('over', SMOTE(sampling_strategy='auto', random_state=18011975, k_neighbors=1))) # sampling_strategy=0.48
        elif oversampling == 'SMOTE-NC':
            non_relaxable_cols_ind = [n for n, col in enumerate(X_train_orig.columns) if '_NR' in col]
            print(f"non_relaxable_cols = {non_relaxable_cols_ind}")
            steps.append(('over', SMOTENC(categorical_features=non_relaxable_cols_ind, sampling_strategy=0.48, 
                                          k_neighbors=3, random_state=18011975, n_jobs=-1)))
        elif oversampling == 'ADASYN':
            steps.append(('over', ADASYN(sampling_strategy=0.48, random_state=18011975)))
        elif oversampling == 'BorderlineSMOTE':
            steps.append(('over', BorderlineSMOTE(sampling_strategy=0.48, random_state=18011975)))
        elif oversampling == 'SVMSMOTE':
            steps.append(('over', SVMSMOTE(sampling_strategy=0.48, random_state=18011975)))
        elif oversampling == 'RandomOverSampler':
            steps.append(('over', RandomOverSampler(sampling_strategy=0.48, random_state=18011975)))
        
    if undersampling == 'RandomUnderSampler':
        steps.append(('under', RandomUnderSampler(sampling_strategy='auto', random_state=18011975)))

    # DEBUG
    if problem_kind=='classification': print(f"y_train.value_counts() = \n{y_train.value_counts()}")
    if oversampling or undersampling:
        temp_pipeline = imbPipeline(steps=steps)
        print("Resampling dataset...")
        X_train_temp, y_train_temp = temp_pipeline.fit_resample(X_train, y_train)
        print("Dataset resampled.")
        print(f"y_train_temp.value_counts() = {y_train_temp.value_counts()}")
    # FIN DEBUG

    # Ajout du modèle
    steps.append(('model', algo))

    # Fabrication du pipeline 
    if oversampling or undersampling:
        if oversampling and undersampling:
            print('+++ WITH oversampling AND undersampling')
        elif oversampling and not undersampling:
            print('+++ WITH oversampling only')
        elif not oversampling and undersampling:
            print('+++ WITH undersampling only')
        pipeline = imbPipeline(steps=steps)
    else:
        print('--- WITHOUT over/undersampling')
        pipeline = Pipeline(steps=steps)

    # KFold
    if problem_kind=='regression':
        cv = 5
    else:
        cv = StratifiedKFold(n_splits=5,
                             shuffle=True,
                             random_state=18011975)

    # On renomme les clés de param_grid pour les associer au classifieur
    param_grid2 = rename_keys_with_prefix_suffix(param_grid, prefix='model__')

    # Prédicteur avec recherche d'hyperparamètres par validation croisée
    if algo_surname=='KerasMLP':
        n_num_cols = X_train.shape[1]
        mdl = initialize_keras_model(n_num_cols, scoring=None)
    else:
        mdl = GridSearchCV(
            pipeline,
            param_grid=param_grid2,
            cv=cv,
            scoring=main_score,
            n_jobs=-1
        )

    # Entraînement de l'algorithme optimal sur ses hyperparamètres sur le jeu d'entraînement
    y_train_enc = y_train_enc.values.reshape(-1, 1) # Pour PyTorch
    if algo_surname=='KerasMLP':
        epochs = 5 # 100
        batch_size = 4
        mdl.fit(X_train, y_train_enc, epochs=epochs, batch_size=batch_size)
    else:
        mdl.fit(X_train, y_train_enc)

    # Régression sur le train set
    if problem_kind=='regression':
        y_train_pred_enc = mdl.predict(X_train)
    else: # classification
        if predict_proba:
            y_train_pred_enc = mdl.predict_proba(X_train)[:, class_to_predict]
        else:
            y_train_pred_enc = mdl.predict(X_train)
    if encode_y:
        y_train_pred = encoder_y.inverse_transform(y_train_pred_enc)
        y_train_index = None 
    else:
        y_train_pred = y_train_pred_enc
        y_train_index = y_train.index
    y_train_pred = pd.DataFrame(y_train_pred,columns=[target_col+'_pred'],index=y_train_index)

    # Régression sur le test set
    if problem_kind=='regression':
        y_test_pred_enc = mdl.predict(X_test)
    else: # classification
        if predict_proba:
            y_test_pred_enc = mdl.predict_proba(X_test)[:, class_to_predict]
        else:
            y_test_pred_enc = mdl.predict(X_test)
    if encode_y:
        y_test_pred = encoder_y.inverse_transform(y_test_pred_enc)
        y_test_index = None 
    else:
        y_test_pred = y_test_pred_enc
        y_test_index = y_test.index
    y_test_pred = pd.DataFrame(y_test_pred,columns=[target_col+'_pred'],index=y_test_index)

    # Fin du chrono
    etime = timeit.default_timer() - start_time


    # Stockage des résultats
    algo_res['scoring'] = main_score
    if algo_surname=='NeuralNetwork':
        nnr_score = mdl.evaluate(X_test, y_test, verbose=0)
        main_score_val = nnr_score
    else:
        main_score_val = mdl.cv_results_['mean_test_score'].max()
    algo_res['Validation score'] = main_score_val
    # algo_res['Validation relative score'] = main_score_val

    # Calcul du score
    if problem_kind=='regression':
        for score in scores:
            # Train
            s, r = compute_abs_and_rel_score(score, y_train, y_train_pred)
            algo_res['Train '+score] = s
            algo_res['Train relative '+score] = r
            # Test
            s, r = compute_abs_and_rel_score(score, y_test, y_test_pred)
            algo_res['Test '+score] = s
            algo_res['Test relative '+score] = r
            if score=='r2':
                n = X_train.shape[0]
                p = X_train.shape[1]
                algo_res['Test adj '+score] = 1. - (n-1.)/(n-p-1.)*(1. - s)
    else: # Classification
        if not custom_score is None:
            score_function = custom_score
            score_str = 'custom_score'
            score = 'custom_score'
            algo_res['Train '+score_str] = compute_classif_score(X_train, X_train_orig, y_train, y_train_pred, target_col, mdl, score, score_str, score_function,
                                                                 predict_proba, class_to_predict, threshold=threshold, alpha=alpha)
            algo_res['Test '+score_str] = compute_classif_score(X_test, X_test_orig, y_test, y_test_pred, target_col, mdl, score, score_str, score_function,
                                                                predict_proba, class_to_predict, threshold=threshold, alpha=alpha)

        for score in scores:
            score_function = get_scorer(score)
            if isinstance(score,str):
                score_str = score
            else:
                score_str = 'custom_score'
            algo_res['Train '+score_str] = compute_classif_score(X_train, X_train_orig, y_train, y_train_pred, target_col, mdl, score, score_str, score_function,
                                                                 predict_proba, class_to_predict, threshold=threshold, alpha=alpha)
            algo_res['Test '+score_str] = compute_classif_score(X_test, X_test_orig, y_test, y_test_pred, target_col, mdl, score, score_str, score_function,
                                                                predict_proba, class_to_predict, threshold=threshold, alpha=alpha)

            # if display_results_text>1: 
            # print(classification_report(y_test, y_test_pred[target_col+'_pred']))


    algo_res['y_train_pred'] = y_train_pred
    algo_res['y_test_pred'] = y_test_pred
    algo_res['model'] = mdl
    algo_res['preprocessor_X'] = preprocessor_X
    algo_res['encoder_y'] = encoder_y
    algo_res['etime'] = etime

    # Affichage des résultats numériques
    DisplotML.display_cv_results(problem_kind, mdl, main_score, display_results_text,
                                 algo_surname)
    
    return mdl, algo_res, preprocessor_X, encoder_y


def compute_classif_score(X_sc, X_orig, y_true, y_pred, target_col, mdl, score, score_str, score_function,
                          predict_proba, class_to_predict, threshold=0.5, alpha=None):
    if score in ['roc_auc', 'precision', 'recall', 'f1']:
        s = score_function._score_func(y_true, y_pred)
    elif score_str=='custom_score':
        s = score_function._score_func(y_true, y_pred[target_col+'_pred'], X_train=X_orig, threshold=threshold, alpha=alpha)
    else:
        s = score_function._score_func(y_true, y_pred[target_col+'_pred'])
    return s


def transform_into_array_vector(y):
    y2 = y
    if isinstance(y, pd.Series):
        y2 = y.values.ravel() # reshape(-1,) # reshape(n,)
    elif isinstance(y, np.ndarray):
        y2 = y.ravel() # reshape(-1,) # reshape(n,)        
    elif isinstance(y, list):
        y2 = np.array(y)
    else:
        # On ne fait rien
        pass
    return(y2)


def compute_abs_and_rel_score(score,y_truth,y_pred):
    s, r = None, None

    # On évite de créer des matrices en divisant 2 vecteurs
    n = y_truth.shape[0]
    y_truth = transform_into_array_vector(y_truth)
    y_pred  = transform_into_array_vector(y_pred)

    # y_truth_rel = y_truth/y_truth # oui ça fait 1.
    # y_pred_rel  = y_pred/y_truth # TODO : tester les valeurs nulles 0.
    zero_truth = np.zeros(y_truth.shape)

    score_function = get_scorer(score)
    # Train
    s = score_function._score_func(y_truth, y_pred)
    if score.endswith('_mean_squared_error'):
        s = sqrt(abs(s))
        r = s / sqrt(abs(score_function._score_func(y_truth, zero_truth))*n)
    if score.endswith('_mean_absolute_error'):
        r = s / (score_function._score_func(y_truth, zero_truth)*n)
    return s, r


def regression_doe(df0, df_doe0,
                   target_col,
                   problem_kind,
                   scores,
                   encode_y=True):
    print(f"Size of the regression train set : {df0.shape}")
    algos_results = []
    score_new  = -np.inf
    score_best = -np.inf
    df_doe = df_doe0.copy()

    # Boucle sur les problèmes à résoudre, càd les lignes de df_doe
    for ind in df_doe.index:

        problem = df_doe.loc[ind]
        categ_cols    = problem['Variables']['categ_cols']
        num_cols      = problem['Variables']['num_cols']
        categ_encoder = problem['Encoder']
        scaler        = problem['Scaler']
        algo_info     = problem['Algorithm']
        ignored_indices_train = problem['Ignored train index']
        algo_surname  = algo_info['surname']

        # Ensemble des colonnes formant les features
        feat_cols = categ_cols + num_cols
        
        print(f"Explanatory variables            : {feat_cols}")

        # Création des matrices et vecteurs de travail 
        X = df0[feat_cols].copy()
        y = df0[[target_col]].copy()

        X[num_cols] = X[num_cols].astype('float32') # for PyTorch
        y = y.astype('float32')

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18011975) # , stratify=y_enc

        # Suppression des index non-désirés dans le dataset d'entraînement
        if ignored_indices_train:
            inds = []
            for ig in ignored_indices_train:
                if ig in X_train.index:
                    inds.append(ig)
            X_train.drop(inds,inplace=True)
            y_train.drop(inds,inplace=True)
            print(f"Elimination of {len(ignored_indices_train)} samples from the train set !")
        print(f"Size of the train set : {X_train.shape}")
        

        # Lance l'algo et renvoie le meilleur pour le score principal spécifié en première position
        mdl, algo_res, preprocessor_X, encoder_y = \
            launch_algo(X_train, y_train, X_test, y_test, 
                        categ_cols, num_cols, target_col,
                        algo_info, scores, problem_kind,
                        categ_encoder=categ_encoder,
                        scaler=scaler,
                        display_results_text=1)
        score_new = algo_res['Validation score']
        # score_new = algo_res['Test score']

        # Mémorisation du meilleur modèle
        if score_new>score_best:
            score_best = score_new
            if algo_surname=='NeuralNetwork':
                best_mdl = 'NNR' # TODO
            else:
                best_mdl = mdl.best_estimator_
            best_mdl_name = algo_surname

        y_train = pd.DataFrame(y_train,columns=[target_col])
        y_test = pd.DataFrame(y_test,columns=[target_col])

        # Mémorisation des résultats dans la structure algo_results
        preprocessor_X.fit(X_train)
        algo_res['Column names'] = preprocessor_X.get_feature_names_out()
        algo_res['X_train'] = X_train
        # algo_res['X_train_enc'] = X_train_enc
        algo_res['y_train'] = y_train
        algo_res['X_test'] = X_test
        # algo_res['X_test_enc'] = X_test_enc
        algo_res['y_test'] = y_test
        algos_results.append(algo_res)

        # Mémorisation des résultats dans le dataframe du DOE
        print(mdl.best_params_)
        df_doe.loc[ind,'Best params'] = f"{mdl.best_params_}"
        df_doe.loc[ind,'Scoring'] = scores[0]
        for score in scores:
            for prefix in ['Train ', 'Train relative ',
                          'Test ', 'Test relative ']:
                key = prefix+score
                if algo_res[key]:
                    df_doe.loc[ind,key] = algo_res[key]
                else:
                    df_doe.loc[ind,key] = "N/A"
        df_doe.loc[ind,'Test adj r2'] = algo_res['Test adj r2']
        df_doe.loc[ind,'Elapsed time'] = algo_res['etime']

        pass

    # Affichages
    if scores[0].startswith('neg_'): score_best = -score_best
    print(bcolors.BOLD+bcolors.OKGREEN+f"\nAnd the Oscar for the best algorithm is awarded to: "+bcolors.ENDC + best_mdl_name + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nfor his role in                                   : "+bcolors.ENDC + f"{best_mdl}" + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nand its (absolute) validation performance of      : "+bcolors.ENDC + f"{score_best:.3e}")
    s = algo_res['Test relative '+scores[0]]
    print(bcolors.BOLD+bcolors.HEADER +f"\nIts (relative) performance on the test set        : "+bcolors.ENDC + f"{s:.3e} ({scores[0]})")

    return df_doe, algos_results, s, best_mdl, best_mdl_name



def classification_doe(df0, df_doe0,
                       target_col,
                       problem_kind,
                       scores,
                       predict_proba=True, class_to_predict=1,
                       oversampling=False, undersampling=False,
                       encode_y=True):
    """Effectue un DOE pour un problème de classification.
    TODO: fusionner les modifs avec regression_doe :
     - [ ] df_doe.loc[ind,'Test adj r2'] : utiliser type de pb passé en param problem_kind= et ne faire cette ligne que si =='regression'
     - [X] 'Ignored train index' : géré
     - [X] problem['Imputer'] : géré
     - [X] initialisations : mieux gérées

    """
    print(f"Size of the classification train set : {df0.shape}")
    algos_results = []
    score_new  = -np.inf
    score_best = -np.inf
    s = None
    best_mdl = None
    best_mdl_name = ''
    df_doe = df_doe0.copy()

    # Boucle sur les problèmes à résoudre, càd les lignes de df_doe
    for ind in df_doe.index:

        problem = df_doe.loc[ind]
        categ_cols    = problem['Variables']['categ_cols']
        num_cols      = problem['Variables']['num_cols']
        categ_encoder = problem['Encoder']
        scaler        = problem['Scaler']
        if 'Imputer' in problem :
            imputer   = problem['Imputer']
        else:
            imputer = None
        algo_info     = problem['Algorithm']
        if 'Ignored train index' in problem:
            ignored_indices_train = problem['Ignored train index']
        else:
            ignored_indices_train = []
        if 'Scoring data' in problem:
            score_data = problem['Scoring data']
            score_function = score_data['score_function']
            greater_is_better = score_data['greater_is_better']
            needs_proba = score_data['needs_proba']
            score_params = score_data['score_params']
            threshold = score_params['threshold']
            alpha = score_params['alpha']
            if score_function == 'roc_auc':
                custom_score = None
                threshold = 0.5
            else:
                custom_score = make_scorer(score_function, greater_is_better=greater_is_better, needs_proba=needs_proba, **score_params)
        else:
            custom_score = None
            threshold = 0.5
            alpha = None
        algo_surname  = algo_info['surname']

        # Ensemble des colonnes formant les features
        feat_cols = categ_cols + num_cols
        
        # Création des matrices et vecteurs de travail 
        X = df0[feat_cols].copy()
        y = df0[target_col].copy()

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18011975, stratify=y)

        # Suppression des index non-désirés dans le dataset d'entraînement
        if ignored_indices_train:
            inds = []
            for ig in ignored_indices_train:
                if ig in X_train.index:
                    inds.append(ig)
            X_train.drop(inds,inplace=True)
            y_train.drop(inds,inplace=True)
            print(f"Elimination of {len(ignored_indices_train)} samples from the train set !")
        print(f"Size of the train set : {X_train.shape}")
        
        # Lance l'algo et renvoie le meilleur pour le score principal spécifié en première position
        mdl, algo_res, preprocessor_X, encoder_y = \
            launch_algo(X_train, y_train, X_test, y_test, 
                        categ_cols, num_cols, target_col,
                        algo_info, scores, problem_kind,
                        categ_encoder=categ_encoder,
                        scaler=scaler,
                        oversampling=oversampling, undersampling=undersampling,
                        predict_proba=predict_proba, class_to_predict=class_to_predict,
                        custom_score=custom_score, threshold=threshold, alpha=alpha,
                        display_results_text=1)
        score_new = algo_res['Validation score']

        # Mémorisation du meilleur modèle
        if score_new>score_best:
            score_best = score_new
            if algo_surname=='NeuralNetwork':
                best_mdl = 'NN' # TODO
            else:
                best_mdl = mdl.best_estimator_
            best_mdl_name = algo_surname

        y_train = pd.DataFrame(y_train,columns=[target_col])
        y_test = pd.DataFrame(y_test,columns=[target_col])

        # Mémorisation des résultats dans la structure algo_results
        algo_res['Column names'] = feature_names_out
        algo_res['X_train'] = X_train
        # algo_res['X_train_enc'] = X_train_enc
        algo_res['y_train'] = y_train
        algo_res['X_test'] = X_test
        # algo_res['X_test_enc'] = X_test_enc
        algo_res['y_test'] = y_test
        algos_results.append(algo_res)

        # Mémorisation des résultats dans le dataframe du DOE
        print(mdl.best_params_)
        df_doe.loc[ind,'Best params'] = f"{mdl.best_params_}"
        df_doe.loc[ind,'Scoring'] = scores[0]
        if not custom_score is None:
            score_function = custom_score
            score_str = 'custom_score'
            for prefix in ['Train ', 'Test ']:
                key = prefix+score_str
                if key in algo_res:
                    df_doe.loc[ind, key] = algo_res[key]
                else:
                    df_doe.loc[ind, key] = "N/A"
        for score in scores:
            if isinstance(score,str):
                score_str = score
            else:
                score_str = 'custom_score'
            for prefix in ['Train ', 'Train relative ',
                          'Test ', 'Test relative ']:
                key = prefix+score_str
                if key in algo_res:
                    df_doe.loc[ind, key] = algo_res[key]
                else:
                    df_doe.loc[ind, key] = "N/A"
        if problem_kind=='regression': df_doe.loc[ind,'Test adj r2'] = algo_res['Test adj r2']
        df_doe.loc[ind,'Elapsed time'] = algo_res['etime']

        pass

    # Affichages
    main_score = scores[0]
    if isinstance(score,str):
        score_str = score
    else:
        score_str = 'custom_score'

    if score_str.startswith('neg_'): score_best = -score_best
    print(bcolors.BOLD+bcolors.OKGREEN+f"\nAnd the Oscar for the best algorithm is awarded to: "+bcolors.ENDC + best_mdl_name + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nfor his role in                                   : "+bcolors.ENDC + f"{best_mdl}" + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nand its (absolute) validation performance of      : "+bcolors.ENDC + f"{score_best:.3e}")
    if problem_kind=='regression':
        s = algo_res['Test relative '+score_str]
        print(bcolors.BOLD+bcolors.HEADER +f"\nIts (relative) performance on the test set        : "+bcolors.ENDC + f"{s:.3e} ({score_str})")

    return df_doe, algos_results, s, best_mdl, best_mdl_name



def impute_df_with_study(df0, df_doe0,
                         imput_col,
                         problem_kind,
                         scores,
                         encode_y=True, excluded_imp_vals=None, data_augm=0):
    """
    Prend un dataframe :
        - sort ses lignes dont la colonne à imputer imput_col a des NaN
        - teste une liste d'algos algo_list
        - réalise une cross validation, avec les paramètres rangés dans le param_dict
        - affiche les résultats en fonction du type de problème problem_kind
        - réalise l'imputation des valeurs manquantes avec l'algo et les hyperparamètres ayant obtenu le meilleur score
        - renvoie le dataframe imputé.
    On peut exclure des valeurs d'entraînement et donc d'imputation avec l'argument optionnel excluded_imp_vals.
    On suppose que le dataframe n'a pas d'autres NaN que ceux de la colonne à imputer.
    """
    print(f"Size of the dataset : {df0.shape}")

    algos_results = []
    score_new  = -np.inf
    score_best = -np.inf
    df_doe = df_doe0.copy()

    # Boucle sur les problèmes à résoudre, càd les lignes de df_doe
    for ind in df_doe.index:
        problem = df_doe.loc[ind]
        categ_cols    = problem['Variables']['categ_cols']
        num_cols      = problem['Variables']['num_cols']
        categ_encoder = problem['Encoder']
        scaler        = problem['Scaler']
        algo_info     = problem['Algorithm']
        algo_surname  = algo_info['surname']

        # Ensemble des colonnes formant les features
        feat_cols = categ_cols + num_cols

        # Mise à l'écart des valeurs d'imputation interdites d'entraînement
        df_work, df_excl = filter_df_with_excluded_vals(df0, imput_col, excluded_imp_vals)

        print(f"df_work : {df_work[imput_col].isna().sum()}")
        print(f"df_excl : {df_excl[imput_col].isna().sum()}")

        # Split train/test : A FAIRE AVANT les transformations, idéalement
        df_known, df_unknown = separate_df_known_unknown(df_work,imput_col)
        print(f"Size of the dataset excluded from imputation     : {df_excl.shape}")
        print(f"Size of the working dataset for the imputation   : {df_known.shape}")
        print(f"Size of the target dataset for the imputation    : {df_unknown.shape}")
        
        # On travaille désormais dans la partie connue
        X = df_known[feat_cols].copy()
        y = df_known[imput_col].copy()

        # Split train/test
        # X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=3) #18011975 , stratify=y_enc
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) #18011975 , stratify=y_enc

        # Augmentation de données
        # X_train, y_train = data_augmentation_mult(X_train, y_train, data_augm)
        if data_augm:
            oversampling = 'SMOTE'
        else:
            oversampling = False

        print(f"problem_kind = {problem_kind}")

        # Lance les algos renvoie le meilleur pour le score spécifié
        mdl, algo_res, preprocessor_X, encoder_y = \
            launch_algo(X_train, y_train, X_test, y_test,
                        categ_cols, num_cols, imput_col, 
                        algo_info, scores, problem_kind,
                        categ_encoder=categ_encoder,
                        scaler=scaler,
                        oversampling=oversampling,
                        predict_proba=False,
                        encode_y=encode_y,
                        display_results_text=2)
        score_new = algo_res['Validation score']
    
        # Mémorisation du meilleur modèle
        if score_new>score_best:
            score_best = score_new
            best_mdl = mdl.best_estimator_
            best_mdl_name = algo_surname

        pass

    # Affichages
    print(bcolors.BOLD+bcolors.OKGREEN+f"\nAnd the Oscar for the best algorithm is awarded to: "+bcolors.ENDC + best_mdl_name + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nfor his role in                                   : "+bcolors.ENDC + f"{best_mdl}" + \
          bcolors.BOLD+bcolors.OKGREEN+f"\nand its (absolute) validation performance of      : "+bcolors.ENDC + f"{score_best:.3e}")

    # Imputation finale
    X_imp = df_unknown[feat_cols].copy()

    y_pred_enc = best_mdl.predict(X_imp)
    if encode_y:
        y_pred = encoder_y.inverse_transform(y_pred_enc)
    else:
        y_pred = y_pred_enc
        
    df_unknown[imput_col] = y_pred

    # On ajoute l'information que la valeur a été imputée, pour les éventuels problèmes liés
    col_IMPUTED = imput_col+'_IMPUTED'
    df_unknown = df_unknown.copy()
    df_known = df_known.copy()
    df_excl = df_excl.copy()
    df_unknown[col_IMPUTED] = 1
    df_known[col_IMPUTED] = 0
    df_excl[col_IMPUTED] = 0

    # On rajoute la partie exclue de l'entraînement
    df = pd.concat([df_excl,df_known,df_unknown],axis=0)
    print(f"Size of the dataset before imputation : {df0.shape}")
    print(f"Size of the dataset after imputation  : {df.shape}")
    
    return df, best_mdl, best_mdl_name, algos_results


def get_n_worst_predictions(n, y_true, y_pred, criteria='neg_mean_absolute_error'):
    score_function = get_scorer(criteria)
    sc = []
    for ind in range(y_true.shape[0]):
        sc.append([y_true.index[ind], score_function._score_func(y_true.iloc[ind], y_pred.iloc[ind])])
    
    sc = np.array(sc)
    top_sc = sc[sc[:,1].argsort()]
    top_sc = top_sc.astype(int)
    top_sc = top_sc[-n:,0]
    worst_pred_indexes = list(top_sc[::-1])
    return worst_pred_indexes


def make_litte_PCA_for_display(X_train, X_test, num_cols):
    X_train_pca = X_train[num_cols].values
    X_test_pca  = X_test[num_cols].values

    names = X_train.index # idCours, ou data.index pour avoir les intitulés
    features = num_cols

    # Centrage et Réduction
    std_scale = StandardScaler().fit(X_train_pca)
    X_train_pca_scaled = std_scale.transform(X_train_pca)
    X_test_pca_scaled  = std_scale.transform(X_test_pca)
    n_comp = 1 # C'est juste pour graphe (x,y) ^^

    # Calcul des composantes principales
    pca = PCA(n_components=n_comp)
    pca.fit(X_train_pca_scaled)
    
    X_train_pca_proj = pca.transform(X_train_pca_scaled)
    X_test_pca_proj  = pca.transform(X_test_pca_scaled)

    X_train_pca_proj = pd.DataFrame(X_train_pca_proj,index=X_train.index)
    X_test_pca_proj  = pd.DataFrame(X_test_pca_proj,index=X_test.index)
    
    return X_train_pca_proj, X_test_pca_proj


def reduce_categ_numeric_info(df0,categ_cols,num_cols,prefix=None):
    """
    Dans le cas de figure où les colonnes `categ_cols` et `num_cols` vont de pair, càd que 
        df.iloc[i,categ_col[j]] = un type donné (catégoriel) et 
        df.iloc[i,num_col[j]]   = valeur associée à ce type
    on réduit la dimension de la matrice finale 
    """
    df = df0.copy()
    
    if not prefix: prefix = ''
        
    # Réunion des valeurs
    val_list = []
    for col in categ_cols:
        val_list.extend(df[col].unique())
    val_list = list(set(val_list))
    
    # Encodage des valeurs
    corresp = {type:ind for ind, type in enumerate(val_list)}
    
    # Init matrice
    m = df.shape[0]   # nombre de lignes
    n = len(val_list) # nombre de colonnes
    M = np.zeros((m, n))
    
    # Boucle sur les samples
    ind = -1
    for index in df.index:
        ind += 1
        for ccol, ncol in zip(categ_cols, num_cols):
            type = df.loc[index,ccol]
            val  = df.loc[index,ncol]
            jnd = corresp[type]
            M[ind,jnd] = val
            
    # Création du dataframe associé à la matrice
    new_cols = [prefix+val for val in val_list]
    df_M = pd.DataFrame(data=M, index=df.index, columns=new_cols)
    
    # Ajout de ce dataframe aux colonnes du dataframe initial
    df = pd.concat([df,df_M], axis=1)
    return df, new_cols

def result_table_wiwo_var(df_doe0, var, scores):
    """
    Crée un table de résultats comparant la présence ou non d'une variable dans un dataset sur les scores, à iso-algo.
    """
    df_doe = df_doe0.copy()
    features = [ line['categ_cols']+line['num_cols'] for line in df_doe['Variables'].values ] 
    df_doe['Features'] = features
    dict_out = {}
    for score in scores:
        selected = np.array([var in feats for feats in df_doe['Features'].values])
        mean_score_wi = df_doe.loc[selected,'Test '+score].mean()
        mean_score_wo = df_doe.loc[~selected,'Test '+score].mean()
        dict_out[score+' gain'] = [mean_score_wi - mean_score_wo]
        
    df_out = pd.DataFrame(data=dict_out)
    return df_out

def iseq(val1,val2):
    return val1==val2

def result_table_compare_2_vals(df_doe0, col, val1, val2, scores, check_class=False):
    """
    Crée un table de résultats comparant 2 valeurs `val1` et `val2` d'une colonne  `col` du plan d'expériences 
    sur les scores. On suppose les algos et variables égalitairement répartis entre avec la `val1` et la `val2`.
    """
    df_doe = df_doe0.copy()
    dict_out = {}
    for score in scores:
        if check_class:
            selected1 = np.array([val.__class__==val1 for val in df_doe[col].values])
        else:
            selected1 = df_doe['Ignored train index'].apply(iseq,val2=val1)
        if check_class:
            selected2 = np.array([val.__class__==val2 for val in df_doe[col].values])
        else:
            selected2 = df_doe['Ignored train index'].apply(iseq,val2=val2)
        mean_score_val1 = df_doe.loc[selected1,'Test '+score].mean()
        mean_score_val2 = df_doe.loc[selected2,'Test '+score].mean()
        dict_out[score+' gain'] = [mean_score_val2 - mean_score_val1]
        
    df_out = pd.DataFrame(data=dict_out)
    return df_out


def result_table_filter_algo(df_doe0, excluded_vals, check_class=False):
    """
    Filtre une table de résultats en excluant les valeurs `excluded_vals` de la colonne `Algorithme` du plan d'expériences 
    """
    df_doe = df_doe0.copy()
    if check_class:
        selected = np.array([algo['class'].__class__ in excluded_vals for algo in df_doe['Algorithm'].values])
    else:
        algo_surname = algo['surname']
        selected = df_doe['Algorithme'].apply(isin,args=excluded_vals)
        
    df_doe = df_doe.loc[~selected]
    return df_doe


def result_table_top_N(df_doe0, score, N):
    """
    Création d'un top N des résiultats d'algorithmes.
    """
    df_doe = df_doe0.sort_values(by=score,ascending=False)
    df_doe = df_doe.iloc[0:N]

    data_dict = {}
    fmt_algos = [algo['surname'] for algo in df_doe['Algorithm'].values]
    data_dict['Algorithm'] = fmt_algos

    fmt_vars  = [vars['categ_cols']+vars['num_cols'] for vars in df_doe['Variables'].values]
    data_dict['Variables'] = fmt_vars

    data_dict['Scaler'] = [sc().__class__.__name__ for sc in df_doe['Scaler'].values]
    
    for feat in ['Ignored train index',
                 'Test neg_root_mean_squared_error',
                 'Test neg_mean_absolute_error',
                 'Test r2',
                 'Test adj r2',
                 'Elapsed time',
                 'Test roc_auc',
                 'Imbalance manager']:
        if feat in df_doe.columns:
            if df_doe[feat].dtype=='object':
                fmt_val = df_doe[feat].values
            else:
                fmt_val = [f"{val:.4f}" for val in df_doe[feat].values]
            data_dict[feat] = fmt_val

    df_out = pd.DataFrame(data=data_dict)
    return df_out
