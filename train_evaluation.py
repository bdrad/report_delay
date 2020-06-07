#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.api as sm
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# https://github.com/bdrad/rad_classify
from rad_classify import EndToEndPreprocessor

from utils import get_data_cv, create_output_file, add_results, save_correct_and_incorrect_preds


def main(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--time_delay', help='interpretation_time or total_time', default='interpretation_time', type=str)
    parser.add_argument('--data_path', help='path to datafile', default='', type=str)
    parser = parser.parse_args()

    assert(parser.time_delay in ['interpretation_time', 'total_time'])

    GRID_SEARCH = False
    seed = 42
    WEIGHTS=[1, 1] # [probs_bow, probs_rf] # weights used to average the 2 models in final predictions
    SECTIONS = ['clinical_history'] #'clinical_history', 'impression', 'findings'
    Y_COLUMN_SECONDS = parser.time_delay # interpretation_time', 'total_time'
    if Y_COLUMN_SECONDS == 'interpretation_time'
        DELAY_TIME = int(60 * 60 * 1) # (output units: seconds) <- reports that are more than this value in seconds are "delayed"
    elif Y_COLUMN_SECONDS == 'total_time':
        DELAY_TIME = int(60 * 60 * 3) # (output units: seconds) <- reports that are more than this value in seconds are "delayed"

    DATA_DIR = './data_ct_cv'

    DATA_PATH = parser.data_path

    LOG_DIR = 'results/results_{}_{}_{}min'.format(Y_COLUMN_SECONDS, '_'.join(SECTIONS), DELAY_TIME//60)
    output_file = LOG_DIR + '/output_rf_xgb_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%Hh_%mm_%Ss'))
    try:
        os.mkdir(LOG_DIR)
    except FileExistsError:
        pass


    train, test = get_data_cv(DATA_PATH)
    print('Train {} median: {} minutes'.format(Y_COLUMN_SECONDS, train[Y_COLUMN_SECONDS].median()//60))

    # generate y values
    def create_y(df, y_col):
        return (df[y_col].values > DELAY_TIME).astype(int)


    y_train = create_y(train, Y_COLUMN_SECONDS)
    y_test = create_y(test, Y_COLUMN_SECONDS)

    y_train_seconds = train[Y_COLUMN_SECONDS]
    y_test_seconds = test[Y_COLUMN_SECONDS]


    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))


    try:
        # load reports from saved
        x_train_reports = pd.read_csv(DATA_DIR + '/train_transformed_reports_{}.csv'.format('_'.join(SECTIONS)))['Report Text Transformed'].values.astype(str)
        x_test_reports = pd.read_csv(DATA_DIR + '/test_transformed_reports_{}.csv'.format('_'.join(SECTIONS)))['Report Text Transformed'].values.astype(str)

        train['Report Text Transformed'] = x_train_reports
        test['Report Text Transformed'] = x_test_reports

    except FileNotFoundError:
        # Apply Radlex and CLEVER transofmrations
        time0 = time.time()
        preprocessor = EndToEndPreprocessor(replacement_path="../rad_classify/rad_classify/semantic_dictionaries/clever_replacements",
                                            radlex_path="../rad_classify/rad_classify/semantic_dictionaries/radlex_replacements",
                                            sections=SECTIONS)
        x_train_reports = preprocessor.transform(train['Report Text'].values)
        x_test_reports = preprocessor.transform(test['Report Text'].values)

        print('Time: {:.2f}s'.format(time.time()-time0))

        # Save these tokenized reports in csv
        train['Report Text Transformed'] = x_train_reports
        test['Report Text Transformed'] = x_test_reports

        train.to_csv(DATA_DIR + '/train_transformed_reports_{}.csv'.format('_'.join(SECTIONS)), index=False)
        test.to_csv(DATA_DIR + '/test_transformed_reports_{}.csv'.format('_'.join(SECTIONS)), index=False)


    header = """
    y = {}
    DELAY_TIME = {} minutes
    SECTIONS for NLP: {}
    Date: {}
    WEIGHTS: {}

    """.format(Y_COLUMN_SECONDS, DELAY_TIME//60, SECTIONS, datetime.datetime.now(), WEIGHTS)

    create_output_file(header, output_file)
    print(header)


    ############################################################
    ### Bag of Words with XGBoost Classifier
    ############################################################
    params_bow = {'random_state': seed,
          'n_estimators': 100, # 200 is best
          'n_jobs': -1,
          'max_depth': None,
          'min_samples_split': 16,
          'min_samples_leaf': 1,
          'max_features': 'log2'} #log2 is best

    pipe  = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer())])

    x_train_reports_piped = pipe.fit_transform(x_train_reports)
    x_test_reports_piped = pipe.transform(x_test_reports)



    # GridSearch
    if GRID_SEARCH:
        grid_params = {'n_estimators': [100], # [50, 100, *200] # NOTE: 200 has 0.01 higher roc than 100
                      'max_depth': [None], # [1, 5, 10, 30, 50, 100, *None]
                      'min_samples_split': [16], # [2, 4, 8, *16, 32, 64, 128],
                      'min_samples_leaf': [1], # [0.1, 0.2, 0.3, 0.4, 0.5, *1, 4, 16]
                      'max_features': ['log2'], # [*'log2', 'auto','sqrt']
                      }
        model_bow = RandomForestClassifier(random_state=seed, n_jobs=-1)
        model_bow_gridsearch = GridSearchCV(model_bow, grid_params,
                                            cv=StratifiedKFold(n_splits=5, random_state=seed),
                                            scoring='roc_auc')
        model_bow_gridsearch.fit(x_train_reports_piped, y_train)
        params_bow = {**params_bow, **model_bow_gridsearch.best_params_}


    model_bow = RandomForestClassifier(**params_bow)

    # cross val to eval hyperparams

    probs_bow = cross_val_predict(model_bow, x_train_reports_piped, y_train,
                                   n_jobs=-1, method='predict_proba', verbose=2,
                                   cv=StratifiedKFold(n_splits=5, random_state=seed))
    probs_bow = np.asarray([prob[1] for prob in probs_bow])

    # fit model for later use
    model_bow.fit(x_train_reports_piped, y_train)

    output = add_results(y_train, probs_bow, 'BoW RandomForest CV Results', output_file, params_bow)
    print(output)


    ### Get most important words
    ft_imp = model_bow.feature_importances_
    vocab = pipe.steps[0][1].vocabulary_
    vocab_inv = {v:k for k,v in vocab.items()} # invert vocab dict to search by index

    desc_order_idx = np.argsort(ft_imp)[::-1]

    # save all to csv
    df_vocab = pd.DataFrame(columns=['words', 'importance'])
    df_vocab['words'] = [vocab_inv.get(key) for key in desc_order_idx]
    df_vocab['importance'] = ft_imp[desc_order_idx]
    df_vocab.to_csv(LOG_DIR + '/importance_words.csv', index=False)


    file = open(output_file, 'a')
    file.write('-'*25)
    file.write('\nMost Important words in predicting delayed reporting: \n')
    for idx in desc_order_idx[:10]:
        file.write('{}: {} \n'.format(vocab_inv[idx], round(ft_imp[idx], 4)))

    file.write('-'*25)
    file.write('\nLeast Important Words in predicting delayed reporting: \n')
    for idx in np.argsort(ft_imp)[:10]:
        file.write('{}: {} \n'.format(vocab_inv[idx], ft_imp[idx], 4))
    file.close()




    ##############################
    ### Features CLASSIFIER
    ##############################
    x_cat_features = ['Patient Status numerical',
                      'Body Part Label numerical',
                      'PGY',
                      'Point of Care',
                      'Exam Code',
                      'Time of Day Label numerical']

    x_num_features = []


    # fit encoder
    combined_df = pd.concat((train, test), axis=0)
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(combined_df[x_cat_features].values)

    # get x from df and prefitted encoder
    def get_x_features(df, encoder):
        # one hot encoding of categoriest
        x = encoder.transform(df[x_cat_features].values).toarray()
        # encoder.categories_
        # encoder.get_feature_names()

        # add in the numerical feauturs (no need for one hot encoding)
        x = np.concatenate((x, df[x_num_features].values), axis=1)
        return x


    x_train = get_x_features(train, encoder)
    x_test = get_x_features(test, encoder)

    params_features = {'random_state':seed,
          'learning_rate': 0.5,
          'n_estimators': 100,
          'n_jobs': -1,
          'objective': 'binary:logistic',
          'booster': 'gbtree',
          'max_depth': 5,
          'colsample_bytree': 0.5,
          'gamma': 1,
          'silent': True}


    if GRID_SEARCH:
        grid_params = {'learning_rate': [0.5],
                      'n_estimators': [50, 100],
                      'objective': ['binary:logistic'],
                      'booster': ['gbtree'],
                      'max_depth': [5],
                      'colsample_bytree': [0.5],
                      'gamma': [1]
                      }

        model_features = XGBClassifier(random_state=seed, n_jobs=-1)
        model_features_gridsearch = GridSearchCV(model_features, grid_params,
                                            cv=StratifiedKFold(n_splits=5, random_state=seed),
                                            scoring='roc_auc')
        model_features_gridsearch.fit(x_train, y_train)
        params_features = {**params_features, **model_features_gridsearch.best_params_}


    model_features = XGBClassifier(**params_features)

    # cross val for hyperparameter selection
    probs_features = cross_val_predict(model_features, x_train, y_train,
                                   n_jobs=-1, method='predict_proba', verbose=2,
                                   cv=StratifiedKFold(n_splits=5, random_state=seed))
    probs_features = np.asarray([prob[1] for prob in probs_features])

    # fit model for future predictions
    model_features.fit(x_train, y_train)

    output = add_results(y_train, probs_features, 'Features CV Results', output_file, params_features)
    print(output)


    features = []
    importances = []

    file = open(output_file, 'a')
    file.write('\nFeature Importance: \n')
    for i, x_feature in enumerate(x_num_features[::-1]):
        file.write('{}: {} \n'.format(x_feature, model_features.feature_importances_[-i]))
        features.append(x_feature)
        importances.append(model_features.feature_importances_[-i])

    n = 0
    for i, group in enumerate(encoder.categories_):
        name = x_cat_features[i]

        # uncomment to see individual features
        #print(name)
        #print(model_features.feature_importances_[n:n+len(group)])

        importance = sum(model_features.feature_importances_[n:n+len(group)])
        file.write('{}: {} (sum of {} categories) \n'.format(name, importance, len(group)))
        features.append(name)
        importances.append(importance)
        n += len(group)

    file.close()

    df_ft_imp = pd.DataFrame({'feature': features, 'importance': importances})
    df_ft_imp.to_csv(LOG_DIR + '/importance_features.csv', index=False)





    ############################################################
    ### AVERAGE PROBABILITY PREDICTIONS AND CALCULATE METRICS
    ############################################################
    probs = np.average([probs_bow, probs_features], weights=WEIGHTS, axis=0)

    output = add_results(y_train, probs, 'Combined Models CV Results: Features + BagOfWords', output_file)
    print(output)



    ############################################################
    ### TEST DATA: AVERAGE PROBABILITY PREDICTIONS AND CALCULATE METRICS
    ############################################################
    if Y_COLUMN_SECONDS == 'interpretation_time':
        thresh = 0.2
    elif  Y_COLUMN_SECONDS == 'total_time':
        thresh = 0.5

    probs_bow_test = model_bow.predict_proba(x_test_reports_piped)
    probs_bow_test = np.asarray([prob[1] for prob in probs_bow_test])
    output_bow = add_results(y_test, probs_bow_test, 'BoW Model Results (Test data)', output_file, thresh=thresh)
    print(output_bow)

    probs_features_test = model_features.predict_proba(x_test)
    probs_features_test = np.asarray([prob[1] for prob in probs_features_test])
    output_features = add_results(y_test, probs_features_test, 'Features Model Results (Test data)', output_file, thresh=thresh)
    print(output_features)


    probs_test = np.average([probs_bow_test, probs_features_test], weights=WEIGHTS, axis=0)
    output_combined = add_results(y_test, probs_test, 'Combined Model Results (Test data): Features + BagOfWords', output_file, thresh=thresh)
    print(output_combined)



    ############################################################
    ### SAVE TEST DATA PREDICTIONS
    ############################################################
    save_correct_and_incorrect_preds(y = y_test,
                                     y_probas = probs_test, # it will split correct and incorrect based on this value
                                     y_probas_bow = probs_bow_test,
                                     y_probas_features = probs_features_test,
                                     data = test,
                                     delay_time = DELAY_TIME,
                                     csv_path = LOG_DIR,
                                     dataset_label = 'test'
                                     )



if __name__ == '__main__':
    main()
