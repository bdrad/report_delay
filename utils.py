
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


def get_data_cv(datadir):
    # Used with Cross Validation
    # returns train/test split  ONLY of df (know val - see below for validation)
    seed = 42
    test_size = 0.15

    # load data
    cols = ['Minimum of Exam Ordered to Prelim/First Com',
            'Minimum of Exam Completed to Prelim/First Com',
            'Report Text',
            'Patient Status',
            'Patient Status numerical',
            'Time of Day Label',
            'Time of Day Label numerical',
            'Body Part Label numerical',
            'Preliminary Report By',
            'Preliminary Report By numerical',
            'Preliminary Report Date',
            'Point of Care',
            'Exam Code']
    df = pd.read_csv(datadir, usecols=cols)
    df_pgy = pd.read_excel('./trainees.xlsx', header=0)

    df = df.merge(df_pgy,
                  left_on='Preliminary Report By',
                  right_on='Preliminary Report By')

    df = df[df['PGY'] != 2]

    df['interpretation_time'] = df['Minimum of Exam Completed to Prelim/First Com']
    df['total_time'] = df['Minimum of Exam Ordered to Prelim/First Com']

    # Drop rows with na
    df = df.dropna(subset=cols,axis=0)


    # train/val/test split
    train, test = train_test_split(df, test_size=test_size, shuffle=True, random_state=seed)
    return train, test


def create_output_file(header, output_file):
    file = open(output_file, 'w+')
    file.write(header)
    file.close()
    return header

def add_scores(scores, title, output_file, params):
    output = """
--------------------------------------------------
{}
--------------------------------------------------
accuracy: {}
f1 mean: {}
roc_auc mean: {}
params: {}
""".format(title,
     scores['test_accuracy'].mean(),
     scores['test_f1'].mean(),
     scores['test_roc_auc'].mean(),
     params)

    file = open(output_file, 'a')
    file.write(output)
    file.close()
    return output

def add_results(y_test, y_probas, title, output_file, params=None, thresh=0.5):
    y_preds = np.where(y_probas >= thresh, 1, 0).astype(int)

    results = {}
    results['accuracy_score'] = accuracy_score(y_test, y_preds)
    results['recall'] = recall_score(y_test, y_preds)
    results['precision'] = precision_score(y_test, y_preds)
    results['f1'] = f1_score(y_test, y_preds)
    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    results['specificity'] = tn / (tn+fp)
    results['sensitivity'] = tp / (tp+fn)
    results['roc'] = roc_auc_score(y_test, y_probas)

    output = """
--------------------------------------------------
{}
--------------------------------------------------
{}
{}
{}

""".format(
    title,
    cm,
    results,
    params
    )

    file = open(output_file, 'a')
    file.write(output)
    file.close()

    return output


def save_correct_and_incorrect_preds(y, y_probas, y_probas_bow, y_probas_features, data, delay_time, csv_path, dataset_label):
    y_preds = np.where(y_probas >= 0.5, 1, 0).astype(int)
    save_df = pd.DataFrame()
    save_df['delay_>_{}min'.format(delay_time//60)] = y
    save_df['pred_delay'] = y_preds
    save_df['pred_delay_probability'] = y_probas
    save_df['pred_delay_probability_text'] = y_probas_bow
    save_df['pred_delay_probability_features'] = y_probas_features
    save_df['report_text'] = data['Report Text'].values
    save_df['report_text_transformed'] = data['Report Text Transformed'].values
    save_df.to_csv(csv_path + '/{}_predictions.csv'.format(dataset_label), index=False)

    incorrect_df = save_df[y_preds != y]
    incorrect_df.to_csv(csv_path + '/{}_predictions_incorrect.csv'.format(dataset_label), index=False)

    correct_df = save_df[y_preds == y]
    correct_df.to_csv(csv_path + '/{}_predictions_correct.csv'.format(dataset_label), index=False)

    return 'Saved'
