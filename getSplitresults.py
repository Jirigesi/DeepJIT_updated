import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def resultsAnalyze(test_result_file: str, test_commit_infor_file: str, threshold, characteristic):
    df = pd.read_csv(test_commit_infor_file)
    result = pd.read_csv(test_result_file)

    easy_ids = []
    easy_actual = []
    easy_prediction_prob = []
    easy_prediction = []

    hard_ids = []
    hard_actual = []
    hard_prediction_prob = []
    hard_prediction = []

    ids_test = result['ids']

    for index, id_ in enumerate(ids_test):
        value = df.loc[df['Commit_Hash'] == id_, characteristic].iloc[0]
        if value <= threshold:
            easy_ids.append(id_)
            easy_actual.append(result._get_value(index, 'actual'))
            easy_prediction_prob.append(result._get_value(index, 'prediction_prob'))
            easy_prediction.append(result._get_value(index, 'prediction'))

        else:
            hard_ids.append(id_)
            hard_actual.append(result._get_value(index, 'actual'))
            hard_prediction_prob.append(result._get_value(index, 'prediction_prob'))
            hard_prediction.append(result._get_value(index, 'prediction'))

    print(len(easy_ids), len(hard_ids))

    fpr, tpr, threshold = metrics.roc_curve(easy_actual, easy_prediction)
    easy_roc_auc = metrics.auc(fpr, tpr)

    easy_precision = precision_score(easy_actual, easy_prediction)
    easy_recall = recall_score(easy_actual, easy_prediction)
    easy_f1 = f1_score(easy_actual, easy_prediction)

    # fpr, tpr, threshold = metrics.roc_curve(hard_actual, hard_prediction)
    # hard_roc_auc = metrics.auc(fpr, tpr)
    #
    # hard_precision = precision_score(hard_actual, hard_prediction)
    # hard_recall = recall_score(hard_actual, hard_prediction)
    # hard_f1 = f1_score(hard_actual, hard_prediction)
    hard_roc_auc = 0
    hard_precision = 0
    hard_recall = 0
    hard_f1 = 0

    return (easy_roc_auc,easy_precision,easy_recall, easy_f1, hard_roc_auc, hard_precision, hard_recall, hard_f1)


