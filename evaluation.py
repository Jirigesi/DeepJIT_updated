from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import numpy as np
from getSplitresults import resultsAnalyze
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import csv


def evaluation_model(data, params, ids):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, label = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:                
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)

    fpr, tpr, threshold = metrics.roc_curve(all_label, all_predict)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = list(roc_t['threshold'])[0]
    prediction = []
    for predict_possible in all_predict:
        if predict_possible >= threshold:
            prediction.append(1)
        else:
            prediction.append(0)

    df = pd.DataFrame(all_label, columns=["actual"])
    df["prediction_prob"] = all_predict
    df["prediction"] = prediction
    df["ids"] = ids
    df.to_csv('siameseEasy_total.csv', index=False)

    print("----------total result----------")

    total_precision = precision_score(df["actual"], df["prediction"])
    total_recall = recall_score(df["actual"], df["prediction"])
    total_f1 = f1_score(df["actual"], df["prediction"])

    print("----------split result----------")
    print("#####Easy part######")

    test_commit_infor_file = "data/OS_result.csv"
    test_result_file = "siameseEasy_total.csv"

    OS_threshold_dict = {
        # "Filecount": 6.04,
        # "Editcount": 143.3,
        # "MultilineCommentscount": 11.6,
        # "Inwards_sum": 15.51,
        "Outwards_sum": 48.04
    }

    for character_name, threshold in OS_threshold_dict.items():

        seperate_result = resultsAnalyze(test_result_file, test_commit_infor_file, threshold, character_name)
        easy_roc_auc, easy_precision, easy_recall, easy_f1, hard_roc_auc, hard_precision, hard_recall, hard_f1 = seperate_result

        with open('siameseEasy_results.csv', mode='a') as result_file:
            result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            result_writer.writerow([character_name, auc_score, total_precision, total_recall, total_f1, easy_roc_auc,
                                    easy_precision, easy_recall, easy_f1, hard_roc_auc, hard_precision, hard_recall, hard_f1])

        print(f"Wrote results for {character_name}")
