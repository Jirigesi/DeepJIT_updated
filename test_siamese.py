from siamese import DeepJITSiamese
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import numpy as np
import torch.nn.functional as F
import random


def evaluation_siamese_model(data, all_bug_data, params):
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

    model = DeepJITSiamese(args=params)

    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # pad_msg_compare, pad_code_compare, labels_compare = all_bug_data
    # need to shuffle here multiple times
    # all_bug_data_zip = list(zip(pad_msg_compare, pad_code_compare, labels_compare))
    # random.shuffle(all_bug_data_zip)
    # pad_msg_compare, pad_code_compare, labels_compare = zip(*all_bug_data_zip)
    # pad_msg_compare, pad_code_compare, labels_compare = np.array(pad_msg_compare), np.array(pad_code_compare), np.array(labels_compare)
    # compare_batches = mini_batches_test(X_msg=pad_msg_compare, X_code=pad_code_compare, Y=labels_compare)

    with torch.no_grad():
        all_distances, all_label = list(), list()

        for i, batch in enumerate(batches):
            distances = []
            pad_msg, pad_code, label = batch
            batch_size = len(pad_msg)

            pad_msg_compare, pad_code_compare, labels_compare = all_bug_data
            shuffler = np.random.permutation(len(pad_msg_compare))
            pad_msg_compare = pad_msg_compare[shuffler]
            pad_code_compare = pad_code_compare[shuffler]
            labels_compare = labels_compare[shuffler]

            print("need to batch size:", batch_size)
            compare_batches = mini_batches_test(X_msg=pad_msg_compare, X_code=pad_code_compare, Y=labels_compare, mini_batch_size=batch_size)

            for j, compare_batch in enumerate(compare_batches):
                print("batches times", i)
                print("compartive times:", j)
                pad_msg_compare, pad_code_compare, label_compare = compare_batch
                print("batch size:", len(pad_msg))
                print("compared batch size:", len(pad_msg_compare))

                if torch.cuda.is_available():
                    pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                        pad_code).cuda(), torch.cuda.FloatTensor(label)
                    pad_msg_compare, pad_code_compare, label_compare = torch.tensor(pad_msg_compare).cuda(), torch.tensor(
                        pad_code_compare).cuda(), torch.cuda.FloatTensor(label_compare)
                else:
                    pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                        labels).float()

                if torch.cuda.is_available():
                    # todo: need to fix here, since lost last batch
                    if len(pad_msg) == len(pad_msg_compare):

                        output1, output2 = model.forward(pad_msg, pad_code, pad_msg_compare, pad_code_compare)
                        print("output1 length", output1.size())
                        print("output2 length", output2.size())
                        eucledian_distance = F.pairwise_distance(output1, output2)
                        eucledian_distance = eucledian_distance.cpu().numpy()

                for index, distance_value in enumerate(eucledian_distance):
                    try:
                        distances[index].append(distance_value)
                    except IndexError:
                        distances.append([distance_value])

            all_distances.extend(distances)

    print(all_distances)
    print(len(all_distances))



                # else:
                #     predict = model.forward(pad_msg, pad_code)
                #     predict = predict.detach().numpy().tolist()

    #         all_predict += predict
    #         all_label += labels.tolist()
    #
    # # write data in a file.
    # with open('siamese_result.txt', 'w') as filehandle:
    #     for listitem in all_predict:
    #         filehandle.write('%s\n' % listitem)



    # auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    # print('Test data -- AUC score:', auc_score)
    #
    # fpr, tpr, threshold = metrics.roc_curve(all_label, all_predict)
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    # roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    # threshold = list(roc_t['threshold'])[0]
    # prediction = []
    # for predict_possible in all_predict:
    #     if predict_possible >= threshold:
    #         prediction.append(1)
    #     else:
    #         prediction.append(0)
    #
    # df = pd.DataFrame(all_label, columns=["actual"])
    # df["prediction_prob"] = all_predict
    # df["prediction"] = prediction
    # df.to_csv('jiri_result.csv', index=False)
    #
    # print(classification_report(all_label, prediction))

