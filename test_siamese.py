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
from statistics import mean


def evaluation_siamese_model(data, all_bug_data, params):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    final_labels = labels
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

    with torch.no_grad():
        all_distances, all_label = list(), list()

        for i, batch in enumerate(tqdm(batches)):
            distances = []
            pad_msg, pad_code, label = batch
            batch_size = len(pad_msg)

            compare_times = 5
            while compare_times > 0:

                pad_msg_compare, pad_code_compare, labels_compare = all_bug_data
                pad_msg_compare, pad_code_compare, labels_compare = np.roll(pad_msg_compare, compare_times), \
                                                                                   np.roll(pad_code_compare, compare_times), \
                                                                                   np.roll(labels_compare, compare_times)

                # shuffler = np.random.permutation(len(pad_msg_compare))
                # pad_msg_compare = pad_msg_compare[shuffler]
                # pad_code_compare = pad_code_compare[shuffler]
                # labels_compare = labels_compare[shuffler]

                compare_batches = mini_batches_test(X_msg=pad_msg_compare, X_code=pad_code_compare, Y=labels_compare, mini_batch_size=batch_size)

                for j, compare_batch in enumerate(compare_batches):
                    pad_msg_compare, pad_code_compare, label_compare = compare_batch

                    if torch.cuda.is_available():
                        pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                            pad_code).cuda(), torch.cuda.FloatTensor(label)
                        pad_msg_compare, pad_code_compare, label_compare = torch.tensor(pad_msg_compare).cuda(), torch.tensor(
                            pad_code_compare).cuda(), torch.cuda.FloatTensor(label_compare)
                    else:
                        pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                            labels).float()

                    if torch.cuda.is_available():

                        if len(pad_msg) == len(pad_msg_compare):

                            output1, output2 = model.forward(pad_msg, pad_code, pad_msg_compare, pad_code_compare)
                            eucledian_distance = F.pairwise_distance(output1, output2)
                            eucledian_distance = eucledian_distance.cpu().numpy()

                    for index, distance_value in enumerate(eucledian_distance):
                        try:
                            distances[index].append(distance_value)
                        except IndexError:
                            distances.append([distance_value])

                compare_times -= 1

            all_distances.extend(distances)

    final_labels = final_labels.tolist()

    preds_max = []
    preds_avg = []
    for distance in all_distances:
        max_value = max(distance)
        preds_max.append(max_value)

        avg_value = sum(distance)/len(distance)
        preds_avg.append(avg_value)


    fpr, tpr, threshold = metrics.roc_curve(final_labels, preds_max)

    roc_auc = metrics.auc(fpr, tpr)
    print('Test data -- AUC score:', roc_auc)


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
    # with open('all_distances.pkl', 'wb') as f:
    #     pickle.dump(all_distances, f)

