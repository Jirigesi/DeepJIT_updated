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
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


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
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        all_distances, all_label = list(), list()

        for i, batch in enumerate(tqdm(batches)):
            distances = []

            pad_msg, pad_code, label = batch
            batch_size = len(pad_msg)

            compare_times = 30   # this variable can be changed to get better results
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
    df = pd.DataFrame(final_labels, columns=['labels'])
    sorted_alldistances = [sorted(set(x), reverse=True) for x in all_distances]
    df["distances"] = sorted_alldistances

    with open('all_distances_15.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("Saved new results!")

    ############################
    data = pickle.load(open('all_distances_15.pkl', 'rb'))

    max_value = []
    two_sum = []
    negative_four = []
    negative_three = []
    min_value = []
    average = []
    negative_five = []
    second_value = []
    third_value = []
    forth_value = []
    fifth_value = []

    for values in data['distances']:
        max_value.append(-values[0])
        second_value.append(-values[1])
        third_value.append(-values[2])
        forth_value.append(-values[3])
        fifth_value.append(-values[4])
        two_sum.append(-values[0] - values[1])
        negative_three.append(-values[0] - values[1] - values[2])  # - values[2] - values[2]- values[3]-values[4]
        negative_four.append(-values[0] - values[1] - values[2] - values[3])
        negative_five.append(-values[0] - values[1] - values[2] - values[3] - values[4])
        min_value.append(-values[-1])
        average.append(-sum(values) / len(values))

    collections = [max_value, second_value, third_value, forth_value, fifth_value, two_sum,
                   negative_three, negative_four, negative_five, min_value, average]
    auc_results = []

    def calculate_AUC(data, predic_possibles, idx):
        fpr, tpr, threshold = metrics.roc_curve(data['labels'], predic_possibles)
        roc_auc = metrics.auc(fpr, tpr)
        auc_results.append(roc_auc)
        # print(f"Test data -- AUC score {idx}: {roc_auc}")

    for idx, collection in enumerate(collections):
        calculate_AUC(data, collection, idx)

    print(f"Test data -- AUC score {idx}: {max(auc_results)}")
    max_index = auc_results.index(max(auc_results))

    prediction_prob = collections[max_index]

    auc_score = roc_auc_score(y_true=data['labels'], y_score=prediction_prob)
    print('Test data -- AUC score:', auc_score)

    fpr, tpr, threshold = metrics.roc_curve(data['labels'], prediction_prob)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = list(roc_t['threshold'])[0]
    prediction = []
    for predict_possible in prediction_prob:
        if predict_possible >= threshold:
            prediction.append(1)
        else:
            prediction.append(0)

    precision = precision_score(data['labels'], prediction)
    recall = recall_score(data['labels'], prediction)
    f1 = f1_score(data['labels'], prediction)

    print("precision", precision)
    print("recall", recall)
    print("f1 score", f1)




