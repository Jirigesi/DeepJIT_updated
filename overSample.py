from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from collections import Counter
import pickle


def label_hard_easy(commit_infor_file: str, original_file: str, threshold, characteristic):
    df = pd.read_csv(commit_infor_file)
    data = pickle.load(open(original_file, 'rb'))
    ids, labels, msgs, codes = data
    easy_hard_label = []

    not_found = []
    for index, id_ in enumerate(ids):
        try:
            value = df.loc[df['Commit_Hash'] == id_, characteristic].iloc[0]
            if value <= threshold:
                easy_hard_label.append(0)

            else:
                easy_hard_label.append(1)
        except:
            not_found.append(id_)




    print('easy_hard_label:', len(easy_hard_label))
    print('not found:', len(not_found))

    return easy_hard_label, not_found

if __name__ == "__main__":
    # train_commit_infor_file = "/Users/fjirigesi/Desktop/Jiri_openstack_train_infor.csv"
    # train_original_file = '/Users/fjirigesi/Downloads/deepjit_os/openstack_train.pkl'

    train_commit_infor_file = "/Users/fjirigesi/Desktop/QT_train_infor.csv"
    train_original_file = '/Users/fjirigesi/Downloads/qt_train_new.pkl'

    data = pickle.load(open(train_original_file, 'rb'))
    easy_hard_label, not_found = label_hard_easy(train_commit_infor_file, train_original_file, 6.58, "Filecount")

    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')

    # oversample = SMOTE()

    ids, labels, msgs, codes = data

    df = pd.DataFrame(ids, columns=['ids'])
    df['labels'] = labels
    df['msgs'] = msgs
    df['codes'] = codes

    # summarize class distribution
    print(Counter(easy_hard_label))

    # fit and apply the transform
    data_over, easy_hard_label_over = oversample.fit_resample(df, easy_hard_label)
    print(Counter(easy_hard_label_over))

    new_ids = data_over['ids'].tolist()
    new_labels = data_over['labels'].tolist()
    new_msgs = data_over['msgs'].tolist()
    new_codes = data_over['codes'].tolist()

    over_data = (new_ids, new_labels, new_msgs, new_codes)

    filename = "data/over_train_QT_data.pkl"

    with open(filename, 'wb') as handle:
        pickle.dump(over_data, handle)
        print(f"Saved in {filename}")

