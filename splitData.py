import pickle
import pandas as pd


def splitData(commit_infor_file: str, original_file: str, threshold, characteristic):
    df = pd.read_csv(commit_infor_file)

    data = pickle.load(open(original_file, 'rb'))
    ids, labels, msgs, codes = data
    print("Total number of data:", len(ids))

    easy_ids = []
    easy_labels = []
    easy_msgs = []
    easy_codes = []

    hard_ids = []
    hard_labels = []
    hard_msgs = []
    hard_codes = []

    for index, id_ in enumerate(ids):
        try:
            value = df.loc[df['Commit_Hash'] == id_, characteristic].iloc[0]
        except:
            # print(id_)
            continue

        try:
            if value <= threshold:
                easy_ids.append(id_)
                easy_labels.append(labels[index])
                easy_msgs.append(msgs[index])
                easy_codes.append(codes[index])

            else:
                hard_ids.append(id_)
                hard_labels.append(labels[index])
                hard_msgs.append(msgs[index])
                hard_codes.append(codes[index])
        except:
            if int(float(value)) <= threshold:
                easy_ids.append(id_)
                easy_labels.append(labels[index])
                easy_msgs.append(msgs[index])
                easy_codes.append(codes[index])

            else:
                hard_ids.append(id_)
                hard_labels.append(labels[index])
                hard_msgs.append(msgs[index])
                hard_codes.append(codes[index])

    easy_data = (easy_ids, easy_labels, easy_msgs, easy_codes)
    print("Splited the easy part: ", len(easy_ids))

    file1name = 'splittedData/easy_' + characteristic + "_" + original_file.split('/')[-1]

    with open(file1name, 'wb') as handle:
        pickle.dump(easy_data, handle)

    hard_data = (hard_ids, hard_labels, hard_msgs, hard_codes)
    print("Splited the hard part: ", len(hard_ids))
    file2name = 'splittedData/hard_' + characteristic + "_" + original_file.split('/')[-1]
    with open(file2name, 'wb') as handle:
        pickle.dump(hard_data, handle)


if __name__ == "__main__":
    #
    dataType = ['Train', 'Test']
    """
    OS data sources 
    """
    # train_commit_infor_file = "/Users/fjirigesi/Documents/DeepJIT_updated/raw_data/OS/train/Jiri_openstack_train_infor.csv"
    # train_original_file = '/Users/fjirigesi/Documents/DeepJIT_updated/raw_data/OS/train/openstack_train.pkl'
    #
    # test_commit_infor_file = "/Users/fjirigesi/Documents/DeepJIT_updated/raw_data/OS/test/OS_result.csv"
    # test_original_file = "/Users/fjirigesi/Documents/DeepJIT_updated/raw_data/OS/test/openstack_test.pkl"
    #
    # commit_infor_file = (train_commit_infor_file, test_commit_infor_file)
    # original_file = (train_original_file, test_original_file)

    """
    QT data sources
    """
    train_commit_infor_file = "/Users/fjirigesi/Desktop/QT_train_infor.csv"
    train_original_file = '/Users/fjirigesi/Downloads/qt_train.pkl'

    test_commit_infor_file = "/Users/fjirigesi/Documents/defect_prediction_unfaieness-main/DeepJIT/QTresult/Qt_results.csv"
    test_original_file = "/Users/fjirigesi/Downloads/qt_test.pkl"

    commit_infor_file = (train_commit_infor_file, test_commit_infor_file)
    original_file = (train_original_file, test_original_file)

    OS_threshold_dict = {
        "Filecount": 6.58,
        "Editcount": 143.35,
        "MultilineCommentscount": 8.84,
        "Inwards_sum": 22.81,
        "Inwards_avg": 5.5,
        "Outwards_sum": 46.785,
        "Outwards_avg": 11.055
    }

    QT_threshold_dict = {
        # "Filecount": 13.225,
        # "Editcount": 247.35,
        # "MultilineCommentscount": 58.135,
        # "Inwards_sum": 71.715,
        # "Inwards_avg": 22.305,
        # "Outwards_sum": 69.245,
        "Outwards_avg": 15.30
    }


    for idx, datatype in enumerate(dataType):

        for characteristic, threshold in QT_threshold_dict.items():
            print(f"Splitting {datatype} based on {characteristic}...")
            splitData(commit_infor_file[idx], original_file[idx], threshold, characteristic)
            print(f"Finish splitting {datatype}!")
