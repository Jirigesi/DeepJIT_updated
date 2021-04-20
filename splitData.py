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
        value = df.loc[df['Commit_Hash'] == id_, characteristic].iloc[0]
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
    train_commit_infor_file = "/Users/fjirigesi/Desktop/Jiri_openstack_train_infor.csv"
    train_original_file = '/Users/fjirigesi/Downloads/deepjit_os/openstack_train.pkl'

    test_commit_infor_file = "/Users/fjirigesi/Documents/defect_prediction_unfaieness-main/DeepJIT/OSresults/OS_result.csv"
    test_original_file = "/Users/fjirigesi/Downloads/deepjit_os/openstack_test.pkl"

    commit_infor_file = (train_commit_infor_file, test_commit_infor_file)
    original_file = (train_original_file, test_original_file)

    threshold = 6.04
    characteristic = "Filecount"

    OS_threshold_dict = {
        "Filecount": 6.04,
        "Editcount": 143.3,
        "MultilineCommentscount": 11.6,
        "Inwards_sum": 15.51,
        "Outwards_sum": 48.04
    }


    for idx, datatype in enumerate(dataType):
        print(f"Splitting {datatype} based on {characteristic}...")
        splitData(commit_infor_file[idx], original_file[idx], threshold, characteristic)
        print(f"Finish splitting {datatype}!")
