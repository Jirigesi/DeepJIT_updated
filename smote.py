from imblearn.over_sampling import SMOTE
import pickle
import pandas as pd

def sampleData(commit_infor_file: str, original_file: str, threshold, characteristic):
    df = pd.read_csv(commit_infor_file)

    original_data = pickle.load(open(original_file, 'rb'))
    ids, labels, msgs, codes = original_data
    print("Total number of data:", len(ids))

    hard_easy_label = []
    for index, id_ in enumerate(ids):
        value = df.loc[df['Commit_Hash'] == id_, characteristic].iloc[0]
        if value <= threshold:
            hard_easy_label.append(1)

        else:
            hard_easy_label.append(0)

    # transform the dataset
    over_samples = SMOTE()
    # Resample on training data
    resamples_X, resamples_y = over_samples.fit_resample(original_data, hard_easy_label)

    filename = 'Resampled_' + characteristic + "_" + original_file.split('/')[-1]

    with open("data/"+ filename, 'wb') as handle:
        pickle.dump(resamples_X, handle)
    print(f"Finish resampling: \n Original data size: {len(original_data[0])} \nNew data size: {len(resamples_X[0])}")

if __name__ == "__main__":
    # give label 0, 1 to commit for characteristics
    OS_threshold_dict = {
        "Filecount": 6.04,
        "Editcount": 143.3,
        "MultilineCommentscount": 11.6,
        "Inwards_sum": 15.51,
        "Outwards_sum": 48.04
    }

    train_infor_file = "/Users/fjirigesi/Desktop/Jiri_openstack_train_infor.csv"
    train_file = '/Users/fjirigesi/Downloads/deepjit_os/openstack_train.pkl'

    sampleData(train_infor_file, train_file, 6.04, "Filecount")
