import subprocess

import time

# model_path = "./snapshot/2021-04-22_11-22-56/epoch_"
# model_path = "./snapshot/2021-04-22_15-36-56/epoch_"
# model_path = "./snapshot/2021-04-23_12-02-18/epoch_"
# model_path_list = ["./snapshot/2021-04-22_10-25-57/epoch_", "./snapshot/2021-04-22_11-22-56/epoch_", "./snapshot/2021-04-22_15-36-56/epoch_",
#                    "./snapshot/2021-04-23_12-02-18/epoch_"]
# model_path_list = ["./snapshot/2021-04-23_15-13-25/epoch_", "./snapshot/2021-04-23_20-54-01/epoch_"]
# model_path_list = ["./snapshot/2021-04-24_12-46-04/epoch_", "./snapshot/2021-04-24_15-08-57/epoch_",
#                    "./snapshot/2021-04-24_19-31-41/epoch_", "./snapshot/2021-04-24_22-36-20/epoch_"]
model_path_list = ["./snapshot/2021-04-25_12-38-39/epoch_"]

# test_File_path = "./splittedData/hard_Outwards_sum_openstack_test.pkl"

test_File_path_list = ["./splittedData/hard_Filecount_openstack_test.pkl", "./splittedData/hard_Editcount_openstack_test.pkl",
                       "./splittedData/hard_MultilineCommentscount_openstack_test.pkl", "./splittedData/hard_Inwards_sum_openstack_test.pkl",
                       "./splittedData/hard_Outwards_sum_openstack_test.pkl"]


for test_File_path in test_File_path_list:
    for model_path in model_path_list:
        for number in range(15, 26):

            cmd = "python main.py -predict -pred_data " + test_File_path + " " + "-buggy_data ./data/train_buggy.pkl -dictionary_data" \
                  " ../deepjit_os/openstack_dict.pkl -load_model " + model_path + str(number) + ".pt"
            # Wait for 5 seconds

            subprocess.check_output(cmd, shell=True)
            time.sleep(5)
            print("finish evaluate:", number)