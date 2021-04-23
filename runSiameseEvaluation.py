import subprocess

import time

# model_path = "./snapshot/2021-04-22_11-22-56/epoch_"
# model_path = "./snapshot/2021-04-22_15-36-56/epoch_"
# model_path = "./snapshot/2021-04-23_12-02-18/epoch_"
# model_path_list = ["./snapshot/2021-04-22_10-25-57/epoch_", "./snapshot/2021-04-22_11-22-56/epoch_", "./snapshot/2021-04-22_15-36-56/epoch_",
#                    "./snapshot/2021-04-23_12-02-18/epoch_"]
model_path_list = ["./snapshot/2021-04-23_12-02-18/epoch_"]

test_File_path = "./splittedData/hard_Editcount_openstack_test.pkl"

for model_path in model_path_list:
    for number in range(15, 26):

        cmd = "python main.py -predict -pred_data " + test_File_path +"-buggy_data ./data/train_buggy.pkl -dictionary_data" \
              " ../deepjit_os/openstack_dict.pkl -load_model " + model_path + str(number) + ".pt"
        # Wait for 5 seconds

        subprocess.check_output(cmd, shell=True)
        time.sleep(5)
        print("finish evaluate:", number)