import subprocess


for number in range(10, 16):

    cmd = "python main.py -predict -pred_data ./splittedData/hard_Filecount_openstack_test.pkl -buggy_data ./data/train_buggy.pkl -dictionary_data" \
          " ../deepjit_os/openstack_dict.pkl -load_model ./snapshot/2021-04-22_11-22-56/epoch_" + str(number) +".pt "
    subprocess.run(cmd)