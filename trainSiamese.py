from siamese import DeepJITSiamese
import torch
from tqdm import tqdm
from utils import mini_batches_train, save
import torch.nn as nn
import os, datetime
from contrastiveLoss import ContrastiveLoss


def train_model_siamese(data, params):
    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DeepJITSiamese(args=params)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    criterion = ContrastiveLoss()

    loss = []
    counter = []
    iteration_number = 0

    for epoch in range(1, params.num_epochs + 1):
        for i in range(len(data_labels) - 1):
            data_pad_msg1 = data_pad_msg[i]
            data_pad_code1 = data_pad_msg[i]
            data_labels1 = data_labels[i]
            data_pad_msg1, data_pad_code1, data_labels1 = torch.tensor(data_pad_msg1).cuda(), torch.tensor(
                data_pad_code1).cuda(), torch.cuda.FloatTensor(data_labels1)
            for j in range(i + 1, len(data_labels)):
                data_pad_msg2 = data_pad_msg[i]
                data_pad_code2 = data_pad_msg[i]
                data_labels2 = data_labels[i]
                data_pad_msg2, data_pad_code2, data_labels2 = torch.tensor(data_pad_msg2).cuda(), torch.tensor(
                    data_pad_code2).cuda(), torch.cuda.FloatTensor(data_labels2)

                if torch.equal(data_labels1, data_labels2):
                    temp_label = torch.tensor(1).cuda()
                else:
                    temp_label = torch.tensor(0).cuda()

                optimizer.zero_grad()
                output1, output2 = model.forward(data_pad_msg1, data_pad_code1, data_pad_msg2, data_pad_code2)
                loss_contrastive = criterion(output1, output2, temp_label)
                loss_contrastive.backward()
                optimizer.step()

        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())

        # print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)

