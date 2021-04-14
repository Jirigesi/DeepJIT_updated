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
        # building batches for training model
        batches1 = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        for _ in range(30):
            batches2 = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
            for i, (batch1, batch2) in enumerate(tqdm(zip(batches1, batches2))):
                pad_msg1, pad_code1, labels1 = batch1
                pad_msg2, pad_code2, labels2 = batch2
                if torch.cuda.is_available():
                    pad_msg1, pad_code1, labels1 = torch.tensor(pad_msg1).cuda(), torch.tensor(
                        pad_code1).cuda(), torch.cuda.FloatTensor(labels1)
                    pad_msg2, pad_code2, labels2 = torch.tensor(pad_msg2).cuda(), torch.tensor(
                        pad_code2).cuda(), torch.cuda.FloatTensor(labels2)
                else:
                    pad_msg1, pad_code1, labels1 = torch.tensor(pad_msg1).long(), torch.tensor(pad_code1).long(), torch.tensor(
                        labels1).float()
                    pad_msg2, pad_code2, labels2 = torch.tensor(pad_msg2).long(), torch.tensor(
                        pad_code2).long(), torch.tensor(labels2).float()

                temp_labels = []
                for i in range(len(labels1)):
                    if labels1[i] == labels2[i]:
                        temp_labels.append(1)
                    else:
                        temp_labels.append(0)
                temp_labels = torch.cuda.FloatTensor(temp_labels)

                optimizer.zero_grad()
                output1, output2 = model.forward(pad_msg1, pad_code1, pad_msg2, pad_code2)
                # print("output1 shape:", output1.shape)
                loss_contrastive = criterion(output1, output2, temp_labels)
                loss_contrastive.backward()
                optimizer.step()

        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())

        # print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)

