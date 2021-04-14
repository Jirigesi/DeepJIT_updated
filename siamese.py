import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepJITSiamese(nn.Module):
    def __init__(self, args):
        super(DeepJITSiamese, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks]) # 4 dimentional
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line\
        # print("x shape[0], [1]: ", x.shape[0], x.shape[1], x.shape[2], x.shape[0])
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        print("x type:", type(x))

        # jiri made change here from 3 to 1
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        # jiri made change here from 2 to 1
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):

        print("x shape[0], [1]: ", x.shape[0], x.shape[1], x.shape[2], x.shape[0])

        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[0])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward_once(self, msg, code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        # remove this step, direct use results from fc2 as output
        # out = self.sigmoid(out).squeeze(1)
        return out

    def forward(self, msg1, code1, msg2, code2):
        # forward pass of input 1
        output1 = self.forward_once(msg1, code1)
        # forward pass of input 2
        output2 = self.forward_once(msg2, code2)

        return output1, output2