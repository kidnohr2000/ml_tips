import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils

class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(self,batch_size,lstm_hid_dim,d_a,r,max_len,feat_dim,device,type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper

        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            feat_dim     : {int} number of feature dim
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self

        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()

        self.dropout = 0.0
        self.device = device
#         self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,1,batch_first=True)
        self.bilstm1 = nn.LSTM(feat_dim, lstm_hid_dim, dropout=self.dropout, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(lstm_hid_dim*2, lstm_hid_dim, dropout=self.dropout, bidirectional=True, batch_first=True)

        self.linear_first = torch.nn.Linear(lstm_hid_dim*2,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim*2,n_classes)
        self.linear_final1 = torch.nn.Linear(lstm_hid_dim*2,d_a)
        self.linear_final2 = torch.nn.Linear(d_a,self.n_classes)
        self.batch_size = batch_size
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state1 = self.init_hidden()
        self.hidden_state2 = self.init_hidden()
        self.r = r
        self.type = type


    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n

        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors


        """

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)


    def init_hidden(self):
        return (
            Variable(torch.zeros(2,self.batch_size,self.lstm_hid_dim)).to(self.device),
            Variable(torch.zeros(2,self.batch_size,self.lstm_hid_dim)).to(self.device)
        )


    def forward(self,x):
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        outputs, self.hidden_state1 = self.bilstm1(x, self.hidden_state1)
        outputs, self.hidden_state2 = self.bilstm2(outputs, self.hidden_state2)
        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x,1)
        attention = x.transpose(1,2)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # attentionあり↓
        att_out = attention@outputs
#         outputs = torch.mean(outputs, dim=1)
#         att_out = F.relu(self.linear_final1(outputs))
#         att_out = F.sigmoid(self.linear_final2(att_out))
        att_out = torch.sum(att_out,1)/self.r
        att_out = F.sigmoid(self.linear_final(att_out))

        return att_out,attention


    #Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation

        Args:
           m: {Variable} ||AAT - I||

        Returns:
            regularized value


        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor)