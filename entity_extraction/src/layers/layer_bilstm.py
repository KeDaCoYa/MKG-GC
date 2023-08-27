"""class implements standard bidirectional LSTM recurrent layer"""
import torch
import torch.nn as nn


from ipdb import set_trace
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LayerBiLSTM(nn.Module):
    """BiLSTM layer implements standard bidirectional LSTM recurrent layer"""
    def __init__(self, input_dim, hidden_dim):
        super(LayerBiLSTM, self).__init__()
        self.num_layers = 1
        self.num_directions = 2
        rnn = nn.LSTM(input_size=input_dim,
                      hidden_size=hidden_dim,
                      num_layers=1,
                      batch_first=True,
                      bidirectional=True)
        self.rnn = rnn

    def lstm_custom_init(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.bias_hh_l0_reverse.data.fill_(0)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_ih_l0_reverse.data.fill_(0)
        # Init forget gates to 1
        for names in self.rnn._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, input_tensor, mask_tensor):
        '''
            input_tensor shape: batch_size x max_seq_len x dim=(100,47,850)
            mask_tensor=torch.Size([100, 47])

        '''
        batch_size, max_seq_len, _ = input_tensor.shape

        input_packed, reverse_sort_index = self.pack(input_tensor, mask_tensor)
        # 这里感觉就是初始化为0，感觉并不需要(h0,c0),反正都是0
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        # 全部初始化为0
        output_packed, _ = self.rnn(input_packed, (h0, c0))

        output_tensor = self.unpack(output_packed, max_seq_len, reverse_sort_index) # shape: batch_size x max_seq_len x hidden_dim*2
        return output_tensor

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
    def pack(self, input_tensor, mask_tensor):
        # seq_len_list获得每个sequence的真实长度
        seq_len_list = self.get_seq_len_list_from_mask_tensor(mask_tensor)

        sorted_seq_len_list, sort_index, reverse_sort_index = self.sort_by_seq_len_list(seq_len_list)
        # 按照sort_index来获得排序后的文本序列
        input_tensor_sorted = torch.index_select(input_tensor, dim=0, index=sort_index)
        res = pack_padded_sequence(input_tensor_sorted, lengths=sorted_seq_len_list, batch_first=True)
        return res,reverse_sort_index

    def unpack(self, output_packed, max_seq_len, reverse_sort_index):

        output_tensor_sorted, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=max_seq_len)
        # 将数据还原回去
        output_tensor = torch.index_select(output_tensor_sorted, dim=0, index=reverse_sort_index)
        return output_tensor