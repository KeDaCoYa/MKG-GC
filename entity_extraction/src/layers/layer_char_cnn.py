"""class implements character-level convolutional 1D layer"""
import torch
import torch.nn as nn

from ipdb import set_trace
class LayerCharCNN(nn.Module):
    """LayerCharCNN implements character-level convolutional 1D layer."""
    def __init__(self, char_embeddings_dim, filter_num, char_window_size, word_len):
        '''

        '''
        super(LayerCharCNN, self).__init__()
        self.char_embeddings_dim = char_embeddings_dim  #为25

        self.char_cnn_filter_num = filter_num  # 30
        self.char_window_size = char_window_size  # 默认值一般为3
        self.word_len = word_len  # 20
        self.conv_feature_len = word_len - char_window_size + 1
        self.output_dim = char_embeddings_dim * filter_num
        # 一维卷积操作，输入通道就是char embedding dim

        # 经过这个卷积操作，输入是(batch_size,seq_len,embedding_dim) -> (batch_size,m)
        # out_channels = 25*30 = 750
        self.conv1d = nn.Conv1d(in_channels=char_embeddings_dim,
                                out_channels=char_embeddings_dim * filter_num,
                                kernel_size=char_window_size,
                                groups=char_embeddings_dim)
        # 这里gropus=25，相当于对每一维使用不同的卷积核进行操作


    def is_cuda(self):
        return self.conv1d.weight.is_cuda

    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len=[100, 47, 25, 20]
        '''


        '''
        batch_num, max_seq_len, char_embeddings_dim, word_len = char_embeddings_feature.shape

        #max_pooling_out.shape = (batch_size,max_batch_len,30*25=750)=(100,47,750)
        # 这是对seq_len进行一个
        max_pooling_out = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len,self.char_cnn_filter_num * self.char_embeddings_dim, dtype=torch.float))
        # 这里进行最大池化操作
        for k in range(max_seq_len):
            # 相当于让卷积一次处理的char_embeddings_feature[:, k, :, :].shape=[100, 25, 20]=(batch_size,character_dim,word_len)
            tmp_cnn = self.conv1d(char_embeddings_feature[:, k, :, :]) #tmp_cnn.shape=(batch_size,output_channels=newd_im,18(卷积之后的seq_len))=(100,750,18)
            max_pooling_out[:, k, :], _ = torch.max(tmp_cnn, dim=2)

        return max_pooling_out # shape: batch_num x max_seq_len x filter_num*char_embeddings_dim=[100, 47, 750]
