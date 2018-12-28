#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/12/23 20:06
"""
import torch
import torch.nn as nn
import math
import numpy as np


class PosEmbedding(nn.Module):
    """
    implementation of Positional Embedding to capture the position information of the sequence
    """
    def __init__(self,maxlen,d_model,pad_idx=None):
        super(PosEmbedding,self).__init__()
        weights = self.get_sinusoid_embedding_mat(maxlen,d_model,pad_idx)
        self.pos_embedding = nn.Embedding.from_pretrained(weights,freeze=True)

    def get_sinusoid_embedding_mat(self,max_len,d_model,pad_id =None):
        weights = torch.Tensor([[pos/np.power(10000,2*i/d_model) for i in range(d_model)] for pos in range(max_len+1)])
        weights[:,0::2] = torch.sin(weights[:,::2])
        weights[:,1::2] = torch.cos(weights[:,1::2])
        if pad_id is not None:
            weights[pad_id]=0.
        return weights

    def forward(self, pos_seq):
        return self.pos_embedding(pos_seq)


class ScaledDotProductAttention(nn.Module):
    """
    implementation ScaledDotProductAttention
    """
    def __init__(self,dropout=0):
        super(ScaledDotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v):
        b, l, h = q.size()
        assert k.size()==v.size(), 'key value must be same size'
        scale_factor = math.sqrt(h)
        v = self.dropout(v)
        matmul_out = torch.bmm(q,k.transpose(1,2))/scale_factor
        att_score = torch.softmax(matmul_out,2)
        att_out = torch.bmm(att_score,v)
        return att_out


class MultiHeadAttention(nn.Module):
    """
    implementation multihead attention
    """
    def __init__(self,d_model=512,n_heads=8,d_k=32,d_v=32,drop_out=0.):
        super(MultiHeadAttention,self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.projections = nn.ModuleDict(dict([('q_proj',nn.Linear(d_model,d_k*n_heads)),
                                                ('k_proj',nn.Linear(d_model,d_k*n_heads)),
                                                ('v_proj',nn.Linear(d_model,d_v*n_heads))]))

        self.attention = ScaledDotProductAttention(drop_out)
        self.fc = nn.Linear(n_heads*d_v,d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_out)
        # init
        for k,m in self.projections.items():
            nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q,k,v):
        q_n,q_l,q_h =q.size()
        k_n,k_l,k_h = k.size()
        v_n,v_l,v_h = v.size()
        assert k.size() == v.size(), 'key value must be same size'
        residual = q
        q_proj = self.projections['q_proj'](q).view(q_n, q_l, self.d_k, self.n_heads).permute(0,3,1,2)
        k_proj = self.projections['k_proj'](k).view(k_n, k_l, self.d_k, self.n_heads).permute(0,3,1,2)
        v_proj = self.projections['v_proj'](v).view(v_n, v_l, self.d_v, self.n_heads).permute(0,3,1,2)
        q_proj = q_proj.contiguous().view(-1, q_l, self.d_k)
        k_proj = k_proj.contiguous().view(-1, k_l, self.d_k)
        v_proj = v_proj.contiguous().view(-1, v_l, self.d_v)
        # print('q_size',q_proj.size(),'k_size',k_proj.size(),'v_size',v_proj.size())

        mulhead_out = self.attention(q_proj,k_proj,v_proj)
        # print('mulhead size',mulhead_out.size())
        mulhead_out=mulhead_out.view(q_n,self.n_heads,q_l,self.d_k).permute(0,2,1,3)
        mulhead_out = mulhead_out.contiguous().view(q_n,q_l,-1)
        mulhead_out = self.dropout(mulhead_out)
        fc = self.fc(mulhead_out)
        ret = self.norm(fc+residual)
        return ret


class FeedForwardNet(nn.Module):
    """
    implementation FFN
    """
    def __init__(self,d_in,d_hidden):
        super(FeedForwardNet,self).__init__()
        self.w1 = nn.Conv1d(d_in,d_hidden,1)
        self.w2 = nn.Conv1d(d_hidden,d_in,1)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x):
        residual = x
        x = self.w2(torch.relu(self.w1(x.transpose(1,2))))
        x =x.transpose(1,2)
        ret = self.norm(x+residual)
        return ret


class EncoderLayer(nn.Module):
    """
    implementation EncoderLayer with stacking MultiHeadAttention and FeedForwardNet
    """
    def __init__(self,d_model=512,n_heads=8,d_k=32,d_v=32,d_hidden=256,dropout=0.5):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model,n_heads=n_heads,d_k=d_k,d_v=d_v,drop_out=dropout)
        self.ffn = FeedForwardNet(d_in=d_model,d_hidden=d_hidden)

    def forward(self, x):
        att_v = self.attention(x,x,x)
        ffn_v = self.ffn(att_v)
        return ffn_v


class DecoderLayer(nn.Module):

    def __init__(self,d_model=512,d_hidden=256,n_heads=8,d_k=32,d_v=32,dropout=0.5):
        super(DecoderLayer,self).__init__()
        self.decoder_self_attention = MultiHeadAttention(d_model=d_model,n_heads=n_heads,d_k=d_k,d_v=d_v,drop_out=dropout)
        self.encoder_decoder_attention = MultiHeadAttention(d_model=d_model,n_heads=n_heads,d_k=d_k,d_v=d_v,drop_out=dropout)
        self.ffn = FeedForwardNet(d_in=d_model,d_hidden=d_hidden)

    def forward(self, encodes,decodes):
        att_decodes = self.decoder_self_attention(decodes,decodes,decodes)
        att_enc_dec = self.encoder_decoder_attention(att_decodes,encodes,encodes)
        ret = self.ffn(att_enc_dec)
        return ret






if __name__=='__main__':
    inp = torch.randint(0,100,size=(32,50),dtype=torch.long)
    # oup = torch.randint(0,100,size=(32,40),dtype=torch.long)
    src_word_embedding = nn.Embedding(100,32)
    src_pos_embedding = PosEmbedding(30,32)
    # dst_embedding = nn.Embedding(100,32)
    # inp_embeded = src_embedding(inp)
    # oup_embeded = dst_embedding(oup)


    # m = MultiHeadAttention(n_heads=5,d_model=100,d_k=20,d_v=20)
    # f = FeedForwardNet(100,200)
    # net = DecoderLayer(d_model=32,n_heads=4)
    # net = ScaledDotProductAttention()
    out = src_pos_embedding((torch.LongTensor([[0,1,2,3,4]])))
    # out = net(oup_embeded,inp_embeded)
    # out = f(out)
    print(out.size())