import torch 
import torch.nn as nn
import math
import numpy as N
#from tflearn.layers.core import dropout

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):#dimension of vector
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model).float())

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)#create matrix of shape (sequencelen,dmodel)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)#apply sin to even position
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#(1,seq len,d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
        return self.dropout(x)
class LayerNormalization(nn.Module):
    def __init__(self, eps:float=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))#multiplied
        self.beta = nn.Parameter(torch.zeros(1))#added
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta   
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)#wiand b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)#w2 and b2
    def forward(self, x):
        #(batch,seq_len,d_model) --->  (batch,sq_len,dff)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float=0.1):#d model 512,h =no of head
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h
        #self.d_v = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        #d_k = query.size(-1)
        d_k = query.size(-1)
        #(batch, h, seq_len, d_k)-->(batch, h, seq_len, seq_len)
      
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        #before apply softmask we need to apply mask to score
       
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)#replace 0 with small so 
        attn_scores = torch.softmax(attn_scores, dim=-1)#(batch, h, seq_len, seq_len)
        if dropout is not None:
            attn_scores = dropout(attn_scores)
       # p_attn = dropout(p_attn)
        return torch.matmul(attn_scores, value), attn_scores
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)#(batch, seq_len, d_model)-->(batch, seq_len, d_model)
        key = self.w_k(k)#(batch, seq_len, d_model)-->(batch, seq_len, d_model)
        key = self.w_k(k)
        #key = self.w_k(k)
        value = self.w_v(v)#(batch, seq_len, d_model)-->(batch, seq_len, d_model)
        #(batch, seq_len, d_model)-->(batch, seq_len, h, d_k)-->(batch, h, seq_len, d_k)
       # query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transporse(1, 2)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attn_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        #(batch, h, seq_len, d_k)-->(batch, seq_len, h, d_k)-->(batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0],-1, self.d_model)
        #x = x.transpose(1, 2).contiguous().view(x.shape[0],-1, self.h*self.d_k)
        #(batch, seq_len, d_model)-->(batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttention, feed_forward:FeedForward, dropout:float=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, mask):#mask apply input of encoder bcz no need to intract padding word to otherword
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Decoderblock(nn.Module):
    def __init__(self, self_attenntion_block:MultiHeadAttention,cross_attension_block:MultiHeadAttention,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention = self_attenntion_block
        self.cross_attention = cross_attension_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x=self.residual_connections[2](x, self.feed_forward)
        return x
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
class Projectionlayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        #(batch,sequence,dmodel) --> (batch,seqlen,vocabsize)
        return torch.log_softmax(self.projection(x), dim=-1)
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection:Projectionlayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    def project(self, x):
        return self.projection(x)
    

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=.1,d_ff:int=2048):
    
   #create the embedding layers
    src_embed=InputEmbedding(d_model, src_vocab_size)
    tgt_embed=InputEmbedding(d_model, tgt_vocab_size)
    #create the positional encoding layers
    #src_pos=PositionalEncoding(d_model, src_seq_len, dropout)
    src_pos=PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos=PositionalEncoding(d_model, tgt_seq_len, dropout)
    #create the encoder blocks
    #encoder_blocks = [EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)]
    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = Decoderblock(self_attention, cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    projection=Projectionlayer(d_model, tgt_vocab_size)
    transformer=Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  
    return transformer