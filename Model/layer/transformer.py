import math

import torch
import torch.nn as nn

import torch.nn.functional as F
class SelfAttention(nn.Module) :
    def __init__(self,  args,input_size, hidden_size,num_attention_heads):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.fc1=nn.Linear(self.attention_head_size,1)
        self.fc2 = nn.Linear(self.num_attention_heads, 1)
        self.fc3 = nn.Linear(args.max_atom, hidden_size*num_attention_heads)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        # return x.permute(0, 2, 1, 3)
        return x.permute(0,1, 3, 2, 4)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        # attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1,2, 4, 3))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0,1, 3, 2, 4).contiguous()
        context=self.fc1(context).squeeze(dim=-1)
        context = self.fc2(context).squeeze(dim=-1)
        context = self.fc3(context)
        #context = context.permute(0, 2, 1, 3).contiguous()
        # new_size = context.size()[ : -2] + (self.all_head_size , )
        # context = context.view(*new_size).mean(dim=-1)
        return context

# class SelfAttentionModule(nn.Module):
#     def __init__(self,args, embed_dim, out_dim, num_heads):
#         super(SelfAttentionModule, self).__init__()
#
#         self.dim1 = embed_dim
#         self.dim2 = out_dim
#         self.num_heads = num_heads
#         self.args=args
#         self.attention = SelfAttention(embed_dim, num_heads)
#         self.linear = nn.Linear(embed_dim, out_dim)
#         self.leaky_relu=nn.LeakyReLU()
#
#     def forward(self, x):
#         # 调整输入张量形状为 (seq_length, batch_size, dim1)
#         x = x.permute(1, 0, 2)
#
#         # 应用自注意力运算
#         attention_output, _ = self.attention(x, x, x)
#
#         # 调整输出张量形状为 (batch_size, seq_length, dim1)
#         attention_output = attention_output.permute(1, 0, 2)
#
#         # 应用线性变换得到最终输出
#         output = self.linear(attention_output)
        return  output



