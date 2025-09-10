import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from util import get_position_encoding, long_tensor_type


class N2N(torch.nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hops, story_size):
        super(N2N, self).__init__()

        self.embed_size = embed_size
        self.batch_size = batch_size
        self.story_size = story_size
        self.hops = hops

        if self.hops <= 0:
            raise ValueError("Number of hops have to be greater than 0")

        if self.hops > 3:
            raise ValueError("Number of hops should be less than 4")

        # story and query embedding
        self.A1 = nn.Embedding(vocab_size, embed_size)
        self.A1.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        # temporal encoding
        self.TA = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))

        # for 1 hop:
        # for >1 hop:
        self.A2 = nn.Embedding(vocab_size, embed_size)
        self.A2.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
        self.TA2 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))

        if self.hops >= 2:
            self.A3 = nn.Embedding(vocab_size, embed_size)
            self.A3.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
            self.TA3 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))

        if self.hops >= 3:
            self.A4 = nn.Embedding(vocab_size, embed_size)
            self.A4.weight = nn.Parameter(torch.randn(vocab_size, embed_size).normal_(0, 0.1))
            self.TA4 = nn.Parameter(torch.randn(self.batch_size, self.story_size, self.embed_size).normal_(0, 0.1))

        # final weight matrix
        self.W = nn.Parameter(torch.randn(embed_size, vocab_size), requires_grad=True)

    def forward(self, trainS, trainQ):
        S = Variable(trainS, requires_grad=False)
        Q = Variable(torch.squeeze(trainQ, 1), requires_grad=False)

        queries_emb = self.A1(Q)
        position_encoding = get_position_encoding(queries_emb.size(0), queries_emb.size(1), self.embed_size)
        queries = queries_emb * position_encoding
        queries_sum = torch.sum(queries, dim=1)

        # w_u = queries_sum
        # for i in range(self.hops):
        #     w_u = self.one_hop(S, w_u, self.A[i], self.A[i + 1], self.TA[i], self.TA[i + 1])

        w_u = self.hop(S, queries_sum, self.A1, self.A2, self.TA, self.TA2)

        if self.hops >= 2:
            w_u = self.hop(S, w_u, self.A2, self.A3, self.TA, self.TA3)

        if self.hops >= 3:
            w_u = self.hop(S, w_u, self.A3, self.A4, self.TA, self.TA4)

        wx = torch.mm(w_u, self.W)

        # Final softmax layer
        y_pred = F.softmax(wx)

        return y_pred

    def hop(self, trainS, u_k_1, A_k, C_k, temp_A_k, temp_C_k):
        mem_emb_A = self.embed_story(trainS, A_k)
        mem_emb_C = self.embed_story(trainS, C_k)

        mem_emb_A_temp = mem_emb_A + temp_A_k
        mem_emb_C_temp = mem_emb_C + temp_C_k

        u_k_1_list = [u_k_1] * self.story_size

        queries_temp = torch.squeeze(torch.stack(u_k_1_list, dim=1), 2)
        probabs = mem_emb_A_temp * queries_temp

        probabs = F.softmax(torch.squeeze(torch.sum(probabs, dim=2)))

        mem_emb_C_temp = mem_emb_C_temp.permute(0, 2, 1)
        probabs_temp = probabs.unsqueeze(1).expand_as(mem_emb_C_temp)

        pre_w = torch.mul(mem_emb_C_temp, probabs_temp)

        o = torch.sum(pre_w, dim=2)

        u_k = torch.squeeze(o) + torch.squeeze(u_k_1)
        return u_k

    def embed_story(self, story_batch, embedding_layer):
        story_embedding_list = []
        position_encoding = get_position_encoding(story_batch.size()[1], story_batch.size()[2], self.embed_size)

        for story in story_batch.split(1):
            story_variable = Variable(torch.squeeze(story, 0).data.type(long_tensor_type))
            story_embedding = embedding_layer(story_variable)
            story_embedding = story_embedding * position_encoding
            story_embedding_list.append(story_embedding)

        batch_story_embedding_temp = torch.stack(story_embedding_list)
        batch_story_embedding = torch.sum(batch_story_embedding_temp, dim=2)
        return torch.squeeze(batch_story_embedding, dim=2)
