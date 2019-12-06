import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False):
        super(GCN_DECONF, self).__init__()


        # self.gc2 = GraphConvolution(nhid, nclass)

        if cuda:
            self.gc = [GraphConvolution(nfeat, nhid).cuda()]
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = [GraphConvolution(nfeat, nhid)]
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, x, adj, t, cuda=False):

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        # print(t.shape,y1.shape,y0.shape)
        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return y, rep, p1
