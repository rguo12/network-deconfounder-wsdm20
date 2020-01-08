# from __future__ import division
# from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from pygcn.utils import load_data, accuracy
# from pygcn.models import GCN
from models.netdeconf import GCN_DECONF
import utils

from scipy import sparse as sp
import csv

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0,
                    help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--extrastr', type=str, default='1')

# parser.add_argument('--exp', type=int, default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4,
                    help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100.,
                    help='gradient clipping')
parser.add_argument('--nout', type=int, default=2)
parser.add_argument('--nin', type=int, default=2)

parser.add_argument('--tr', type=float, default=0.6)
parser.add_argument(
    '--path', type=str, default='./datasets/')
parser.add_argument('--normy', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
alpha = torch.FloatTensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

if args.cuda:
    loss = loss.cuda()
    bce_loss = bce_loss.cuda()

def prepare(i_exp):
    X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(i_exp), extra_str=args.extrastr)

    n = X.shape[0]
    n_train = int(n * args.tr)
    n_test = int(n * 0.2)
    n_valid = n_test

    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]

    # X = sp.csr_matrix(X)
    X = utils.normalize(X) #row-normalize
#    A = utils.normalize(A+sp.eye(n))

    X = X.todense()
    X = torch.FloatTensor(X)

    Y1 = torch.FloatTensor(np.squeeze(Y1))
    Y0 = torch.FloatTensor(np.squeeze(Y0))
    T = torch.LongTensor(np.squeeze(T))

    A = utils.sparse_mx_to_torch_sparse_tensor(A)

    # print(X.shape, Y1.shape, A.shape)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Model and optimizer
    model = GCN_DECONF(nfeat=X.shape[1],
                nhid=args.hidden,
                dropout=args.dropout,n_out=args.nout,n_in=args.nin,cuda=args.cuda)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        # model = model.cuda()
        X = X.cuda()
        A = A.cuda()
        T = T.cuda()
        Y1 = Y1.cuda()
        Y0 = Y0.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return X, A, T, Y1, Y0, idx_train, idx_val, idx_test,model, optimizer


def train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer):
    t = time.time()
    model.train()
#    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()
    yf_pred, rep, p1 = model(X, A, T)
    ycf_pred, _, p1 = model(X, A, 1-T)

    #balancing
    rep_t1, rep_t0 = rep[idx_train][(T[idx_train] > 0).nonzero()], rep[idx_train][(T[idx_train] < 1).nonzero()]
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)

    # print(yf_pred.shape, idx_train.shape)
    # yf_pred_tr = yf_pred[idx_train]
    YF = torch.where(T>0,Y1,Y0)
    YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]
    
    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist

    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)
    if epoch%10==0:
        # loss_val_f = loss(yf_pred[idx_val], torch.where(T>0,Y1,Y0)[idx_val])
        # loss_val_cf = loss(ycf_pred[idx_val], torch.where(T>0,Y0,Y1)[idx_val])
        y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)
        # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
        if args.normy:
            y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym
        pehe_val = torch.sqrt(loss((y1_pred - y0_pred)[idx_val],(Y1 - Y0)[idx_val]))

        mae_ate_val = torch.abs(
            torch.mean((y1_pred - y0_pred)[idx_val])-torch.mean((Y1 - Y0)[idx_val]))
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              # 'acc_train: {:.4f}'.format(acc_train.item()),
              'pehe_val: {:.4f}'.format(pehe_val.item()),
              'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp):
    model.eval()
    yf_pred, rep, p1 = model(X, A, T)
    # yf = torch.where(T>0, Y1, Y0)
    ycf_pred, _, _ = model(X, A, 1-T)

    YF = torch.where(T>0,Y1,Y0)
    YCF = torch.where(T>0,Y0,Y1)

    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
    # YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys

    y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)

    if args.normy:
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
    pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test],(Y1 - Y0)[idx_test]))
    mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test])-torch.mean((Y1 - Y0)[idx_test]))
    print("Test set results:",
          "pehe_ts= {:.4f}".format(pehe_ts.item()),
          "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))

    of_path = './new_results/' + args.dataset + args.extrastr + '/' + str(args.tr)
    
    if args.lr != 1e-2:
        of_path += 'lr'+str(args.lr)
    if args.hidden != 100:
        of_path += 'hid'+str(args.hidden)
    if args.dropout != 0.5:
        of_path += 'do'+str(args.dropout)
    if args.epochs != 50:
        of_path += 'ep'+str(args.epochs)
    if args.weight_decay != 1e-5:
        of_path += 'lbd'+str(args.weight_decay)
    if args.nout != 1:
        of_path += 'nout'+str(args.nout)
    if args.alpha != 1e-5:
        of_path += 'alp'+str(args.alpha)
    if args.normy == 1:
        of_path += 'normy'

    of_path += '.csv'

    of = open(of_path,'a')
    wrt = csv.writer(of)
    wrt.writerow([pehe_ts.item(),mae_ate_ts.item()])


if __name__ == '__main__':
    for i_exp in range(0,10):
        # Train model
        X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(i_exp)
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp)
