import os

import os.path as osp
import utils
from utils import AverageMeter
import mydataset
import argparse
import time
from model import get_model,EarlyStopping
import torch
import numpy as np
import myloss
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
from sklearn.cluster import KMeans
from evaluation import clustering_metric
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def evaluate(H, all_label, estimator):
    end = time.time()
    preds = estimator.fit_predict(H)
    all_label = all_label.reshape(-1)
    results = clustering_metric(all_label, preds)
    return results,preds

def train_1(args,data, label,inc_V_ind,model, loss_model ,opt, sche, estimator):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    inc_V_ind = torch.from_numpy(inc_V_ind).to('cuda:0')

    model.train()
    end = time.time()
    data_time.update(time.time() - end)

    q, x_bar, x_bar_1, Q, z = model(copy.deepcopy(data), mask=inc_V_ind)
    result,preds = evaluate(z.cpu().detach().numpy(), label, estimator)
    mse_loss = loss_model.weighted_wmse_loss(x_bar, data, inc_V_ind)
    mse_loss_1 = loss_model.weighted_wmse_loss(x_bar_1, data, None)
    L_ccl = loss_model.L_lc(Q,1,Q.shape[1])
    l_ddc = loss_model.l_ddc(q,z)
    loss = mse_loss * 1 + args.beta * mse_loss_1 + args.gamma * (L_ccl  + l_ddc )

    opt.zero_grad()
    loss.backward()


    if isinstance(sche, CosineAnnealingWarmRestarts):
        sche.step()
    opt.step()
    losses.update(loss.item())
    batch_time.update(time.time() - end)

    return loss, model, result,preds,z

def pre_train(data,inc_V_ind, model, loss_model,opt, sche):
    inc_V_ind = torch.from_numpy(inc_V_ind).to('cuda:0')
    model.train()
    _, x_bar, x_bar_1, _ , _ = model(copy.deepcopy(data), mask=inc_V_ind)
    mse_loss = loss_model.weighted_wmse_loss(x_bar, data, inc_V_ind)

    loss = mse_loss * 1
    opt.zero_grad()
    loss.backward()

    if isinstance(sche, CosineAnnealingWarmRestarts):
        sche.step()
    opt.step()

    return loss


def main(args):
    print(args.mask_view_ratio)
    data_path = osp.join(args.data_dir, args.dataset + '.mat')
    fold_data_path = osp.join(args.fold_dir, args.dataset + '_percentDel_' + str(args.mask_view_ratio) + '.mat')
    folds_num = args.folds_num
    logger = utils.setLogger(None)

    ACC_a = AverageMeter()
    NMI_a = AverageMeter()
    ARI_a = AverageMeter()

    for fold_idx in range(folds_num):
        train_dataset = mydataset.getIncDataloader(data_path, fold_data_path, fold_idx=fold_idx, num_workers=4)

        data = [torch.tensor(v_data).to(args.device) for v_data in train_dataset.mv_data]
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        model = get_model(d_list, d_model=args.dim, n_layers=1, heads=4, classes_num=train_dataset.classes_num,
                          dropout=0.,view = train_dataset.view_num,device=args.device)

        loss_model = myloss.MyLoss(classes_num)
        optimizer = Adam(model.parameters(), lr=args.lr)

        flag = 0        # the flag of early stopping
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        early_1 = EarlyStopping(args,patience=7)

        estimator = KMeans(n_clusters=classes_num, max_iter=300, n_init=10, random_state=928)
        # record the best result
        best_acc2 = 0
        best_nmi2 = 0
        best_ari2 = 0
        best_epoch = 0

        logger.info('train_data_num:' + str(len(train_dataset)) + '   fold_idx:' + str(fold_idx))
        # pre-train
        for epoch in range(args.pre_epochs):
            train_losses = pre_train(data, train_dataset.inc_V_ind,model,
                    loss_model, optimizer, scheduler)
            early_1(train_losses, model,[])
            if early_1.early_stop:
                flag = 1
                print("Early stopping")
                break
        logger.info("pre-train have finished!")
        shrink = {'acc':[],'loss':[],'nmi':[]}
        #    train
        for epoch in range(args.epochs):
            if flag == 1 and epoch == 0:
                state_dict = torch.load(os.path.join(args.save_path,'model.pth'))
                model.load_state_dict(state_dict)

            loss, model, result,preds,z = train_1(args,data, train_dataset.labels, train_dataset.inc_V_ind, model,
                                                            loss_model, optimizer, scheduler, estimator)

            shrink['loss'].append(loss.item())
            shrink['nmi'].append(result['NMI'])
            shrink['acc'].append(result['ACC'])
            if result['ACC'] > best_acc2:
                best_acc2 = np.copy(result['ACC'])
                best_nmi2 = np.copy(result['NMI'])
                best_ari2 = np.copy(result['ARI'])
                best_epoch = epoch
                best_pred = preds
                # torch.save(model, os.path.join(args.save_path,'model.pth'))

            print('best-acc: epoch {}:{}     epoch {}: ACC:{}  NMI:{}  ARI:{} loss: {}'.format(best_epoch, best_acc2,
                                                                                               epoch, result['ACC'],
                                                                                               result['NMI'],
                                                                                               result['ARI'],
                                                                                               loss))
        ####  各轮结果
        ACC_a.update(best_acc2)
        NMI_a.update(best_nmi2)
        ARI_a.update(best_ari2)

    print('---------------------------Result of {}----------------------------'.format(args.dataset))
    print('------------------------- MissRate = {:.2f}--------------------'.format(args.mask_view_ratio))
    print('ACC = {:.4f} ± {:.4f}\t NMI = {:.4f} ± {:.4f}\t ARI = {:.4f} ± {:.4f}\t'.format(ACC_a.avg,ACC_a.std,NMI_a.avg,NMI_a.std,ARI_a.avg,ARI_a.std))
    print('------------------------Training over------------------------')
    return 'ACC = {:.4f} ± {:.4f}\t NMI = {:.4f} ± {:.4f}\t ARI = {:.4f} ± {:.4f}\t'.format(ACC_a.avg,ACC_a.std,NMI_a.avg,NMI_a.std,ARI_a.avg,ARI_a.std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))

    #   定义文件位置
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='data/')
    parser.add_argument('--fold-dir', type=str, metavar='PATH',
                        default='data/')

    #   设定数据集
    parser.add_argument('--dataset', type=str,
                        default='UCI')
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--folds-num', default=1, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'output'))
    parser.add_argument('--img-dir', type=str, metavar='PATH',
                        default='hw-imgs/0.5_')
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--save-img', default=False, type=bool)
    parser.add_argument('--save_path', default='./weight/', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='final_')

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--pre_epochs', type=int, default=30)


    # Training args
    parser.add_argument('--dim', type=int, default=600)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-3)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.lr = 1e-4
    args.beta = 10
    args.gamma = 0.1
    a = main(args)



