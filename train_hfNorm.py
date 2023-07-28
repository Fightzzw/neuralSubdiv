from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import sys
import argparse
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from include import *
from models.models_hfNorm import *


def main(args):
    # load hyper parameters
    folder = args.job
    if os.path.exists(os.path.join(args.job, args.output_dir,'hyperparameters.json')):
        print('Loading hyperparameters from ' + os.path.join(args.job, args.output_dir, 'hyperparameters.json'))
        with open(os.path.join(args.job, args.output_dir, 'hyperparameters.json'), 'r') as f:
            params = json.load(f)
    else:
        print('Loading hyperparameters from ' + folder + 'hyperparameters.json')
        with open(folder + 'hyperparameters.json', 'r') as f:
            params = json.load(f)
    # updata params in the training process
    params['output_path'] = os.path.join(args.job, args.output_dir)
    os.path.exists(params['output_path']) or os.makedirs(params['output_path'])
    params['lr'] = args.learning_rate
    pretrain_net = args.pretrain_net
    print(params)

    # load traininig data
    S = pickle.load(open(params["train_pkl"], "rb"))
    S.computeParameters()
    S.toDevice(params["device"])

    # load validation set
    T = pickle.load(open(params["valid_pkl"], "rb"))
    T.computeParameters()
    T.toDevice(params["device"])

    print('S.nM: ', S.nM)
    print('T.nM: ', T.nM)
    print('numSubd: ', params['numSubd'])

    # initialize network
    # detect if the path params['output_path'] + NETPARAMS exists

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    net = SubdNet(params)
    net = net.to(params['device'])
    pretrain_epoch = 0

    if pretrain_net is not None:
        pretrain_net_path = os.path.join(params['output_path'], pretrain_net)
        if os.path.exists(pretrain_net_path):
            net.load_state_dict(torch.load(pretrain_net_path, map_location=torch.device(params["device"])))
            pretrain_epoch = int(pretrain_net_path.split('_')[-1].split('.')[0])
            print('pretrain_net_path: ', pretrain_net_path)
            print('pretrain_epoch: ', pretrain_epoch)
            lr_ratio = 0.5 ** (pretrain_epoch // 200)
        else:
            print('Error: pretrain_net_path: %s is not exist!' % pretrain_net_path)
    else:
        net.apply(init_weights)

    total_epoch = pretrain_epoch + params['epochs']
    NETPARAMS = 'netparams_%d.dat' % total_epoch
    train_loss_txt = 'train_loss_%d.txt' % total_epoch
    valid_loss_txt = 'valid_loss_%d.txt' % total_epoch

    # loss function
    lossFunc = torch.nn.MSELoss().to(params['device'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    # learning rate scheduler
    def lr_lambda(epoch):
        return 0.5 ** (epoch // 1000)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 定义SummaryWriter对象
    log_dir = os.path.join(params['output_path'], 'log')
    os.path.exists(log_dir) or os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # training
    trainLossHis = []
    validLossHis = []
    bestLoss = np.inf
    nSub = params["numSubd"]
    for epoch in range(params['epochs']):
        ts = time.time()

        # loop over training shapes
        trainErr = 0.0
        for mIdx in range(S.nM):
            # forward pass
            x = S.getInputData(mIdx)
            outputs = net(x, mIdx, S.hfList, S.poolMats, S.dofs)

            # target mesh
            Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = S.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"] + 1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV, :])
                # loss += mse_loss(outputs[ii], Vt[:nV, :])

            # move
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training error
            trainErr += loss.cpu().data.numpy()

        train_loss_m = trainErr / S.nM
        trainLossHis.append(train_loss_m)
        writer.add_scalar('train_loss', train_loss_m, pretrain_epoch + epoch)

        scheduler.step()

        # loop over validation shapes
        validErr = 0.0
        for mIdx in range(T.nM):
            x = T.getInputData(mIdx)
            outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

            # target mesh
            Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = T.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"] + 1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV, :])

            # record validation error
            validErr += loss.cpu().data.numpy()
        valid_loss_m = validErr / T.nM
        validLossHis.append(valid_loss_m)
        writer.add_scalar('valid_loss', valid_loss_m, pretrain_epoch + epoch)

        # save the best model
        if trainErr < bestLoss:
            bestLoss = trainErr
            # 每500epoch共享一个netparams name
            epoch_i = (pretrain_epoch + epoch) // 500 + 1
            NETPARAMS = 'netparams_e%d_t%d_v%d.dat' % (epoch_i * 500, S.nM, T.nM)
            torch.save(net.state_dict(), os.path.join(params['output_path'], NETPARAMS))

        print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %s s" % (
        pretrain_epoch + epoch, trainLossHis[-1], validLossHis[-1],
        int(round((params['epochs'] - epoch) * (time.time() - ts)))))

    ## 关闭SummaryWriter对象
    writer.close()
    # save loss history
    np.savetxt(os.path.join(params['output_path'], train_loss_txt), np.array(trainLossHis), delimiter=',')
    np.savetxt(os.path.join(params['output_path'], valid_loss_txt), np.array(validLossHis), delimiter=',')

    # write output shapes (validation set)
    mIdx = 0
    x = T.getInputData(mIdx)
    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    # write unrotated outputs
    tgp.writeOBJ(os.path.join(params['output_path'], str(mIdx) + '_oracle.obj'),
                 T.meshes[mIdx][len(outputs) - 1].V.to('cpu'),
                 T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        obj_path = os.path.join(params['output_path'], str(mIdx) + '_subd' + str(ii) + '.obj')
        tgp.writeOBJ(obj_path, x, T.meshes[mIdx][ii].F.to('cpu'))

    # write rotated output shapes (validation set)
    mIdx = 0
    x = T.getInputData(mIdx)

    dV = torch.rand(1, 3).to(params['device'])
    R = random3DRotation().to(params['device'])
    x[:, :3] = x[:, :3].mm(R.t())
    x[:, 3:] = x[:, 3:].mm(R.t())
    x[:, :3] += dV
    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    # write rotated outputs
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        obj_path = os.path.join(params['output_path'], str(mIdx) + '_rot_subd' + str(ii) + '.obj')
        tgp.writeOBJ(obj_path, x, T.meshes[mIdx][ii].F.to('cpu'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='7', help='GPU list to use, e.g. 0,1,2,3')
    parser.add_argument('-j', '--job', type=str, default='jobs/Lion_f500_ns3_nm10/', help='Path to the job directory')
    parser.add_argument('-p', '--pretrain_net', type=str, default=None, help='pretrained net parameters file name')
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('-o', '--output_dir', type=str, default='test', help='output directory name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
