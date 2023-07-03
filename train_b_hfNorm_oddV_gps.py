from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from include import *
from models.models_b_hfNorm_oddV import *

def filter_mesh_before_train(S, T, net, params, lossFunc):
    """
    Filter out meshes that are too large for the GPU memory
    :param S: training set
    :param T: validation set
    :param net: network
    :param params: hyper parameters
    :param lossFunc: loss function
    :return: train_pass_id, valid_pass_id
    """
    train_pass_id = []
    valid_pass_id = []
    for mIdx in range(S.nM):
        torch.cuda.empty_cache()
        # forward pass
        S.toDeviceId(mIdx, params['device'])
        x = S.getInputData(mIdx)
        try:
            outputs = net(x, S.hfList[mIdx], S.poolMats[mIdx], S.dofs[mIdx])
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('[Filter: Train] mIdx: %d, WARNING: out of memory, will pass this object' % mIdx)
                torch.cuda.empty_cache()
                train_pass_id.append(mIdx)
                continue
            else:
                raise exception

        # target mesh
        Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])

        # compute loss function
        loss = 0.0

        for ii in range(params["numSubd"] + 1):
            nV = outputs[ii].size(0)
            loss += lossFunc(outputs[ii], Vt[:nV, :])

        if loss.isnan().item():
            print('[Filter: Train] mIdx: %d, loss: nan, will pass this object' % mIdx)
            train_pass_id.append(mIdx)
        else:
            print('[Filter: Train] mIdx: %d, loss: %f' % (mIdx, loss.cpu()))

    # loop over validation shapes
    for mIdx in range(T.nM):
        torch.cuda.empty_cache()
        T.toDeviceId(mIdx, params['device'])
        x = T.getInputData(mIdx)
        
        try:
            outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('[Filter: Valid] mIdx: %d, WARNING: out of memory, will pass this object' % mIdx)
                torch.cuda.empty_cache()
                valid_pass_id.append(mIdx)
                continue
            else:
                raise exception

        # target mesh
        Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])

        # compute loss function
        loss = 0.0

        for ii in range(params["numSubd"] + 1):
            nV = outputs[ii].size(0)
            loss += lossFunc(outputs[ii], Vt[:nV, :])

        # record validation error
        if loss.isnan().item():
            print('[Filter: Valid] mIdx: %d, loss: nan, will pass this object' % mIdx)
            valid_pass_id.append(mIdx)
        else:
            print('[Filter: Valid] mIdx: %d, loss: %f' % (mIdx, loss.cpu()))

    return train_pass_id, valid_pass_id
def main(args):
    # load hyper parameters
    if os.path.exists(os.path.join(args.job, args.output_dir, 'hyperparameters.json')):
        print('Loading hyperparameters from ' + os.path.join(args.job, args.output_dir, 'hyperparameters.json'))
        with open(os.path.join(args.job, args.output_dir, 'hyperparameters.json'), 'r') as f:
            params = json.load(f)
    else:
        print('Loading hyperparameters from ' + os.path.join(args.job, 'hyperparameters.json'))
        with open(os.path.join(args.job, 'hyperparameters.json'), 'r') as f:
            params = json.load(f)
        # make dir and cp hyperparameters.json
        output_path = os.path.join(args.job, args.output_dir)
        os.path.exists(output_path) or os.makedirs(output_path)
        cmd = 'cp ' + os.path.join(args.job, 'hyperparameters.json') + ' ' + output_path
        os.system(cmd)
        print(cmd)
        
    # updata params in the training process
    params['output_path'] = os.path.join(args.job, args.output_dir)
    os.path.exists(params['output_path']) or os.makedirs(params['output_path'])
    params['lr'] = args.learning_rate
    if args.num_subd is not None:
        params['numSubd'] = args.num_subd
    pretrain_net = args.pretrain_net
    print(params)

    # load traininig data
    S = pickle.load(open(params["train_pkl"], "rb"))
    S.computeParameters()
    # S.toDevice(params["device"])

    # load validation set
    T = pickle.load(open(params["valid_pkl"], "rb"))
    T.computeParameters()
    # T.toDevice(params["device"])

    print('Input Train Object num, S.nM: ', S.nM)
    print('Input Valid Object num, T.nM: ', T.nM)
    print('Input numSubd: ', params['numSubd'])

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

        else:
            print('Error: pretrain_net_path: %s is not exist!' % pretrain_net_path)
    else:
        net.apply(init_weights)

    total_epoch = pretrain_epoch + params['epochs']

    # loss function
    lossFunc = torch.nn.MSELoss().to(params['device'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    # learning rate scheduler
    def lr_lambda(epoch):
        return 1 ** (epoch // 1000)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 定义SummaryWriter对象
    log_dir = os.path.join(params['output_path'], 'log')
    os.path.exists(log_dir) or os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # training
    bestLoss = np.inf

    # filter meshes before training and validation
    train_pass_id, valid_pass_id = filter_mesh_before_train(S, T, net, params, lossFunc)
    S.removeMesh(train_pass_id)
    T.removeMesh(valid_pass_id)
    print('Filter ending, remove train_pass_id and valid_pass_id from S and T respectively')
    print('Real Train Object num, S.nM: ', S.nM)
    print('Real Valid Object num, T.nM: ', T.nM)
    NETPARAMS = 'netparams_e%d_t%d_v%d.dat' % (total_epoch, S.nM, T.nM)
    train_loss_txt = 'train_loss_e%d_t%d_v%d.txt' % (total_epoch, S.nM, T.nM)
    valid_loss_txt = 'valid_loss_e%d_t%d_v%d.txt' % (total_epoch, S.nM, T.nM)
    f_train_loss = open(os.path.join(params['output_path'], train_loss_txt), 'a')
    f_valid_loss = open(os.path.join(params['output_path'], valid_loss_txt), 'a')

    # batch_size = params['batch_size']
    batch_size = 8
    trian_batch_num = int(np.ceil(S.nM / batch_size))
    train_mIdx_list = np.arange(S.nM)
    np.random.shuffle(train_mIdx_list)
    trian_batch_mIdx_lists = np.array_split(train_mIdx_list, trian_batch_num)

    valid_batch_num = int(np.ceil(T.nM / batch_size))
    valid_mIdx_list = np.arange(T.nM)
    np.random.shuffle(valid_mIdx_list)
    valid_batch_mIdx_lists = np.array_split(valid_mIdx_list, valid_batch_num)


    for epoch in range(params['epochs']):
        trainErr = 0.0
        ts = time.time()

        for batch_idx in range(trian_batch_num):
            optimizer.zero_grad()
            batch_loss = 0.0
            for mIdx in trian_batch_mIdx_lists[batch_idx]:
                # forward pass
                S.toDeviceId(mIdx, params["device"])
                x = S.getInputData(mIdx)
                outputs = net(x, S.hfList[mIdx], S.poolMats[mIdx], S.dofs[mIdx])
                # target mesh
                Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])

                # compute loss function
                loss = 0.0
                torch.cuda.empty_cache()
                for ii in range(params["numSubd"] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])
                batch_loss += loss
            batch_loss /= len(trian_batch_mIdx_lists[batch_idx])
            batch_loss.backward()
            optimizer.step()
            trainErr += batch_loss.cpu()

        train_loss_m = trainErr / trian_batch_num
        scheduler.step()


        # validation
        validErr = 0.0
        for batch_idx in range(valid_batch_num):
            batch_loss = 0.0
            for mIdx in valid_batch_mIdx_lists[batch_idx]:
                # forward pass
                T.toDeviceId(mIdx, params["device"])
                x = T.getInputData(mIdx)
                outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
                # target mesh
                Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])

                # compute loss function
                loss = 0.0
                torch.cuda.empty_cache()
                for ii in range(params["numSubd"] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])
                batch_loss += loss
            batch_loss /= len(valid_batch_mIdx_lists[batch_idx])
            validErr += batch_loss.cpu()

        valid_loss_m = validErr / valid_batch_num
        if trainErr < bestLoss:
            bestLoss = trainErr
            torch.save(net.state_dict(), os.path.join(params['output_path'], NETPARAMS))

        print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %s s" % (
        pretrain_epoch + epoch, train_loss_m, valid_loss_m,
        int(round((params['epochs'] - epoch) * (time.time() - ts)))))

        # save loss history
        writer.add_scalar('train_loss', train_loss_m, pretrain_epoch + epoch)
        f_train_loss.write(str(train_loss_m) + '\n')

        writer.add_scalar('valid_loss', valid_loss_m, pretrain_epoch + epoch)
        f_valid_loss.write(str(valid_loss_m) + '\n')

    # 关闭SummaryWriter,file对象
    writer.close()
    f_train_loss.close()
    f_valid_loss.close()
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
    parser.add_argument('-j', '--job', type=str, default='../jobs/Lion_f500_ns3_nm10/', help='Path to the job directory')
    parser.add_argument('-p', '--pretrain_net', type=str, default=None, help='pretrained net parameters file name')
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('-o', '--output_dir', type=str, default='test', help='output directory name')
    parser.add_argument('-s', '--num_subd', type=int, default=None, help='num of subdivision in training')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)