from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from include import *
from models.models_b_rnn import *


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
        # loss = 0.0
        loss = torch.tensor(0.0, requires_grad=False).to(params['device'])

        for ii in range(params["numSubd"] + 1):
            nV = outputs[ii].size(0)
            loss += lossFunc(outputs[ii], Vt[:nV, :])

        if loss.isnan().item():
            print('[Filter: Train] mIdx: %d, loss: nan, will pass this object' % mIdx)
            train_pass_id.append(mIdx)
        else:
            print('[Filter: Train] mIdx: %d, loss: %f' % (mIdx, loss.cpu().detach().numpy()))
        # 训练结束，将当前mesh从gpu转到cpu，但这样可能会导致训练减速
        if params['memory_compact']:
            S.toDeviceId(mIdx, 'cpu')

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
        # loss = 0.0
        loss = torch.tensor(0.0, requires_grad=False).to(params['device'])

        for ii in range(params["numSubd"] + 1):
            nV = outputs[ii].size(0)
            loss += lossFunc(outputs[ii], Vt[:nV, :])

        # record validation error
        if loss.isnan().item():
            print('[Filter: Valid] mIdx: %d, loss: nan, will pass this object' % mIdx)
            valid_pass_id.append(mIdx)
        else:
            print('[Filter: Valid] mIdx: %d, loss: %f' % (mIdx, loss.cpu().detach().numpy()))
        # 训练结束，将当前mesh从gpu转到cpu，但这样可能会导致训练减速
        if params['memory_compact']:
            T.toDeviceId(mIdx, 'cpu')

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

    # updata params in the training process
    params['output_path'] = os.path.join(args.job, args.output_dir)
    os.path.exists(params['output_path']) or os.makedirs(params['output_path'])
    # 优先使用json文件的参数，除非命令行参数不为None
    if args.learning_rate is not None:
        params['lr'] = args.learning_rate
    if args.num_subd is not None:
        params['numSubd'] = args.num_subd
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.dout is not None:
        params['Dout'] = args.dout
    params['memory_compact'] = args.memory_compact
    params['pretrain_net'] = args.pretrain_net
    print(params)
    # make dir and cp hyperparameters.json
    with open(os.path.join(params['output_path'], 'hyperparameters.json'), 'w') as f:
        print('Write current training params to: ', os.path.join(params['output_path'], 'hyperparameters.json'))
        json.dump(params, f)

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

    if params['pretrain_net'] is not None:
        pretrain_net_path = os.path.join(params['output_path'], params['pretrain_net'])
        if os.path.exists(pretrain_net_path):
            net.load_state_dict(torch.load(pretrain_net_path, map_location=torch.device(params["device"])))
            pretrain_epoch = int(pretrain_net_path.split('_e')[-1].split('.')[0])
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
        return 0.5 ** (epoch // 500)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 定义SummaryWriter对象
    log_dir = os.path.join(params['output_path'], 'log')
    os.path.exists(log_dir) or os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # training
    bestLoss = np.inf

    # filter meshes before training and validation
    train_pass_id, valid_pass_id = filter_mesh_before_train(S, T, net, params, lossFunc)
    torch.cuda.empty_cache()
    S.removeMesh(train_pass_id)
    T.removeMesh(valid_pass_id)
    print('Filter ending, remove train_pass_id(%d) and valid_pass_id(%d) from S and T respectively'
          % (len(train_pass_id), len(valid_pass_id)))
    print('Real Train Object num, S.nM: ', S.nM)
    print('Real Valid Object num, T.nM: ', T.nM)

    train_loss_txt = 'train_loss_b%d_t%d_v%d_e%d.txt' % (params['batch_size'], S.nM, T.nM, total_epoch)
    valid_loss_txt = 'valid_loss_b%d_t%d_v%d_e%d.txt' % (params['batch_size'], S.nM, T.nM, total_epoch)
    f_train_loss = open(os.path.join(params['output_path'], train_loss_txt), 'a')
    f_valid_loss = open(os.path.join(params['output_path'], valid_loss_txt), 'a')

    # batch_size = params['batch_size']
    batch_size = params['batch_size']
    train_batch_num = int(np.ceil(S.nM / batch_size))
    train_mIdx_list = np.arange(S.nM)
    np.random.shuffle(train_mIdx_list)
    train_batch_mIdx_lists = np.array_split(train_mIdx_list, train_batch_num)

    valid_batch_num = int(np.ceil(T.nM / batch_size))
    valid_mIdx_list = np.arange(T.nM)
    np.random.shuffle(valid_mIdx_list)
    valid_batch_mIdx_lists = np.array_split(valid_mIdx_list, valid_batch_num)

    for epoch in range(params['epochs']):
        trainErr = 0.0
        ts = time.time()

        for batch_idx in range(train_batch_num):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # batch_loss = 0.0
            batch_loss = torch.tensor(0.0, requires_grad=True).to(params['device'])
            for mIdx in train_batch_mIdx_lists[batch_idx]:
                # forward pass
                S.toDeviceId(mIdx, params["device"])
                x = S.getInputData(mIdx)
                outputs = net(x, S.hfList[mIdx], S.poolMats[mIdx], S.dofs[mIdx])
                # target mesh
                Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])

                # compute loss function
                loss = 0.0
                # loss = torch.tensor(0.0, requires_grad=True).to(params['device'])
                for ii in range(params["numSubd"] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])
                batch_loss += loss
                # 训练结束，将当前mesh从gpu转到cpu，但这样可能会导致训练减速
                if params['memory_compact']:
                    S.toDeviceId(mIdx, 'cpu')
            batch_loss /= len(train_batch_mIdx_lists[batch_idx])
            batch_loss.backward()
            optimizer.step()
            trainErr += batch_loss.cpu().detach().numpy()

        train_loss_m = trainErr / train_batch_num
        scheduler.step()

        # validation
        validErr = 0.0
        for batch_idx in range(valid_batch_num):
            torch.cuda.empty_cache()

            batch_loss = torch.tensor(0.0, requires_grad=True).to(params['device'])
            for mIdx in valid_batch_mIdx_lists[batch_idx]:
                # forward pass
                T.toDeviceId(mIdx, params["device"])
                x = T.getInputData(mIdx)
                outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
                # target mesh
                Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])

                # compute loss function
                loss = 0.0
                # loss = torch.tensor(0.0, requires_grad=True).to(params['device'])
                torch.cuda.empty_cache()
                for ii in range(params["numSubd"] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])
                batch_loss += loss
                # 训练结束，将当前mesh从gpu转到cpu，但这样可能会导致训练减速
                if params['memory_compact']:
                    T.toDeviceId(mIdx, 'cpu')
            batch_loss /= len(valid_batch_mIdx_lists[batch_idx])
            validErr += batch_loss.cpu().detach().numpy()

        valid_loss_m = validErr / valid_batch_num
        if trainErr < bestLoss:
            bestLoss = trainErr
            # 每500epoch共享一个netparams name
            epoch_i = (pretrain_epoch + epoch) // 500 + 1
            NETPARAMS = 'netparams_b%d_t%d_v%d_e%d.dat' % (params['batch_size'], S.nM, T.nM, epoch_i * 500)
            torch.save(net.state_dict(), os.path.join(params['output_path'], NETPARAMS))

        print("epoch %d, train loss %.6e, valid loss %.6e, cost time: %.2f s" % (
            pretrain_epoch + epoch, train_loss_m, valid_loss_m, (time.time() - ts)))

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
    # mIdx = 0
    # x = T.getInputData(mIdx)
    # outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
    #
    # # write unrotated outputs
    # tgp.writeOBJ(os.path.join(params['output_path'], str(mIdx) + '_oracle.obj'),
    #              T.meshes[mIdx][len(outputs) - 1].V.to('cpu'),
    #              T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
    # for ii in range(len(outputs)):
    #     x = outputs[ii].cpu()
    #     obj_path = os.path.join(params['output_path'], str(mIdx) + '_subd' + str(ii) + '.obj')
    #     tgp.writeOBJ(obj_path, x, T.meshes[mIdx][ii].F.to('cpu'))
    #
    # # write rotated output shapes (validation set)
    # mIdx = 0
    # x = T.getInputData(mIdx)
    #
    # dV = torch.rand(1, 3).to(params['device'])
    # R = random3DRotation().to(params['device'])
    # x[:, :3] = x[:, :3].mm(R.t())
    # x[:, 3:] = x[:, 3:].mm(R.t())
    # x[:, :3] += dV
    # outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
    #
    # # write rotated outputs
    # for ii in range(len(outputs)):
    #     x = outputs[ii].cpu()
    #     obj_path = os.path.join(params['output_path'], str(mIdx) + '_rot_subd' + str(ii) + '.obj')
    #     tgp.writeOBJ(obj_path, x, T.meshes[mIdx][ii].F.to('cpu'))
    out_test_dir = os.path.join(params['output_path'], 'test_e%d' % total_epoch)
    test_and_evaluate(out_test_dir, net, T, params['numSubd'])


def test_and_evaluate(out_test_dir, net, T, numSubd):
    os.path.exists(out_test_dir) or os.makedirs(out_test_dir)

    def call_mm_compare(refer, test, mode='pcc'):
        import subprocess
        mm = '/work/Users/zhuzhiwei/vmesh/externaltools/mpeg-pcc-mmetric/build/Release/bin/mm'
        cmd = mm + ' compare --inputModelA ' + refer + ' --inputModelB ' + test + ' --mode ' + mode
        print(cmd)
        output = subprocess.check_output(cmd, shell=True, encoding='utf-8')
        return match_mm_log(output)

    def match_mm_log(log):
        import re
        # mseF, PSNR(p2point) Mean=64.8143145, 用正则表达式从该语句匹配数字
        pattern_point = re.compile(r'PSNR\(p2point\) Mean=(\d+.\d+)')
        p2point = pattern_point.findall(log)
        print('p2point: ', p2point[0])
        # mseF, PSNR(p2plane) Mean=64.8143145, 用正则表达式从该语句匹配数字
        pattern_plane = re.compile(r'PSNR\(p2plane\) Mean=(\d+.\d+)')
        p2plane = pattern_plane.findall(log)
        print('p2plane: ', p2plane[0])
        return p2point[0], p2plane[0]

    metric_file_path = os.path.join(out_test_dir, 'metric.txt')
    metric_file = open(metric_file_path, 'w')
    metric_file.write('mIdx\tmse_loss\tp2point\tp2plane\n')
    mse_loss_list = []
    p2point_list = []
    p2plane_list = []

    os.path.exists(out_test_dir) or os.makedirs(out_test_dir)
    max_num = min(10, T.nM)
    for mIdx in range(0, max_num):
        metric_file.write(str(mIdx) + '\t')
        gt_path = os.path.join(out_test_dir, str(mIdx).zfill(len(str(T.nM))) + '_subd' + str(numSubd) + '_gt.obj')
        pred_path = os.path.join(out_test_dir, str(mIdx).zfill(len(str(T.nM))) + '_subd' + str(numSubd) + '.obj')
        x = T.getInputData(mIdx)
        outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])

        # write unrotated outputs
        gt_x = T.meshes[mIdx][len(outputs) - 1].V.to('cpu')
        tgp.writeOBJ(gt_path, gt_x, T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
        ii = len(outputs) - 1
        pred_x = outputs[ii].cpu()
        tgp.writeOBJ(pred_path, pred_x, T.meshes[mIdx][ii].F.to('cpu'))
        mse_loss = np.mean(np.square(np.linalg.norm(gt_x.detach().numpy() - pred_x.detach().numpy(), axis=1)))
        p2point, p2plane = call_mm_compare(gt_path, pred_path)

        mse_loss_list.append(mse_loss)
        p2point_list.append(float(p2point))
        p2plane_list.append(float(p2plane))

        metric_file.write(str(mse_loss) + '\t' + str(p2point) + '\t' + str(p2plane) + '\n')

    average_mse_loss = np.mean(mse_loss_list)
    average_p2point = np.mean(p2point_list)
    average_p2plane = np.mean(p2plane_list)
    print('average_mse_loss: ', average_mse_loss)
    print('average_p2point: ', average_p2point)
    print('average_p2plane: ', average_p2plane)
    metric_file.write('average\t' + str(average_mse_loss) + '\t' + str(average_p2point) + '\t'
                      + str(average_p2plane) + '\n')
    metric_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='7', help='GPU list to use, e.g. 0,1,2,3')
    parser.add_argument('-j', '--job', type=str, default='../jobs/Lion_f500_ns3_nm10/',
                        help='Path to the job directory')
    parser.add_argument('-p', '--pretrain_net', type=str, default=None, help='pretrained net parameters file name')
    parser.add_argument('-l', '--learning_rate', type=float, default=None, help='learning rate')
    parser.add_argument('-o', '--output_dir', type=str, default='test', help='output directory name')
    parser.add_argument('-s', '--num_subd', type=int, default=None, help='num of subdivision in training')
    parser.add_argument('-do', '--dout', type=int, default=None, help='num dimension of Dout in training net')
    parser.add_argument('-e', '--epochs', type=int, default=None, help='epochs in this training')
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='batch size in training')
    parser.add_argument('-m', '--memory_compact', type=bool, default=False,
                        help='Input anything will activate memory compact mode, but speed will be slower. '
                             'Do not input, default is False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
