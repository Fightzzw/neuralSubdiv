from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import torch

from include import *
from models.models_b_rnn import *
NETPARAMS = 'netparams.dat'
import argparse
import random
# run the file as "python test.py ./path/to/folder/ ./path/to/testMesh.obj"
# WARNING
# - the current code only works for .obj file without boundaries :P
# - there are two failure cases mentioned in the paper:
#   1. testing triangles has aspect ratios that are not in the training data
#   2. testing triangles has triangle size that are not in the training data
#   (we include a failure case in the data: the ear of the horse.obj)

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

def cross_test():
    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    cross_list = ['76947_sf_f1000_ns3_nm10','70558_sf_f1000_ns3_nm10','203289_sf_f1000_ns3_nm10','1717684_sf_f1000_ns3_nm10']
    mesh_pkl_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_PKL/'
    job_list = ['b1_anchor', 'b1_hfNorm', 'b1_oddV_MfV', 'b1_rnn']
    job_list = ['b1_rnn_dout48', 'b1_rnn_ori']

    for folder in cross_list[2:3]:
        print('current job folder is: ', folder)
        for job in job_list:
            cur_work_dir = os.path.join(work_dir, folder, job)
            if os.path.exists(os.path.join(cur_work_dir, 'hyperparameters.json')):
                print('Loading hyperparameters from ' + os.path.join(cur_work_dir, 'hyperparameters.json'))
                with open(os.path.join(cur_work_dir, 'hyperparameters.json'), 'r') as f:
                    params = json.load(f)
            else:
                print('Error: File not exist! ' + cur_work_dir + '/hyperparameters.json')
                exit(-1)
            if folder == '70558_sf_f1000_ns3_nm10':
                net_path = os.path.join(cur_work_dir, 'netparams_b1_t9_v9_e3000.dat')
            else:
                net_path = os.path.join(cur_work_dir, 'netparams_b1_t10_v10_e3000.dat')

            net = SubdNet(params)
            net = net.to(params['device'])
            net.load_state_dict(torch.load(net_path, map_location=torch.device(params["device"])))
            net.eval()
            for test_folder in cross_list:
                print('\ttest folder: ', test_folder)
                if test_folder == folder:
                    continue
                test_pkl = os.path.join(mesh_pkl_dir, test_folder + '.pkl')
                T = pickle.load(open(test_pkl, "rb"))
                T.computeParameters()
                T.toDevice(params["device"])
                out_test_dir = os.path.join(cur_work_dir, 'test_e3000', test_folder)
                test_and_evaluate(out_test_dir, net, T, params['numSubd'])



def self_test():
    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    cross_list = ['76947_sf_f1000_ns3_nm10','70558_sf_f1000_ns3_nm10','203289_sf_f1000_ns3_nm10','1717684_sf_f1000_ns3_nm10']
    mesh_pkl_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_PKL/'
    job_list = ['b1_anchor', 'b1_hfNorm', 'b1_oddV_MfV', 'b1_rnn']
    job_list = [ 'b1_rnn_ori']

    for folder in cross_list[3:4]:
        print('current job folder is: ', folder)
        for job in job_list:
            cur_work_dir = os.path.join(work_dir, folder, job)
            if os.path.exists(os.path.join(cur_work_dir, 'hyperparameters.json')):
                print('Loading hyperparameters from ' + os.path.join(cur_work_dir, 'hyperparameters.json'))
                with open(os.path.join(cur_work_dir, 'hyperparameters.json'), 'r') as f:
                    params = json.load(f)
            else:
                print('Error: File not exist! ' + cur_work_dir + '/hyperparameters.json')
                exit(-1)
            if folder == '70558_sf_f1000_ns3_nm10':
                net_path = os.path.join(cur_work_dir, 'netparams_b1_t9_v9_e3000.dat')
            else:
                net_path = os.path.join(cur_work_dir, 'netparams_b1_t10_v10_e3000.dat')

            net = SubdNet(params)
            net = net.to(params['device'])
            net.load_state_dict(torch.load(net_path, map_location=torch.device(params["device"])))
            net.eval()
            for test_folder in cross_list:
                print('\ttest folder: ', test_folder)
                if test_folder != folder:
                    continue
                test_pkl = os.path.join(mesh_pkl_dir, test_folder + '.pkl')
                T = pickle.load(open(test_pkl, "rb"))
                T.computeParameters()
                T.toDevice(params["device"])
                out_test_dir = os.path.join(cur_work_dir, 'test_e3000')
                test_and_evaluate(out_test_dir, net, T, params['numSubd'])


def test_models_midx():

    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'
    # load hyper parameters

    folder_list = ['anchor_subd2', 'hf_norm_12_pos', 'hf_norm_12_pos+fea', 'hf_norm_12x34_pos+fea']

    folder_list = ['anchor_subd2']
    folder_list = ['odd_hiddenLayer2_dout64','odd_hiddenLayer4_dout3','odd_hiddenLayer4_dout64']
    folder_list = ['hf_norm_12_pos+fea_trainloss']
    folder_list = ['b1_anchor', 'b1_hfNorm_12', 'b1_hfNorm_12x34', 'b1_oddV_netIV']
    folder_list = ['b8_hfNorm']

    folder = folder_list[0]
    print('current net is: ', folder)

    if os.path.exists(os.path.join(work_dir, folder, 'hyperparameters.json')):
        print('Loading hyperparameters from ' + os.path.join(work_dir, folder, 'hyperparameters.json'))
        with open(os.path.join(work_dir, folder, 'hyperparameters.json'), 'r') as f:
            params = json.load(f)
    else:
        print('Loading hyperparameters from ' + work_dir + 'hyperparameters.json')
        with open(work_dir + 'hyperparameters.json', 'r') as f:
            params = json.load(f)
    # params['epochs'] =6000
    params['numSubd'] = 3
    net_path = os.path.join(work_dir, folder, 'netparams_b8_t300_v30_e2000.dat')
    # initialize network
    net = SubdNet(params)
    net = net.to(params['device'])
    net.load_state_dict(torch.load(net_path, map_location=torch.device(params["device"])))
    net.eval()



    output_root_dir = os.path.join(work_dir, folder, 'test_10k_sf_train60_subdiv_fn1000_e2000')
    # 训练集 
    # mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    # mesh_list = ['Camel_f1000_ns3_nm50', 'Dog_f1000_ns3_nm50', 'Lion_f1000_ns3_nm50', 'Panda_f1000_ns3_nm50']
    # 测试集 
    #mesh_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/'
    #mesh_list = ['bob_f600_ns3_nm20']

    # 测试集：
    mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60_subdiv_fn1000/'
    # 将mesh_root_dir中的文件夹名字作为mesh_list

    mesh_list = os.listdir(mesh_root_dir)

    
    for mesh_name in mesh_list:
        max_num = 10
        for i in range(1, max_num, 10):
            mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd0')
            meshPath = [os.path.join(mesh_dir, str(i).zfill(len(str(max_num)))+'.obj')]
            output_dir = os.path.join(output_root_dir, mesh_name)
            os.path.exists(output_dir) or os.makedirs(output_dir)
            out_path = os.path.join(output_dir, str(i).zfill(len(str(max_num)))+'_subd'+str(params['numSubd'])+'.obj')
            print(meshPath)
            T = TestMeshes(meshPath, params['numSubd'])
            T.computeParameters()
            if not torch.cuda.is_available():
                params['device'] = 'cpu'
            T.toDevice(params["device"])

            # write output shapes (test set)
            mIdx = 0
            scale = T.getScale(mIdx) # may need to adjust the scale of the mesh since the network is not scale invariant

            x = T.getInputData(mIdx)
            #outputs = net.forward_midfeature(x, T.hfList[mIdx],T.poolMats[mIdx],T.dofs[mIdx])
            outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])

            ii = len(outputs) - 1
            x = outputs[ii].cpu() * scale[1] + scale[0].unsqueeze(0)
            # tgp.writeOBJ(params['output_path'] + meshName + '_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
            tgp.writeOBJ(out_path, x, T.meshes[mIdx][ii].F.to('cpu'))
            print('write to ', out_path)
def test_models():

    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'
    # load hyper parameters

    folder_list = ['anchor_subd2', 'hf_norm_12_pos', 'hf_norm_12_pos+fea', 'hf_norm_12x34_pos+fea']

    folder_list = ['anchor_subd2']
    folder_list = ['odd_hiddenLayer2_dout64','odd_hiddenLayer4_dout3','odd_hiddenLayer4_dout64']
    folder_list = ['hf_norm_12_pos+fea_trainloss']
    for folder in folder_list:
        print('current net is: ', folder)

        if os.path.exists(os.path.join(work_dir, folder, 'hyperparameters.json')):
            print('Loading hyperparameters from ' + os.path.join(work_dir, folder, 'hyperparameters.json'))
            with open(os.path.join(work_dir, folder, 'hyperparameters.json'), 'r') as f:
                params = json.load(f)
        else:
            print('Loading hyperparameters from ' + work_dir + 'hyperparameters.json')
            with open(work_dir + 'hyperparameters.json', 'r') as f:
                params = json.load(f)
        # params['epochs'] =6000
        params['numSubd'] = 3
        net_path = os.path.join(work_dir, folder, 'netparams_e2000_t300_v30.dat')
        # initialize network
        net = SubdNet(params)
        net = net.to(params['device'])
        net.load_state_dict(torch.load(net_path, map_location=torch.device(params["device"])))
        net.eval()

        output_root_dir = os.path.join(work_dir, folder, 'test_e2000')

        mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
        mesh_list = ['Camel_f1000_ns3_nm50', 'Dog_f1000_ns3_nm50', 'Lion_f1000_ns3_nm50', 'Panda_f1000_ns3_nm50']
        for mesh_name in mesh_list:
            max_num = 50
            for i in range(1, max_num, 10):
                mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd0')
                meshPath = [os.path.join(mesh_dir, str(i).zfill(len(str(max_num)))+'.obj')]
                output_dir = os.path.join(output_root_dir, mesh_name)
                os.path.exists(output_dir) or os.makedirs(output_dir)
                out_path = os.path.join(output_dir, str(i).zfill(len(str(max_num)))+'_subd'+str(params['numSubd'])+'.obj')
                print(meshPath)
                T = TestMeshes(meshPath, params['numSubd'])
                T.computeParameters()
                if not torch.cuda.is_available():
                    params['device'] = 'cpu'
                T.toDevice(params["device"])

                # write output shapes (test set)
                mIdx = 0
                scale = T.getScale(mIdx) # may need to adjust the scale of the mesh since the network is not scale invariant

                x = T.getInputData(mIdx)
                outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)

                ii = len(outputs) - 1
                x = outputs[ii].cpu() * scale[1] + scale[0].unsqueeze(0)
                # tgp.writeOBJ(params['output_path'] + meshName + '_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
                tgp.writeOBJ(out_path, x, T.meshes[mIdx][ii].F.to('cpu'))
                print('write to ', out_path)
def main():

    # load hyper parameters
    # folder = sys.argv[1]
    folder_list = ['jobs/net_Horse_f1000_ns2_nm200','jobs/net_Horse_f1000_ns3_nm200','jobs/net_Horse_f1000_ns3_nm200_32*3','jobs/net_Horse_f1000_ns3_nm200_64*2']
    for folder in folder_list:
        print('current net is: ', folder)
        with open(os.path.join(folder,'hyperparameters.json'), 'r') as f:
            params = json.load(f)
        params['numSubd'] = 3  # number of subdivision levels at test time
        params['output_path'] = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/neural_subdiv_'+folder
        mesh_path_list = []
        mesh_file_list = []
        # for path in os.listdir(sys.argv[2]):
        #     # check if current path is a obj file
        #     if os.path.splitext(path)[-1] == ".obj":
        #         mesh_file_list.append(path)
        #         file_path = os.path.join(sys.argv[2], path)
        #         mesh_path_list.append(file_path)
        mesh_path_list = ['/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/watertight5_1000/Horse_f1000.obj']
        mesh_file_list = ['Horse_f1000.obj']
        print('The following mesh will be subdivided.')
        print(mesh_file_list)

        if os.path.isdir(params['output_path']):
            print('文件夹已存在')
            pass
        else:
            os.makedirs(params['output_path'])


        # initialize network
        net = SubdNet(params)
        net = net.to(params['device'])
        net.load_state_dict(torch.load(folder + '/' + NETPARAMS, map_location=torch.device(params["device"])))
        net.eval()

        # load validation set
        meshPath = mesh_path_list
        T = TestMeshes(meshPath, params['numSubd'])
        T.computeParameters()
        if not torch.cuda.is_available():
            params['device'] = 'cpu'
        T.toDevice(params["device"])

        for mIdx, filename in enumerate(mesh_file_list):
            filename = filename[:-4]

            # write output shapes (test set)
            # mIdx = 0
            scale = 1.0 # may need to adjust the scale of the mesh since the network is not scale invariant

            x = T.getInputData(mIdx)
            t0 = time.time()
            outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)
            t1 = time.time()-t0
            print('[iter: %d]  [filename: %s] [cost time: %d]' % (params['numSubd'], filename,  t1))
            for ii in range(len(outputs)):
                x = outputs[ii].cpu() * scale
                if ii == 0:

                    tgp.writeOBJ(os.path.join(params['output_path'], filename + '_subd' + '_ori' + '.obj'),x, T.meshes[mIdx][ii].F.to('cpu'))
                else:
                    tgp.writeOBJ(os.path.join(params['output_path'], filename + '_subd' + str(ii-1) + '.obj'), x,
                                  T.meshes[mIdx][ii].F.to('cpu'))

            # write rotated output shapes
            # x = T.getInputData(mIdx)
            #
            # dV = torch.rand(1, 3).to(params['device'])
            # R = random3DRotation().to(params['device'])
            # x[:,:3] = x[:,:3].mm(R.t())
            # x[:,3:] = x[:,3:].mm(R.t())
            # x[:,:3] += dV
            # outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)
            #
            # for ii in range(len(outputs)):
            #     x = outputs[ii].cpu() * scale
            #     tgp.writeMESH(params['output_path'] + filename + '_rot_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))

def test():

    # load hyper parameters
    folder = sys.argv[1]
    meshPath = [sys.argv[2]]
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)
    params['numSubd'] = int(sys.argv[4]) # number of subdivision levels at test time
    output_mesh_path = sys.argv[3]

    print(os.path.basename(sys.argv[2]))

    # load validation set

    T = TestMeshes(meshPath, params['numSubd'])
    T.computeParameters()
    if not torch.cuda.is_available():
        params['device'] = 'cpu'
    T.toDevice(params["device"])

    # initialize network
    net = SubdNet(params)
    net = net.to(params['device'])
    net.load_state_dict(torch.load(params['output_path'] + NETPARAMS, map_location=torch.device(params["device"])))
    net.eval()

    # write output shapes (test set)
    mIdx = 0
    scale = 1.0 # may need to adjust the scale of the mesh since the network is not scale invariant
    meshName = os.path.basename(sys.argv[2])[:-4] # meshName: "bunny"
    x = T.getInputData(mIdx)
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)

    ii = len(outputs) -1
    x = outputs[ii].cpu() * scale
    tgp.writeOBJ(output_mesh_path, x, T.meshes[mIdx][ii].F.to('cpu'))

def test2():

    parse = argparse.ArgumentParser()
    parse.add_argument('-j', '--job_folder', type=str, default='./jobs//net_animal-3kk_ns3_nm50x6_norm/')
    parse.add_argument('-i', '--mesh_dir', type=str, default='/work/Users/zhuzhiwei/dataset/mesh/'
                                                             'free3d-animal/animal-3kk/')
    parse.add_argument('-p', '--pretrain_net', type=str, default='netparams_1400.dat',
                       help='pretrained net parameters file name')
    parse.add_argument('-o', '--output_dir', type=str, default='test')
    parse.add_argument('-s', '--num_subd', type=int, default=3)
    args = parse.parse_args()
    # print args information
    print('job folder: ', args.job_folder)
    print('mesh dir: ', args.mesh_dir)
    print('pretrain net: ', args.pretrain_net)
    print('output dir: ', args.output_dir)
    print('num subd: ', args.num_subd)

    folder = args.job_folder
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)

    output_file_path = os.path.join(folder, args.output_dir)
    os.path.exists(output_file_path) or os.makedirs(output_file_path)

    mesh_name = ['Panda', 'Camel', 'Dog', 'Elephant', 'Lion']
    test = args.mesh_dir
    num_subd = 3

    for name in mesh_name:
        mesh_dir = name + '_f1000_ns3_nm50/subd0/'
        output_name = os.path.join(output_file_path, name)
        os.path.exists(output_name) or os.makedirs(output_name)
        print(mesh_dir)

        max_num = 50
        d1_psnr_list = []

        for i in range(1, max_num, 5):
            fill = len(str(max_num))

            test_path = os.path.join(test, mesh_dir, str(i).zfill(fill) + '.obj')
            meshPath = [test_path]
            out_path = os.path.join(output_name, str(i).zfill(fill) + '_subd' + str(num_subd) + '.obj')
            print(meshPath)
            T = TestMeshes(meshPath, num_subd)
            T.computeParameters()
            if not torch.cuda.is_available():
                params['device'] = 'cpu'
            T.toDevice(params["device"])

            # initialize network
            net = SubdNet(params)
            net = net.to(params['device'])
            net.load_state_dict(torch.load(folder + args.pretrain_net, map_location=torch.device(params["device"])))
            net.eval()

            # write output shapes (test set)
            mIdx = 0
            scale = T.getScale(
                mIdx)  # may need to adjust the scale of the mesh since the network is not scale invariant

            x = T.getInputData(mIdx)
            outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

            ii = len(outputs) - 1
            x = outputs[ii].cpu() * scale[1] + scale[0].unsqueeze(0)
            # tgp.writeOBJ(params['output_path'] + meshName + '_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
            tgp.writeOBJ(out_path, x, T.meshes[mIdx][ii].F.to('cpu'))

def bat_test():

    parse = argparse.ArgumentParser()
    parse.add_argument('-j', '--job_folder',  type=str, default='./jobs/Owl_voxelized12_f5975_ns3_nm30+10_e1500/')
    parse.add_argument('-i', '--mesh_dir', type=str, default='/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal'
                                                             '-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd0/')
    parse.add_argument('-p', '--pretrain_net', type=str, default='netparams.dat', help='pretrained net parameters file name')
    parse.add_argument('-o', '--output_dir', type=str, default='test')
    parse.add_argument('-s', '--num_subd', type=int, default=3)
    args = parse.parse_args()
    # load hyper parameters
    output_dir = os.path.join(args.job_folder, args.output_dir)
    os.path.exists(output_dir) or os.makedirs(output_dir)
    folder = args.job_folder
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)


    # load validation set

    for i in range(1, 30, 3):
        meshPath = [os.path.join(args.mesh_dir, str(i).zfill(2)+'.obj')]
        out_path = os.path.join(output_dir, str(i).zfill(2)+'_subd'+str(args.num_subd)+'.obj')
        print(meshPath)
        T = TestMeshes(meshPath, args.num_subd)
        T.computeParameters()
        if not torch.cuda.is_available():
            params['device'] = 'cpu'
        T.toDevice(params["device"])

        # initialize network
        net = SubdNet(params)
        net = net.to(params['device'])
        net.load_state_dict(torch.load(folder + args.pretrain_net, map_location=torch.device(params["device"])))
        net.eval()

        # write output shapes (test set)
        mIdx = 0
        scale = T.getScale(mIdx) # may need to adjust the scale of the mesh since the network is not scale invariant

        x = T.getInputData(mIdx)
        outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)

        ii = len(outputs) - 1
        x = outputs[ii].cpu() * scale[1] + scale[0].unsqueeze(0)
        # tgp.writeOBJ(params['output_path'] + meshName + '_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
        tgp.writeOBJ(out_path, x, T.meshes[mIdx][ii].F.to('cpu'))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    self_test()

    # gt_points = torch.randn(10,3)
    # pred_points = torch.randn(10,3)
    # lossfunc = torch.nn.MSELoss()
    # print(lossfunc(gt_points, pred_points))
    # print(chamfer_distance(gt_points, pred_points))

    # mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60_subdiv_fn1000/'
    # # 将mesh_root_dir中的文件夹名字作为mesh_list
    #
    # mesh_list = os.listdir(mesh_root_dir)
    # delete_list = []
    #
    # for mesh_name in mesh_list:
    #     max_num = 10
    #     for i in range(1, max_num, 10):
    #         mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd0')
    #         meshPath = os.path.join(mesh_dir, str(i).zfill(len(str(max_num)))+'.obj')
    #         if os.path.getsize(meshPath) < 1024:
    #             # 1024 bytes
    #            delete_list.append(os.path.join(mesh_root_dir, mesh_name))
    #            os.system('rm -rf '+os.path.join(mesh_root_dir, mesh_name))
    #            continue
    # print(delete_list)


