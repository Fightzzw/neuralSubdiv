from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from include import *
from models.models_hfNorm import *
NETPARAMS = 'netparams.dat'
import argparse

# run the file as "python test.py ./path/to/folder/ ./path/to/testMesh.obj"
# WARNING
# - the current code only works for .obj file without boundaries :P
# - there are two failure cases mentioned in the paper:
#   1. testing triangles has aspect ratios that are not in the training data
#   2. testing triangles has triangle size that are not in the training data
#   (we include a failure case in the data: the ear of the horse.obj)

def test_models():

    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'
    # load hyper parameters

    folder_list = ['anchor_subd2', 'hf_norm_12_pos', 'hf_norm_12_pos+fea', 'hf_norm_12x34_pos+fea']

    folder_list = ['anchor_subd2']
    folder_list = ['odd_hiddenLayer2_dout64','odd_hiddenLayer4_dout3','odd_hiddenLayer4_dout64']
    folder_list = ['hf_norm_12x34_pos+fea']
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
        net_path = os.path.join(work_dir, folder, 'netparams_%d.dat'% params['epochs'])
        # initialize network
        net = SubdNet(params)
        net = net.to(params['device'])
        net.load_state_dict(torch.load(net_path, map_location=torch.device(params["device"])))
        net.eval()

        output_root_dir = os.path.join(work_dir, folder, 'test')

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
    test_models()

