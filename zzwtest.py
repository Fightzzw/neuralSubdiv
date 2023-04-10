from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from include import *
from models import *
NETPARAMS = 'netparams.dat'

# run the file as "python test.py ./path/to/folder/ ./path/to/testMesh.obj"
# WARNING
# - the current code only works for .obj file without boundaries :P 
# - there are two failure cases mentioned in the paper:
#   1. testing triangles has aspect ratios that are not in the training data
#   2. testing triangles has triangle size that are not in the training data
#   (we include a failure case in the data: the ear of the horse.obj)

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



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    test()
