from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Pool
from include import *



def multiple_mesh():
    train_mesh_folders = []
    test_mesh_folders = []
    mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    for path in os.listdir(mesh_dir):
        if path[-4:] == 'nm50':
            train_mesh_folders.append(os.path.join(mesh_dir,path))
        if path[-3:] == 'nm5':
            test_mesh_folders.append(os.path.join(mesh_dir, path))
    print('train_mesh_folders: ', train_mesh_folders)
    print('test_mesh_folders: ', test_mesh_folders)
    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S1 = TrainMeshes(train_mesh_folders)

    pickle.dump(S1, file=open("./data_PKL/animal-3kk_ns3_nm50*6_train.pkl", "wb"))
    S2 = TrainMeshes(test_mesh_folders)

    pickle.dump(S2, file=open("./data_PKL/animal-3kk_ns3_nm50*6_valid.pkl", "wb"))

def multiple_mesh1():
    train_mesh_folders = []

    mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    for path in os.listdir(mesh_dir):
        if path[-4:] == 'nm50':
            train_mesh_folders.append(os.path.join(mesh_dir,path))

    print('train_mesh_folders: ', train_mesh_folders)

    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S1 = TrainMeshes(train_mesh_folders)

    pickle.dump(S1, file=open("./data_PKL/animal-3kk_ns3_nm50*6_norm_train.pkl", "wb"))


def multiple_mesh2():

    test_mesh_folders = []
    mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    for path in os.listdir(mesh_dir):

        if path[-3:] == 'nm5':
            test_mesh_folders.append(os.path.join(mesh_dir, path))

    print('test_mesh_folders: ', test_mesh_folders)
    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']

    S2 = TrainMeshes(test_mesh_folders)

    pickle.dump(S2, file=open("./data_PKL/animal-3kk_ns3_nm50*6_norm_valid.pkl", "wb"))



def main():

    train_mesh_folders =['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    test_mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm20/']

    print('train_mesh_folders: ', train_mesh_folders)
    print('test_mesh_folders: ', test_mesh_folders)
    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S1 = TrainMeshes(train_mesh_folders)
    pickle.dump(S1, file=open("./data_PKL/bob_f600_ns3_nm100_train.pkl", "wb"))
    S2 = TrainMeshes(test_mesh_folders)
    pickle.dump(S2, file=open("./data_PKL/bob_f600_ns3_nm100_valid.pkl", "wb"))


def main1():
    train_mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']

    print('train_mesh_folders: ', train_mesh_folders)

    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S1 = TrainMeshes(train_mesh_folders)
    pickle.dump(S1, file=open("./data_PKL/bob_f600_ns3_nm100_train.pkl", "wb"))


def main2():
    test_mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm20/']
    print('test_mesh_folders: ', test_mesh_folders)
    # mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']

    S2 = TrainMeshes(test_mesh_folders)
    pickle.dump(S2, file=open("./data_PKL/bob_f600_ns3_nm100_valid.pkl", "wb"))




if __name__ == '__main__':
    # main()
    # multiple_mesh()
    # train_mesh_folders = ['/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm100']
    # # mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    # # for path in os.listdir(mesh_dir):
    # #     if path[-4:] == 'nm50':
    # #         train_mesh_folders.append(os.path.join(mesh_dir, path))
    # for model in train_mesh_folders:
    #     for sub in os.listdir(model):
    #         sub_dir = os.path.join(model, sub)
    #         print('rm ' + sub_dir + '/000.obj')
    #         os.system('rm ' + sub_dir + '/000.obj')


    n_processes = os.cpu_count()
    print('n_processes: ', n_processes)
    n_processes = 2  # 进程数
    pool = Pool(processes=n_processes)  # 进程池

    # pool.apply_async(main1)  # 启动多进程
    # pool.apply_async(main2)  # 启动多进程
    pool.apply_async(multiple_mesh1)

    pool.apply_async(multiple_mesh2)

    pool.close()  # 使进程池不能添加新任务
    pool.join()  # 等待进程结束

