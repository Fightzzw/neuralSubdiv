from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.pool import ThreadPool as Pool
from include import *


def process_train_mesh_folders_output(idx, folders, o_dir):
    train_mesh = TrainMeshes(folders)
    output_path = os.path.join(o_dir, str(idx) + '.pkl')
    pickle.dump(train_mesh, file=open(output_path, "wb"))

def process_train_mesh_folders_output_v2(folders, output_path):
    train_mesh = TrainMeshes(folders)
    pickle.dump(train_mesh, file=open(output_path, "wb"))

def process_train_mesh_folders(idx, folders, o_dir):
    train_mesh = TrainMeshes(folders)
    return train_mesh


def error_callback(e):
    print('error_callback: ', e)


def merge_pkl(i_dir, o_name):
    # i_dir = './data_PKL/10k_surface_fr0.06_ns3_nm5550_train'
    # o_name = './data_PKL/10k_surface_fr0.06_ns3_nm5550_train.pkl'

    pkl_list = []
    for path in os.listdir(i_dir):
        if path.endswith('.pkl'):
            pkl_list.append(os.path.join(i_dir, path))
    print('len of pkl_list: ', len(pkl_list))
    flag = 1
    for pkl in pkl_list:
        split_meshes = pickle.load(open(pkl, 'rb'))
        if flag:
            total_meshes = split_meshes
            flag = 0

        total_meshes.appendMeshList(split_meshes.meshes)

    print('len of total mesh: ', total_meshes.nM)
    output_path = os.path.join(i_dir, o_name + '_total_%d.pkl' % total_meshes.nM)
    pickle.dump(total_meshes, file=open(output_path, "wb"))


def multiple_gen_data_pkl_and_merge(train_mesh_folders, output_dir):
    # obj_list = []
    # root_dir = "/work/Users/zhuzhiwei/dataset/mesh/10k_surface_5550_subdiv_fr0.06/"
    # train_txt = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
    #             '10k_surface_5550_valid.txt'
    # with open(train_txt, 'r') as f:
    #     for line in f.readlines()[1:]:
    #         obj_list.append(os.path.join(root_dir, line.strip('\n').split('\t')[0][:-4]))
    #
    # train_mesh_folders = obj_list
    #
    # output_dir = './data_PKL/10k_surface_5550_subdiv_fr0.06_valid'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('len of train_mesh_folders: ', len(train_mesh_folders))

    n_processes = os.cpu_count()

    print('n_processes: ', n_processes)
    pool = Pool(processes=n_processes)  # 进程池
    # split folders into n_processes parts
    split_folders = np.array_split(train_mesh_folders, n_processes)
    split_meshes = [None] * n_processes

    # start multi-processes
    for i in range(n_processes):
        # pool.apply_async(processFolder, args=(split_folders[i],), callback=lambda x: split_meshes.__setitem__(i, x))
        pool.apply_async(process_train_mesh_folders,
                         args=(i, split_folders[i], output_dir,),
                         callback=lambda x: split_meshes.__setitem__(i, x),
                         error_callback=error_callback)
        print('processFolder: ', i)

    pool.close()
    pool.join()

    # merge results
    train_mesh = TrainMeshes([])
    for i in range(n_processes):
        train_mesh.appendMeshList(split_meshes[i].meshes,
                                  split_meshes[i].hfList,
                                  split_meshes[i].poolMats,
                                  split_meshes[i].dofs,
                                  split_meshes[i].LCs)
    output_path = os.path.join(output_dir, 'valid_%d.pkl' % (len(train_mesh.nM)))
    pickle.dump(train_mesh, file=open(output_path, "wb"))


def multiple_gen_data_pkl(train_mesh_folders, output_dir, n_processes=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('len of train_mesh_folders: ', len(train_mesh_folders))
    # n_processes = os.cpu_count()
    print('n_processes: ', n_processes)
    pool = Pool(processes=n_processes)  # 进程池
    # split folders into n_processes parts
    split_folders = np.array_split(train_mesh_folders, n_processes)

    # start multi-processes
    processes = []
    for i in range(n_processes):
        # p = Process(target=process_train_mesh_folders_output, args=(i, split_folders[i], output_dir,))
        # p.start()
        # processes.append(p)
        # pool.apply(process_train_mesh_folders_output, args=(i, split_folders[i], output_dir,))
        pool.apply_async(process_train_mesh_folders_output,
                         args=(i, split_folders[i], output_dir,),
                         error_callback=error_callback)
        print('processFolder: ', i)
    # for p in processes:
    #     p.join()

    pool.close()
    pool.join()

def multiple_gen_data_pkl_v2(train_mesh_dir, output_dir, n_processes=1):

    train_mesh_folders_name = os.listdir(train_mesh_dir)
    train_mesh_folders = [os.path.join(train_mesh_dir, name) for name in train_mesh_folders_name[3:4]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('len of train_mesh_folders: ', len(train_mesh_folders))
    # n_processes = os.cpu_count()
    n_processes = len(train_mesh_folders)

    print('n_processes: ', n_processes)
    pool = Pool(processes=n_processes)  # 进程池
    # split folders into n_processes parts
    split_folders = np.array_split(train_mesh_folders, n_processes)

    # start multi-processes
    processes = []
    for i in range(n_processes):
        output_path = os.path.join(output_dir, train_mesh_folders_name[i] + '.pkl')
        pool.apply_async(process_train_mesh_folders_output_v2,
                         args=(split_folders[i], output_path,),
                         error_callback=error_callback)
        print('processFolder: ', i)


    pool.close()
    pool.join()

def base_gen_data_pkl():
    train_mesh_folders = []

    mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60_subdiv_fn1000/'
    for path in os.listdir(mesh_dir):
        train_mesh_folders.append(os.path.join(mesh_dir, path))

    print('len(train_mesh_folders): ', len(train_mesh_folders))

    S1 = TrainMeshes(train_mesh_folders)
    pickle.dump(S1, file=open("./data_PKL/10k_sf_train60_subdiv_fr0.06.pkl", "wb"))


def process_with_txt():
    obj_list = []
    root_dir = "/work/Users/zhuzhiwei/dataset/mesh/10k_surface_5550_subdiv_fr0.06/"
    train_txt = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
                '10k_surface_5550_valid.txt'
    with open(train_txt, 'r') as f:
        for line in f.readlines()[1:]:
            obj_list.append(os.path.join(root_dir, line.strip('\n').split('\t')[0][:-4]))

    train_mesh_folders = obj_list

    print('train_mesh_folders: ', train_mesh_folders)

    S2 = TrainMeshes(train_mesh_folders)

    pickle.dump(S2, file=open("./data_PKL/10k_surface_fr0.06_ns3_nm5550_valid.pkl", "wb"))


if __name__ == '__main__':
    multiple_gen_data_pkl_v2('/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60_subdiv_fn1000/',
                             '/work/Users/zhuzhiwei/neuralSubdiv-master/data_PKL/10k_sf_train60_fn1000/')
    # main2()
    # obj_list = []
    # root_dir = "/work/Users/zhuzhiwei/dataset/mesh/10k_surface_5550_subdiv_fr0.06/"
    # train_txt = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
    #             '10k_surface_5550_train.txt'
    # with open(train_txt, 'r') as f:
    #     for line in f.readlines()[1:]:
    #         obj_list.append(os.path.join(root_dir, line.strip('\n').split('\t')[0][:-4]))
    #
    # train_mesh_folders = obj_list
    # output_dir = './data_PKL/10k_surface_fr0.06_ns3_nm5550_train'
    # # multiple_gen_data_pkl(train_mesh_folders, output_dir, 256)
    # # merge_pkl('./data_PKL/10k_surface_fr0.06_ns3_nm5550_train', 'train')
    # pkl_list = []
    # for dir in os.listdir(output_dir):
    #     dir_path = os.path.join(output_dir, dir)
    #     for pkl in os.listdir(dir_path):
    #         if pkl.startswith('split'):
    #             pkl_list.append(os.path.join(dir_path, pkl))
    #             print('pkl: ', pkl)
    # print('len of pkl_list: ', len(pkl_list))
    #
    # flag = 1
    #
    # for pkl in pkl_list:
    #     split_meshes = pickle.load(open(pkl, 'rb'))
    #     if flag:
    #         total_meshes = split_meshes
    #         flag = 0
    #
    #     total_meshes.appendMeshList(split_meshes.meshes)
    #
    # print('len of total mesh: ', total_meshes.nM)
    # output_path = os.path.join(output_dir, 'train_total_%d.pkl' % total_meshes.nM)
    # pickle.dump(total_meshes, file=open(output_path, "wb"))
