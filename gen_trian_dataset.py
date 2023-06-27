import os


def gen_train_valid_test():
    info_file = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
                '10k_surface_1component+watertight+noSelfIntersect_5550_info.txt'

    with open(info_file, 'r') as f:
        lines = f.readlines()
        # 将lines分为train, valid, test三部分
        train_lines = lines[:int(len(lines) * 0.8)]
        valid_lines = lines[int(len(lines) * 0.8):int(len(lines) * 0.9)]
        test_lines = lines[int(len(lines) * 0.9):]
        # output train, valid, test
        train_file = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
                     '10k_surface_5550_train.txt'
        valid_file = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
                     '10k_surface_5550_valid.txt'
        test_file = '/work/Users/zhuzhiwei/dataset/mesh/info_10k_surface/' \
                    '10k_surface_5550_test.txt'
        with open(train_file, 'w') as f:
            for line in train_lines:
                f.write(line)
        with open(valid_file, 'w') as f:
            for line in valid_lines:
                f.write(line)
        with open(test_file, 'w') as f:
            for line in test_lines:
                f.write(line)

def random_subdiv_faceratio():
    target = [0.03, 0.06]
    exe_path = '/work/Users/zhuzhiwei/surface_multigrid_code/' \
               '09_random_subdiv_remesh/build_tarRatiov2/random_subdiv_remesh_bin'
    obj_dir = "/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60"
    file_list = os.listdir(obj_dir)
    for idx, obj_name in enumerate(file_list):
        print('%d/%d: %s' % (idx, len(file_list), obj_name))
        obj_path = os.path.join(obj_dir, obj_name)
        for t in target:
            output_dir = obj_dir + '_subdiv_fr' + str(t)
            cmd = exe_path + ' ' + obj_path + ' ' + output_dir + ' ' + str(t) + ' 3 10'
            print(cmd)
            os.system(cmd)

def random_subdiv_facenum():
    facenum = [500, 1000]
    exe_path = '/work/Users/zhuzhiwei/surface_multigrid_code/' \
                  '09_random_subdiv_remesh/build_tarfaces/random_subdiv_remesh_bin'
    obj_dir = "/work/Users/zhuzhiwei/dataset/mesh/10k_sf_train60"
    file_list = os.listdir(obj_dir)
    for idx, obj_name in enumerate(file_list):
        print('%d/%d: %s' % (idx, len(file_list), obj_name))
        obj_path = os.path.join(obj_dir, obj_name)
        for f in facenum:
            output_dir = obj_dir + '_subdiv_fn' + str(f)
            cmd = exe_path + ' ' + obj_path + ' ' + output_dir + ' ' + str(f) + ' 3 10'
            print(cmd)
            os.system(cmd)

if __name__ == '__main__':
    random_subdiv_facenum()

