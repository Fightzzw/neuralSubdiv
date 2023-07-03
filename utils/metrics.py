import os
# import sys
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import matplotlib
# import xlwt
# import xlrd
# import time
# import math
# import pandas as pd
from my_meshFunction import cal_d1_psnr

def bat_call_cal_d1_psnr():
    numSubd = 3
    output_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/'

    mesh_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv/data_meshes/'
    mesh_list = ['jaw_f1000_ns3_nm200', 'Horse_f1000_ns3_nm200', 'horse_f600_ns3_nm200']

    anchor_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/anchor/test'
    hdNorm_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/hf_norm_12_pos+fea/test/'
    gps_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/gps/'
    loop_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/loop/'

    job_list =[anchor_dir, hdNorm_dir, gps_dir, loop_dir]

    f_log = open(output_root_dir + 'd1_psnr.txt', 'w')
    f_log.write('pred_mesh_name\tanchor\thdNorm\tgps\tloop\n')

    for mesh_name in mesh_list:
        f_log.write(mesh_name + '\n')
        print(mesh_name)
        for i in range(1, 200, 20):
            gt_mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd3')
            gt_meshPath = os.path.join(gt_mesh_dir, str(i).zfill(3) + '.obj')
            pred_mesh_name = str(i).zfill(3) + '_subd' + str(numSubd) + '.obj'
            print('\t'+pred_mesh_name)
            f_log.write(pred_mesh_name + '\t')
            for job in job_list:
                pred_meshPath = os.path.join(job, mesh_name, pred_mesh_name)
                d1_psnr = cal_d1_psnr(gt_meshPath, pred_meshPath)
                if job == job_list[-1]:
                    f_log.write(str(d1_psnr) + '\n')
                else:
                    f_log.write(str(d1_psnr) + '\t')
                print('\t\t' + str(d1_psnr))
    f_log.close()

def bat_call_mm():
    print('start bat_cal_vmesh_d1psnr ')
    refer_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/'
    work_dir = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/subd3_net_animal-3kk_ns3_nm50x6_norm/'
    output_file_path = work_dir + '/Camel+Lion_neural_d1_psnr_number.txt'

    job_list = ['Camel', 'Lion']

    for job in job_list:
        print(job)
        refer = refer_dir + job + '_voxelized12.obj'
        for i in range(4):
            dec_mesh_path = work_dir + job + '/dec/r' + str(i + 1) + '/' + job + '_draco_dec.obj'
            call_mm_compare(refer, dec_mesh_path, 'pcc')


def call_mm_compare(refer, test, mode='pcc'):
    mm = '/work/Users/zhuzhiwei/vmesh/externaltools/mpeg-pcc-mmetric/build/Release/bin/mm'
    cmd = mm + ' compare --inputModelA ' + refer + ' --inputModelB ' + test + ' --mode ' + mode
    print(cmd)
    os.system(cmd)




if __name__ == '__main__':
    numSubd = 3
    output_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/'

    mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    mesh_list = ['Camel_f1000_ns3_nm50', 'Dog_f1000_ns3_nm50', 'Lion_f1000_ns3_nm50', 'Panda_f1000_ns3_nm50']

    anchor_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/anchor/test'
    hdNorm_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/hf_norm_12_pos+fea/test/'
    gps_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/gps/'
    loop_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/loop/'

    for mesh_name in mesh_list:
        max_num = 50
        for i in range(1, max_num, 10):
            mesh_dir = os.path.join(anchor_dir, mesh_name)
            meshPath = os.path.join(mesh_dir,
                                    str(i).zfill(3) + '_subd3' + '.obj')

            out_path = os.path.join(mesh_dir,
                                    str(i).zfill(len(str(max_num))) + '_subd3' + '.obj')
            cmd = 'mv ' + meshPath + ' '+out_path
            print(cmd)
            os.system(cmd)


    # mesh_name = mesh_list[1]
    # gt_mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd3')
    # gt_meshPath = os.path.join(gt_mesh_dir, str(1).zfill(2) + '.obj')
    # pred_mesh_name = str(1).zfill(3) + '_subd' + str(numSubd) + '.obj'
    # pred_meshPath1 = os.path.join(anchor_dir, mesh_name, pred_mesh_name)
    # pred_meshPath2 = os.path.join(hdNorm_dir, mesh_name, pred_mesh_name)
    # call_mm_compare(gt_meshPath, pred_meshPath1)
    # call_mm_compare(gt_meshPath, pred_meshPath2)
    

    #job_list =[anchor_dir, hdNorm_dir, gps_dir, loop_dir]
    #
    #f_log = open(output_root_dir + 'd1_psnr.txt', 'w')
    #f_log.write('pred_mesh_name\tanchor\thdNorm\tgps\tloop\n')
    #
    #for mesh_name in mesh_list:
    #    f_log.write(mesh_name + '\n')
    #    print(mesh_name)
    #    for i in range(1, 200, 20):
    #        gt_mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd3')
    #        gt_meshPath = os.path.join(gt_mesh_dir, str(i).zfill(3) + '.obj')
    #        pred_mesh_name = str(i).zfill(3) + '_subd' + str(numSubd) + '.obj'
    #        print('\t'+pred_mesh_name)
    #        f_log.write(pred_mesh_name + '\t')
    #        for job in job_list:
    #            pred_meshPath = os.path.join(job, mesh_name, pred_mesh_name)
    #            d1_psnr = cal_d1_psnr(gt_meshPath, pred_meshPath)
    #            if job == job_list[-1]:
    #                f_log.write(str(d1_psnr) + '\n')
    #            else:
    #                f_log.write(str(d1_psnr) + '\t')
    #            print('\t\t' + str(d1_psnr))
    #f_log.close()

