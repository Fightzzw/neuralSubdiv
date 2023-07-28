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
    print('start bat_call_mm!')
    numSubd = 3
    max_file_num = 50
    num_split = 10
    fill_num = len(str(max_file_num))
    output_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'

    mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    mesh_list = ['Camel_f1000_ns3_nm50', 'Dog_f1000_ns3_nm50', 'Lion_f1000_ns3_nm50', 'Panda_f1000_ns3_nm50']

  
    #mesh_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/'
    #mesh_list = ['bob_f600_ns3_nm20']

    short_job_list = ['b8_anchor', 'b8_hfNorm', 'b8_oddV_MfV']
    job_list = [output_root_dir+'/'+job+'/test_e2000/' for job in short_job_list]

    # short_job_list = ['b8_hfNorm', ]
    # job_list = [output_root_dir + '/' + job + '/test_e1000/' for job in short_job_list]

    p2point_file_path = output_root_dir + '/b8_mm_p2point_number_e2000.txt'
    p2plane_file_path = output_root_dir + '/b8_mm_p2plane_number_e2000.txt'
    p2point_log = open(p2point_file_path, 'w')
    p2point_log.write('mesh_dir\tpred_mesh_name')

    p2plane_log = open(p2plane_file_path, 'w')
    p2plane_log.write('mesh_dir\tpred_mesh_name')
    for job in short_job_list:
        p2point_log.write('\t'+job)
        p2plane_log.write('\t'+job)
        
    p2point_log.write('\n')
    p2plane_log.write('\n')
    
    for mesh_name in mesh_list:
       # p2point_log.write(mesh_name + '\n')
       print(mesh_name)
       for i in range(1, max_file_num, num_split):
           gt_mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd3')
           gt_meshPath = os.path.join(gt_mesh_dir, str(i).zfill(fill_num) + '.obj')
           pred_mesh_name = str(i).zfill(fill_num) + '_subd' + str(numSubd) + '.obj'
           # print('\t'+pred_mesh_name)
           p2point_log.write(mesh_name+'\t'+pred_mesh_name + '\t')
           p2plane_log.write(mesh_name+'\t'+pred_mesh_name + '\t')
           for job in job_list:
           
               pred_meshPath = os.path.join(job, mesh_name, pred_mesh_name)
               p2point, p2plane = call_mm_compare(gt_meshPath, pred_meshPath)
               if job == job_list[-1]:
                   p2point_log.write(str(p2point) + '\n')
                   p2plane_log.write(str(p2plane) + '\n')
               else:
                   p2point_log.write(str(p2point) + '\t')
                   p2plane_log.write(str(p2plane) + '\t')
               # print('\t\t' + str(p2point)+ '\t' + str(p2plane))
    p2point_log.close()
    p2plane_log.close()

def bat_jian():
    import trimesh
    import numpy as np
    print('start bat_jian!')
    numSubd = 2
    max_file_num = 20
    num_split = 20
    fill_num = len(str(max_file_num))
    output_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/net_bob_f600_ns3_nm100/'

    # mesh_root_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'
    # mesh_list = ['Camel_f1000_ns3_nm50', 'Dog_f1000_ns3_nm50', 'Lion_f1000_ns3_nm50', 'Panda_f1000_ns3_nm50']

  
    mesh_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/'
    mesh_list = ['bob_f600_ns3_nm20']

    # short_job_list = ['b8_anchor', 'b8_hfNorm', 'b8_oddV_MfV']
    # job_list = [output_root_dir+'/'+job+'/test_e1500/' for job in short_job_list]

    short_job_list = ['b8_anchor', 'b8_hfNorm', 'b8_oddV_MfV']
    job_list = [output_root_dir + '/' + job  for job in short_job_list]


    p2point_file_path =output_root_dir + '/b8_jian_p2point_number_e2000.txt'
   
    p2point_log = open(p2point_file_path, 'w')
    p2point_log.write('mesh_dir\tpred_mesh_name')

   
    for job in short_job_list:
        p2point_log.write('\t'+job)
       
        
    p2point_log.write('\n')
    
    
    for mesh_name in mesh_list:
       # p2point_log.write(mesh_name + '\n')
       print(mesh_name)
       for i in range(1, max_file_num, num_split):
           gt_mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd' + str(numSubd))
           gt_meshPath = os.path.join(gt_mesh_dir, str(i).zfill(fill_num) + '.obj')
           pred_mesh_name ='0_subd' + str(numSubd) + '.obj'
           # print('\t'+pred_mesh_name)
           p2point_log.write(mesh_name+'\t'+pred_mesh_name + '\t')
           
           for job in job_list:
           
               pred_meshPath = os.path.join(job, pred_mesh_name)
               gt_mesh = trimesh.load(gt_meshPath).vertices
               pred_mesh = trimesh.load(pred_meshPath).vertices

               p2point = np.mean(np.linalg.norm(gt_mesh - pred_mesh))
               print(p2point)
               if job == job_list[-1]:
                   p2point_log.write(str(p2point) + '\n')
                  
               else:
                   p2point_log.write(str(p2point) + '\t')
                   
    p2point_log.close()
   


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






if __name__ == '__main__':
    bat_jian()
    # bat_call_mm()
    