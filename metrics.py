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
    bat_call_mm()
    print('finish bat_cal_vmesh_d1psnr ')

