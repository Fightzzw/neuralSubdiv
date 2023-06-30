import argparse
import os.path
import numpy as np
import xlwt
import time
import xlrd
import math
import re


def read_vmesh_enc_log(enc_log_path):
    with open(enc_log_path, 'r') as file:
        lines = file.readlines()
        processing_time = 0
        base_mesh_bitrate = 0
        disp_bitrate = 0
        total_bitrate = 0
        for line in lines:
            if "Sequence processing time" in line:
                match = re.search(r"-?\d+\.?\d*", line)
                # 如果匹配到了数字，打印出来
                if match:
                    processing_time = float(match.group())
                    # print(processing_time)
                # 如果没有匹配到数字，打印出错误信息
                else:
                    print("No number found in line.")
                continue
            if "Sequence meshes bitrate" in line:
                match = re.search(r"-?\d+\.?\d*", line)
                # 如果匹配到了数字，打印出来
                if match:
                    base_mesh_bitrate = float(match.group())
                    # print(base_mesh_bitrate)
                # 如果没有匹配到数字，打印出错误信息
                else:
                    print("No number found in line.")
                continue
            if "Sequence displacements bitrate" in line:
                match = re.search(r"-?\d+\.?\d*", line)
                # 如果匹配到了数字，打印出来
                if match:
                    disp_bitrate = float(match.group())
                    # print(disp_bitrate)
                # 如果没有匹配到数字，打印出错误信息
                else:
                    print("No number found in line.")
                continue
            if "Sequence total bitrate" in line:
                match = re.search(r"-?\d+\.?\d*", line)
                # 如果匹配到了数字，打印出来
                if match:
                    total_bitrate = float(match.group())
                    # print(total_bitrate)
                # 如果没有匹配到数字，打印出错误信息
                else:
                    print("No number found in line.")
                continue
        print([base_mesh_bitrate, disp_bitrate, total_bitrate, processing_time])
    return [base_mesh_bitrate, disp_bitrate, total_bitrate, processing_time]

def read_vmesh_dec_log(dec_log_path):
    with open(dec_log_path, 'r') as file:
        lines = file.readlines()
        dec_processing_time = 0
        for line in lines:
            # if "Sequence processing time" in line:
            #     match = re.search(r"-?\d+\.?\d*", line)
            #     # 如果匹配到了数字，打印出来
            #     if match:
            #         dec_processing_time = float(match.group())
            #         # print(processing_time)
            #     # 如果没有匹配到数字，打印出错误信息
            #     else:
            #         print("No number found in line.")
            #     continue
            if 'gof decoded: frameCount' in line:
                # dec_processing_time = line.split(" ")[-2]
                match = re.search(r"-?\d+\.?\d*", line.split(" ")[-3])
                if match:
                    dec_processing_time = float(match.group())
                    # print(processing_time)
                # 如果没有匹配到数字，打印出错误信息
                else:
                    print("No number found in line.")
                continue
        print(dec_processing_time)
    return dec_processing_time

def bat_read_vmesh_log():

    work_dir = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/subd2_net_animal-3kk_ns3_nm50x6_norm/'
    #work_dir = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/subd2_mid/'
    output_file_path = work_dir+'/Camel+Lion_log_number.txt'
    # job_list = ['Owl_midpoint_kdDisp/subd3_LinearLifting', 'Owl_midpoint_kdDisp/subd3',
    #             'Owl_midpoint/subd3_LinearLifting', 'Owl_midpoint/subd3', 'Owl_neuralsubdiv/subd3']
    # job_list = ['Owl_neuralsubdiv/subd3_Owl_f2987']
    # job_list = ['elephant_neuralsubdiv/subd3_f888/', 'elephant_neuralsubdiv/subd3_f1776/',
    #             'elephant_midpoint/subd3_linearlifting/', 'elephant_midpoint/subd3/']
    job_list = ['Camel', 'Lion']
    with open(output_file_path, 'w') as f:

        for job in job_list:
            f.write(job + '\n')
            f.write('rate_point\tbase_mesh_bitrate\tdisp_bitrate\ttotal_bitrate\tenc_time\tdec_time\n')
            for i in range(4):
                enc_log_path = work_dir + job + '/enc/r' + str(i + 1) + '/enc_log.txt'
                dec_log_path = work_dir + job + '/dec/r' + str(i + 1) + '/dec_log.txt'
                enc_list = read_vmesh_enc_log(enc_log_path)
                dec_time = read_vmesh_dec_log(dec_log_path)
                f.write('r' + str(i + 1) + '\t')
                for num in enc_list:
                    f.write(str(num) + '\t')
                f.write(str(dec_time) + '\n')

if __name__=='__main__':
    # bat_read_vmesh_log()
    job_list = ['ori_nv']
    # job_list = ['hf_norm', 'hf_norm_12_pos+fea', 'hf_norm_12x34_pos+fea','odd','odd_hiddenLayer4', 'odd_hiddenLayer4_dout3']
    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'
    import os
    for job in job_list:
        print(job)
        os.makedirs(work_dir+'/log_bijiao/'+job, exist_ok=True)
        job_log_path = work_dir+'/'+job+'/log/'
        file = os.listdir(job_log_path)
        for f in file:
            cmd = 'cp '+job_log_path+f+' '+work_dir+'/log_bijiao/'+job+'/'+f
            os.system(cmd)
