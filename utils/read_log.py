import argparse
import os.path
import numpy as np
import xlwt
import time
import xlrd
import math
import re

def extract_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders
def read_metric_log():
    def match(file):
        import re
        log = open(file, 'r').read()
        average_line = re.findall(r'average.*', log)[0]
        # average	1.9850645e-06	65.3476378	68.85427075555555, 用正则表达式从该语句匹配数字
        # pattern = re.compile(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
        # numbers = pattern.findall(average_line)
        numbers = average_line.split('\t')[1:]
        return numbers

    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    cross_list = ['76947_sf_f1000_ns3_nm10',  '203289_sf_f1000_ns3_nm10',
                  '70558_sf_f1000_ns3_nm10', '1717684_sf_f1000_ns3_nm10']

    job_list = ['b1_anchor', 'b1_hfNorm', 'b1_oddV_MfV', 'b1_rnn', 'b1_rnn_dout16','b1_rnn_dout48',
                'b1_rnn_ori','b1_chamfer','b1_hfNorm_MfV_rnn']
    for folder in cross_list:
        print('current job folder is: ', folder)
        total_result = os.path.join(work_dir, folder, 'test_e3000_result.txt')
        f = open(total_result, 'w')
        f.write(folder + '\n')
        f.write('job_name\tmse\tp2point\tp2plane\n')
        cur_folder = os.path.join(work_dir, folder)
        # list the directory in cur_folder
        # job_list = extract_folders(cur_folder)
        for job in job_list:
            cur_work_dir = os.path.join(work_dir, folder, job)
            print('current job is: ', job)
            test_dir = os.path.join(cur_work_dir, 'test_e3000')
            f.write(job + '\t')
            root_numbers = match(os.path.join(test_dir, 'metric.txt'))
            for num in root_numbers:
                f.write(num + '\t')
            f.write('\n')
        f.close()

def bat_read_metric_log():
    def match(file):
        import re
        log = open(file, 'r').read()
        average_line = re.findall(r'average.*', log)[0]
        # average	1.9850645e-06	65.3476378	68.85427075555555, 用正则表达式从该语句匹配数字
        # pattern = re.compile(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
        # numbers = pattern.findall(average_line)
        numbers = average_line.split('\t')[1:]
        return numbers

    work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    cross_list = ['76947_sf_f1000_ns3_nm10', '70558_sf_f1000_ns3_nm10', '203289_sf_f1000_ns3_nm10',
                  '1717684_sf_f1000_ns3_nm10']

    job_list = ['b1_anchor', 'b1_hfNorm', 'b1_oddV_MfV', 'b1_rnn']
    for folder in cross_list:
        print('current job folder is: ', folder)
        total_result = os.path.join(work_dir, folder, 'test_e3000_total_result.txt')
        f = open(total_result, 'w')
        f.write('mesh_name\tmse\tp2point\tp2plane\n')
        for job in job_list:
            cur_work_dir = os.path.join(work_dir, folder, job)
            print('current job is: ', job)
            test_dir = os.path.join(cur_work_dir, 'test_e3000')
            f.write(job + '\n')
            root_numbers = match(os.path.join(test_dir, 'metric.txt'))
            numbers_direct = {folder: root_numbers}
            for test_folder in cross_list:
                if test_folder == folder:
                    continue
                test_numbers = match(os.path.join(test_dir, test_folder, 'metric.txt'))
                numbers_direct[test_folder] = test_numbers
            for key in cross_list:
                f.write(key + '\t')
                for num in numbers_direct[key]:
                    f.write(num + '\t')
                f.write('\n')
        f.close()

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
    read_metric_log()
    # job_list = ['ori_nv']
    # # job_list = ['hf_norm', 'hf_norm_12_pos+fea', 'hf_norm_12x34_pos+fea','odd','odd_hiddenLayer4', 'odd_hiddenLayer4_dout3']
    # work_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/animal6_f1000_ns3_nm50+5/'
    # import os
    # for job in job_list:
    #     print(job)
    #     os.makedirs(work_dir+'/log_bijiao/'+job, exist_ok=True)
    #     job_log_path = work_dir+'/'+job+'/log/'
    #     file = os.listdir(job_log_path)
    #     for f in file:
    #         cmd = 'cp '+job_log_path+f+' '+work_dir+'/log_bijiao/'+job+'/'+f
    #         os.system(cmd)
