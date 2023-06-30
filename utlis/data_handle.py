import os
import numpy as np
import xlwt


def to_excel(data, out_path):

    """
    Inputs:
    data: a dict like: {'sheet_name': data}, data is a 2 dimension list
    out_path: output path of an Excel
    """

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    for key, value in data.items():
        sheet = book.add_sheet(key, cell_overwrite_ok=True)
        for i in range(len(value)):
            for j in range(len(value[0])):
                sheet.write(i, j, value[i][j])

    book.save(out_path)
    print('成功将数据输出到：' + out_path)


def get_train_loss(jobs_dir):
    subFolders = os.listdir(jobs_dir)
    subFolders = [x for x in subFolders if not (x.startswith('.'))]

    jobs_loss_matrix = []  # 2维的
    top_title = ['jobID', 'train_min_loss', 'valid_min_loss']
    jobs_loss_matrix.append(top_title)

    for folders in subFolders:
        jobs = []
        trainloss_filepath = os.path.join(jobs_dir, folders + '/train_loss.txt')
        validloss_filepath = os.path.join(jobs_dir, folders + '/valid_loss.txt')
        if os.path.exists(trainloss_filepath) & os.path.exists(validloss_filepath):
            trainloss = np.genfromtxt(trainloss_filepath, dtype=float)
            validloss = np.genfromtxt(validloss_filepath, dtype=float)
        else:
            print('Warning: ' + trainloss_filepath + ' is not exist! Skip it.')
            print('Warning: ' + validloss_filepath + ' is not exist! Skip it.')
            continue

        jobs = [folders, trainloss.min(), validloss.min()]
        jobs_loss_matrix.append(jobs)

    data = {'jobs_min_loss': jobs_loss_matrix}
    for key, value in data.items():
        print(key)
        for i in range(len(value)):
            print('\t', value[i])

    excel_path = os.path.join(jobs_dir, 'min_loss.xls')
    to_excel(data, excel_path)


def main():
    jobs_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    get_train_loss(jobs_dir)


if __name__ == '__main__':
    main()
