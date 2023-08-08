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

def calculate_gaussian_curvature(mesh):
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # Load the mesh
    mesh = trimesh.load(mesh)
    # Compute the Gaussian curvature for each vertex
    points, radius = trimesh.sample.sample_surface(mesh, count=1000)  # Sample points on the mesh surface
    gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, points, radius)
    # Normalize the Gaussian curvature values to a range between 0 and 1
    min_curvature = np.min(gaussian_curvature)
    max_curvature = np.max(gaussian_curvature)
    normalized_curvature = (gaussian_curvature - min_curvature) / (max_curvature - min_curvature)
    # Assign a color to each vertex based on its normalized Gaussian curvature value
    colors = cm.coolwarm(normalized_curvature)
    # Visualize the mesh with the assigned colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color=colors[:, :3])
    plt.show()

def calculate_distance(point_p, point_a, point_b):
    # Convert points to PyTorch tensors
    import torch
    p = torch.tensor(point_p, dtype=torch.float32)
    a = torch.tensor(point_a, dtype=torch.float32)
    b = torch.tensor(point_b, dtype=torch.float32)
    # Calculate vectors
    ab = b - a
    ap = p - a
    # Calculate cross product
    cross_product = torch.cross(ab, ap)
    # Calculate magnitudes
    magnitude = torch.norm(cross_product)
    print(magnitude)
    magnitude_ab = torch.norm(ab)
    print(magnitude_ab)
    # Calculate distance
    distance = magnitude / magnitude_ab
    return distance
def main():
    jobs_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/'
    get_train_loss(jobs_dir)


if __name__ == '__main__':
    # main()
    calculate_gaussian_curvature('/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/bob_f600_ns3_nm20/subd0/01.obj')



