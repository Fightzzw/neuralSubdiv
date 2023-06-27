import open3d as o3d
import argparse
import os.path
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import xlwt
import time
import xlrd
import math


def bat_mesh_subdivide():
    parser = argparse.ArgumentParser(description='mesh subdivide help document，同时执行loop和midpoint')
    parser.add_argument('-i', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会subdivide')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')
    parser.add_argument('-r', '--iter', type=int, help='划分次数', default=3)

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = ['elephant_overfit_f1000.obj']
    # # Iterate directory
    # for path in os.listdir(args.input_mesh_dir):
    #     # check if current path is a obj file
    #     if os.path.splitext(path)[-1] == ".obj":
    #         mesh_file_list.append(path)
    print('The following mesh will be subdivided.')
    print(mesh_file_list)
    for filename in mesh_file_list:
        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh = o3d.io.read_triangle_mesh(file_path)
        base_mesh_filename = filename[:-4]
        print('Base mesh ' + filename + ' has [vertices: %d], [faces: %d]' % (
            len(base_mesh.vertices), len(base_mesh.triangles)))
        for i in range(args.iter):
            print('\t subdivide iteration process:', i)
            loop_mesh = base_mesh.subdivide_loop(i + 1)
            midpoint_mesh = base_mesh.subdivide_midpoint(i + 1)
            loop_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_loop' + str(i) + '.obj'
            midpoint_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_midpoint' + str(i) + '.obj'
            o3d.io.write_triangle_mesh(loop_mesh_out_path, loop_mesh)
            o3d.io.write_triangle_mesh(midpoint_mesh_out_path, midpoint_mesh)


def mesh_subdivide(input_path, output_path, iters=2, subdivide_method='midpoint'):
    base_mesh = o3d.io.read_triangle_mesh(input_path)
    filename = os.path.basename(input_path)
    original_faces_num = len(base_mesh.triangles)
    print('Base mesh: ' + filename + ' has [vertices: %d], [faces: %d]' % (
        len(base_mesh.vertices), original_faces_num))

    output_mesh_path = ''

    if subdivide_method == 'midpoint':
        subdiv_mesh = base_mesh.subdivide_midpoint(iters)
        output_mesh_name = filename[:-4] + '_mid' + str(iters) + '.obj'
        output_mesh_path = os.path.join(output_path, output_mesh_name)
        o3d.io.write_triangle_mesh(output_mesh_path, subdiv_mesh)
    elif subdivide_method == 'loop':
        subdiv_mesh = base_mesh.subdivide_loop(iters)
        output_mesh_name = filename[:-4] + '_loop' + str(iters) + '.obj'
        output_mesh_path = os.path.join(output_path, output_mesh_name)
        o3d.io.write_triangle_mesh(output_mesh_path, subdiv_mesh)
    elif subdivide_method == 'butterfly':
        output_mesh_name = filename[:-4] + '_btf' + str(iters) + '.obj'
        output_mesh_path = os.path.join(output_path, output_mesh_name)
        btf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide ' + \
                  input_path + ' ' + output_mesh_path + ' butterfly ' + str(iters)
        os.system(btf_cmd)
    elif subdivide_method == 'modified_butterfly':
        output_mesh_name = filename[:-4] + '_mbtf' + str(iters) + '.obj'
        output_mesh_path = os.path.join(output_path, output_mesh_name)
        mbtf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide ' + \
                   input_path + ' ' + output_mesh_path + ' modified_butterfly ' + str(iters)
        os.system(mbtf_cmd)
    elif subdivide_method == 'neural_subdiv':
        output_mesh_name = filename[:-4] + '_nrsd' + str(iters) + '.obj'
        output_mesh_path = os.path.join(output_path, output_mesh_name)
        nrsd_cmd = '/work/Users/zhuzhiwei/anaconda3/envs/mesh_subdiv/bin/python ' \
                   '/work/Users/zhuzhiwei/neuralSubdiv-master/zzwtest.py ' \
                   '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/net_cartoon_elephant/ ' + \
                   input_path + ' ' + output_mesh_path + ' ' + str(iters)
        os.system(nrsd_cmd)
        mesh_scale(input_path, output_mesh_path)
    return output_mesh_path


def bat_mesh_downsample():
    parser = argparse.ArgumentParser(description='mesh downsample help document')
    parser.add_argument('-i', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会downsample')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')
    parser.add_argument('-f', '--faces_num', type=int, help='目标面数', default=5000)

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is a obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    print('The following mesh will be downsampled.')
    print(mesh_file_list)
    for filename in mesh_file_list:
        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh = o3d.io.read_triangle_mesh(file_path)
        base_mesh_filename = filename[:-4]
        print('Base mesh ' + filename + ' has [vertices: %d], [faces: %d]' % (
            len(base_mesh.vertices), len(base_mesh.triangles)))
        downsample_mesh = base_mesh.simplify_quadric_decimation(args.faces_num)
        downsample_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_f' + str(args.faces_num) + '.obj'
        o3d.io.write_triangle_mesh(downsample_mesh_out_path, downsample_mesh)


def mesh_downsample(input_path, output_path, iters):
    base_mesh = o3d.io.read_triangle_mesh(input_path)
    filename = os.path.basename(input_path)
    original_faces_num = len(base_mesh.triangles)
    print('Base mesh: ' + filename + ' has [vertices: %d], [faces: %d]' % (
        len(base_mesh.vertices), original_faces_num))
    downsample_faces_num = int(original_faces_num / pow(4, iters))  # 抹去小数
    downsample_mesh = base_mesh.simplify_quadric_decimation(downsample_faces_num)
    print('downsample mesh: ' + filename + ' has [vertices: %d], [faces: %d]' % (
        len(downsample_mesh.vertices), len(downsample_mesh.triangles)))
    o3d.io.write_triangle_mesh(output_path, downsample_mesh)


def mesh_normalize():
    parser = argparse.ArgumentParser(description='mesh normalize help document')
    parser.add_argument('-i', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会normalize')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is a obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    print('The following mesh will be downsampled.')
    print(mesh_file_list)
    for filename in mesh_file_list:
        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh = o3d.io.read_triangle_mesh(file_path)
        base_mesh_filename = filename[:-4]
        print('Base mesh ' + filename + ' has [vertices: %d], [faces: %d]' % (
            len(base_mesh.vertices), len(base_mesh.triangles)))
        min_bound = base_mesh.get_min_bound()
        max_bound = base_mesh.get_max_bound()
        scale_len = (max_bound - min_bound).max()

        vertices = base_mesh.vertices
        for v in range(len(vertices)):
            vertices[v] -= min_bound
            vertices[v] /= scale_len
        base_mesh.vertices = vertices

        normalize_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_normalize' + '.obj'
        o3d.io.write_triangle_mesh(normalize_mesh_out_path, base_mesh)


def bat_mesh_scale():
    parser = argparse.ArgumentParser(description='mesh scale help document')
    parser.add_argument('-i1', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会scale')
    parser.add_argument('-i2', '--input_reference_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会scale')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is a obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    print('The following mesh will be scaled.')
    print(mesh_file_list)
    for filename in mesh_file_list:
        # refer_file_name = filename.split("_")[0] + '_f1000.obj'
        refer_file_name = 'elephant_overfit_f1000.obj'
        refer_file_path = os.path.join(args.input_reference_mesh_dir, refer_file_name)
        refer_mesh = o3d.io.read_triangle_mesh(refer_file_path)

        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh = o3d.io.read_triangle_mesh(file_path)
        # base_mesh_filename = filename[:-4]

        min_bound = refer_mesh.get_min_bound()
        max_bound = refer_mesh.get_max_bound()
        scale_len = (max_bound - min_bound).max()

        vertices = base_mesh.vertices
        for v in range(len(vertices)):
            vertices[v] *= scale_len
            vertices[v] += min_bound
        base_mesh.vertices = vertices

        scale_mesh_out_path = args.output_dir + '/' + filename
        o3d.io.write_triangle_mesh(scale_mesh_out_path, base_mesh)


def mesh_scale(refer_file_path, file_path):
    refer_mesh = o3d.io.read_triangle_mesh(refer_file_path)

    base_mesh = o3d.io.read_triangle_mesh(file_path)
    # base_mesh_filename = filename[:-4]

    min_bound = refer_mesh.get_min_bound()
    max_bound = refer_mesh.get_max_bound()
    scale_len = (max_bound - min_bound).max()

    vertices = base_mesh.vertices
    for v in range(len(vertices)):
        vertices[v] *= scale_len
        vertices[v] += min_bound
    base_mesh.vertices = vertices

    scale_mesh_out_path = file_path
    o3d.io.write_triangle_mesh(scale_mesh_out_path, base_mesh)


def mesh_geometry_voxelized():
    parser = argparse.ArgumentParser(description='mesh geometry voxelized help document')
    parser.add_argument('-i', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会geometry voxelized')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')
    parser.add_argument('-b', '--bitDepth', type=int, help='voxel几何比特位宽')

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is a obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    print('The following mesh will be geometry voxelized.')
    mesh_file_list = ['elephant_overfit.obj']
    print(mesh_file_list)
    for filename in mesh_file_list:
        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh = o3d.io.read_triangle_mesh(file_path)
        base_mesh_filename = filename[:-4]
        print('Base mesh ' + filename + ' has [vertices: %d], [faces: %d]' % (
            len(base_mesh.vertices), len(base_mesh.triangles)))
        min_bound = base_mesh.get_min_bound()
        max_bound = base_mesh.get_max_bound()
        max_len = (max_bound - min_bound).max()
        scale_len = pow(2, args.bitDepth) / max_len

        vertices = base_mesh.vertices
        for v in range(len(vertices)):
            vertices[v] -= min_bound
            vertices[v] = np.round(vertices[v] * scale_len)
        base_mesh.vertices = vertices

        # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(base_mesh, 0.01)
        out_path = args.output_dir + '/' + base_mesh_filename + '_voxelized' + str(args.bitDepth) + '.obj'
        o3d.io.write_triangle_mesh(out_path, base_mesh)


def mesh_metric():
    parser = argparse.ArgumentParser(description='mesh scale help document')
    parser.add_argument('-i1', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会metric')
    parser.add_argument('-i2', '--input_reference_mesh_dir', type=str, help='参考mesh的目录')
    parser.add_argument('-r', '--iter', type=int, help='划分次数', default=3)
    parser.add_argument('-v', '--vertex_num', type=int, help='采样点数', default=200000)

    args = parser.parse_args()

    col_title = ['modelID']
    for i in range(args.iter):
        # col_title.append('midpoint'+str(i))
        # col_title.append('loop' + str(i))
        # col_title.append('subd' + str(i))
        # col_title.append('btf' + str(i))
        col_title.append('mbtf' + str(i))

    D1_dataMatrix = [col_title]
    D2_dataMatrix = [col_title]
    H_dataMatrix = [col_title]
    absD_dataMatrix = [col_title]
    # if os.path.isdir(args.output_dir):
    #     print('文件夹已存在')
    #     pass
    # else:
    #     os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_reference_mesh_dir):
        # check if current path is an obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(os.path.splitext(path)[0])
    print('The following original mesh will be metric.')
    print(mesh_file_list)

    for midx, filename in enumerate(mesh_file_list):
        print('正在处理： ' + filename)
        D1_midx = []
        D2_midx = []
        H_midx = []
        absD_midx = []  # 相较于D1是距离的平方，这使用的是绝对值
        D1_midx.append(filename)
        D2_midx.append(filename)
        H_midx.append(filename)
        absD_midx.append(filename)

        refer_file_name = filename + '.obj'
        refer_file_path = os.path.join(args.input_reference_mesh_dir, refer_file_name)
        # gt_mesh = o3d.io.read_triangle_mesh(refer_file_path)
        gt_mesh = trimesh.load(refer_file_path)
        # gt_points, gt_indices = gt_mesh.sample(200000, return_index=True)
        gt_points, gt_indices = trimesh.sample.sample_surface_even(gt_mesh, args.vertex_num)
        gt_points = gt_points.astype(np.float32)
        gt_normals = gt_mesh.face_normals[gt_indices]

        gt_kdtree = KDTree(gt_points)

        for methodName in col_title[1:]:
            pred_file_name = filename + '_f1000_' + methodName + '.obj'
            print('\t正在与 ' + pred_file_name + ' 计算距离')

            pred_file_path = os.path.join(args.input_mesh_dir, pred_file_name)
            # pred_mesh = o3d.io.read_triangle_mesh(file_path)
            pred_mesh = trimesh.load(pred_file_path)

            # trimesh/base.py, sample.py, sample_on_surface
            # pred_points, pred_indices = pred_mesh.sample(200000, return_index=True)
            pred_points, pred_indices = trimesh.sample.sample_surface_even(pred_mesh, args.vertex_num)
            pred_points = pred_points.astype(np.float32)
            pred_normals = pred_mesh.face_normals[pred_indices]

            dist_p2g, indices_p2g = gt_kdtree.query(pred_points)

            pred_kdtree = KDTree(pred_points)
            dist_g2p, indices_g2p = pred_kdtree.query(gt_points)

            # point to point distance
            D1 = max(np.mean(dist_p2g ** 2), np.mean(dist_g2p ** 2))

            # point to point average displacement
            absD = max(np.mean(np.abs(dist_p2g)), np.mean(np.abs(dist_g2p)))

            # Hausdorff distance
            H = max(np.max(dist_p2g), np.max(dist_g2p))

            # point to plane distance
            gt_points = np.array(gt_points)
            pred_points = np.array(pred_points)
            vector_p2g = pred_points - gt_points[indices_p2g]
            normals_p2g = gt_normals[indices_p2g]
            n_dist_p2g = np.abs(np.sum(vector_p2g * normals_p2g, axis=1)) / np.linalg.norm(normals_p2g, axis=1)

            vector_g2p = gt_points - pred_points[indices_g2p]
            normals_g2p = pred_normals[indices_g2p]
            n_dist_g2p = np.abs(np.sum(vector_g2p * normals_g2p, axis=1)) / np.linalg.norm(normals_g2p, axis=1)
            D2 = max(np.mean(n_dist_p2g ** 2), np.mean(n_dist_g2p ** 2))

            D1_midx.append(D1)
            D2_midx.append(D2)
            H_midx.append(H)
            absD_midx.append(absD)
        # 1个模型处理完毕
        D1_dataMatrix.append(D1_midx)
        D2_dataMatrix.append(D2_midx)
        H_dataMatrix.append(H_midx)
        absD_dataMatrix.append(absD_midx)

    # 所有模型都处理完毕,写入excel
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheetD1 = book.add_sheet('D1', cell_overwrite_ok=True)
    sheetD2 = book.add_sheet('D2', cell_overwrite_ok=True)
    sheetH = book.add_sheet('H', cell_overwrite_ok=True)
    sheetabsD = book.add_sheet('absD', cell_overwrite_ok=True)
    book_name = args.input_reference_mesh_dir.split('/')[-1] + '_metric_sample_surface_even_v' + str(
        args.vertex_num) + '_6.xls'
    book_path = os.path.join(args.input_mesh_dir, book_name)
    print('所有模型比较完毕，开始将数据写入文件')
    for i in range(len(D1_dataMatrix)):
        for j in range(len(col_title)):
            sheetD1.write(i, j, D1_dataMatrix[i][j])
            sheetD2.write(i, j, D2_dataMatrix[i][j])
            sheetH.write(i, j, H_dataMatrix[i][j])
            sheetabsD.write(i, j, absD_dataMatrix[i][j])
    book.save(book_path)
    print('成功将数据输出到：' + book_path)


def mesh_metric2():
    parser = argparse.ArgumentParser(description='mesh scale help document')
    parser.add_argument('-i1', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会metric')
    parser.add_argument('-i2', '--input_reference_mesh_dir', type=str, help='参考mesh的目录')
    parser.add_argument('-r', '--iter', type=int, help='划分次数', default=3)
    parser.add_argument('-v', '--vertex_num', type=int, help='采样点数', default=200000)

    args = parser.parse_args()
    col_title = ['model_ID', 'net_ID']
    for i in range(args.iter):
        col_title.append('subd' + str(i))
    net_list = ['net_Horse_f1000_ns2_nm200', 'net_Horse_f1000_ns3_nm200', 'net_Horse_f1000_ns3_nm200_32*3',
                'net_Horse_f1000_ns3_nm200_64*2']

    D1_dataMatrix = [col_title]
    D2_dataMatrix = [col_title]
    H_dataMatrix = [col_title]
    absD_dataMatrix = [col_title]
    # if os.path.isdir(args.output_dir):
    #     print('文件夹已存在')
    #     pass
    # else:
    #     os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_reference_mesh_dir):
        # check if current path is an obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(os.path.splitext(path)[0])
    print('The following original mesh will be metric.')
    mesh_file_list = ['Horse']
    print(mesh_file_list)
    for midx, filename in enumerate(mesh_file_list):
        # sheetD1.write(midx+1, 0, filename)
        # sheetD2.write(midx+1, 0, filename)
        # sheetH.write(midx+1, 0, filename)
        print('正在处理： ' + filename)

        refer_file_name = filename + '.obj'
        refer_file_path = os.path.join(args.input_reference_mesh_dir, refer_file_name)
        # gt_mesh = o3d.io.read_triangle_mesh(refer_file_path)
        gt_mesh = trimesh.load(refer_file_path)
        # gt_points, gt_indices = gt_mesh.sample(200000, return_index=True)
        gt_points, gt_indices = trimesh.sample.sample_surface_even(gt_mesh, args.vertex_num)
        gt_points = gt_points.astype(np.float32)
        gt_normals = gt_mesh.face_normals[gt_indices]

        gt_kdtree = KDTree(gt_points)

        for net in net_list:
            D1_midx = []
            D2_midx = []
            H_midx = []
            absD_midx = []  # 相较于D1是距离的平方，这使用的是绝对值
            D1_midx.append(filename)
            D2_midx.append(filename)
            H_midx.append(filename)
            absD_midx.append(filename)

            D1_midx.append(net)
            D2_midx.append(net)
            H_midx.append(net)
            absD_midx.append(net)

            pred_dir = os.path.join(args.input_mesh_dir, net)

            for methodName in col_title[2:]:
                pred_file_name = filename + '_f1000_' + methodName + '.obj'
                pred_file_path = os.path.join(pred_dir, pred_file_name)

                print('\t正在与 ' + pred_file_path + ' 计算距离')

                pred_mesh = trimesh.load(pred_file_path)

                # trimesh/base.py, sample.py, sample_on_surface
                # pred_points, pred_indices = pred_mesh.sample(200000, return_index=True)
                pred_points, pred_indices = trimesh.sample.sample_surface_even(pred_mesh, args.vertex_num)
                pred_points = pred_points.astype(np.float32)
                pred_normals = pred_mesh.face_normals[pred_indices]

                dist_p2g, indices_p2g = gt_kdtree.query(pred_points)

                pred_kdtree = KDTree(pred_points)
                dist_g2p, indices_g2p = pred_kdtree.query(gt_points)

                # point to point distance
                D1 = max(np.mean(dist_p2g ** 2), np.mean(dist_g2p ** 2))

                # point to point average displacement
                absD = max(np.mean(np.abs(dist_p2g)), np.mean(np.abs(dist_g2p)))

                # Hausdorff distance
                H = max(np.max(dist_p2g), np.max(dist_g2p))

                # point to plane distance
                gt_points = np.array(gt_points)
                pred_points = np.array(pred_points)
                vector_p2g = pred_points - gt_points[indices_p2g]
                normals_p2g = gt_normals[indices_p2g]
                n_dist_p2g = np.abs(np.sum(vector_p2g * normals_p2g, axis=1)) / np.linalg.norm(normals_p2g, axis=1)

                vector_g2p = gt_points - pred_points[indices_g2p]
                normals_g2p = pred_normals[indices_g2p]
                n_dist_g2p = np.abs(np.sum(vector_g2p * normals_g2p, axis=1)) / np.linalg.norm(normals_g2p, axis=1)
                D2 = max(np.mean(n_dist_p2g ** 2), np.mean(n_dist_g2p ** 2))

                D1_midx.append(D1)
                D2_midx.append(D2)
                H_midx.append(H)
                absD_midx.append(absD)
            # 1个模型处理完毕
            D1_dataMatrix.append(D1_midx)
            D2_dataMatrix.append(D2_midx)
            H_dataMatrix.append(H_midx)
            absD_dataMatrix.append(absD_midx)

    # 所有模型都处理完毕,写入excel
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheetD1 = book.add_sheet('D1', cell_overwrite_ok=True)
    sheetD2 = book.add_sheet('D2', cell_overwrite_ok=True)
    sheetH = book.add_sheet('H', cell_overwrite_ok=True)
    sheetabsD = book.add_sheet('absD', cell_overwrite_ok=True)
    book_name = args.input_reference_mesh_dir.split('/')[-1] + '_metric_sample_surface_even_v' + str(
        args.vertex_num) + '_4net.xls'
    book_path = os.path.join(args.input_mesh_dir, book_name)
    print('所有模型比较完毕，开始将数据写入文件')
    for i in range(len(D1_dataMatrix)):
        for j in range(len(col_title)):
            sheetD1.write(i, j, D1_dataMatrix[i][j])
            sheetD2.write(i, j, D2_dataMatrix[i][j])
            sheetH.write(i, j, H_dataMatrix[i][j])
            sheetabsD.write(i, j, absD_dataMatrix[i][j])
    book.save(book_path)
    print('成功将数据输出到：' + book_path)


def mesh_subdivide_butterfly():
    parser = argparse.ArgumentParser(description='mesh subdivide help document，同时执行butterfly和modified butterfly')
    parser.add_argument('-i', '--input_mesh_dir', type=str, help='输入mesh的目录，里面的mesh全部会subdivide')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录的路径')
    parser.add_argument('-r', '--iter', type=int, help='划分次数', default=3)

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(args.output_dir)

    # list to store files
    mesh_file_list = []

    # # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is an obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    # mesh_file_list = ['Horse_f1000.obj']

    print('The following mesh will be subdivided.')
    print(mesh_file_list)
    for filename in mesh_file_list:
        file_path = os.path.join(args.input_mesh_dir, filename)
        base_mesh_filename = filename[:-4]
        print('current subdivided mesh is: ', filename)

        # btf_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_btf'
        mbtf_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_mbtf'

        # btf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide_new ' + file_path + ' ' + btf_mesh_out_path + ' butterfly ' + str(
        #    args.iter)
        mbtf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide_20230326 ' + file_path + ' ' + mbtf_mesh_out_path + ' modified_butterfly ' + str(
            args.iter)
        print('\t iter: ', args.iter)

        # t0 = time.time()
        # os.system(btf_cmd)
        t_btf = time.time()
        # print('\t\tbtf_time', t_btf - t0)
        os.system(mbtf_cmd)
        t_mbtf = time.time()
        print('\t\tmbtf_time', t_mbtf - t_btf)
        # for i in range(args.iter):
        #     loop_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_btf' + str(i) + '.obj'
        #     btf_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_btf' + str(i) + '.obj'
        #     mbtf_mesh_out_path = args.output_dir + '/' + base_mesh_filename + '_mbtf' + str(i) + '.obj'
        #
        #     loop_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide ' + file_path + ' ' + btf_mesh_out_path + ' butterfly ' + str(
        #         i + 1)
        #
        #     btf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide '+file_path + ' '+ btf_mesh_out_path + ' butterfly ' + str(i+1)
        #     mbtf_cmd = '/work/Users/zhuzhiwei/SubdivisionSurfaces-master/subdivide ' + file_path + ' ' + mbtf_mesh_out_path + ' modified_butterfly ' + str(
        #         i+1)
        #     print('\t iter: ', i)
        #
        #     t0 = time.time()
        #
        #     os.system(loop_cmd)
        #     t_loop = time.time()
        #     print('\t\tloop_time', t_loop-t0)
        #     os.system(btf_cmd)
        #     t_btf = time.time()
        #     print('\t\tbtf_time', t_btf - t_loop)
        #     os.system(mbtf_cmd)
        #     t_mbtf = time.time()
        #     print('\t\tmbtf_time', t_mbtf - t_btf)


def distance_metric(gt_mesh_path, pred_mesh_path, vertex_num=500000):
    # gt_points, gt_indices = gt_mesh.sample(200000, return_index=True)
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_points, gt_indices = trimesh.sample.sample_surface_even(gt_mesh, vertex_num)
    gt_points = gt_points.astype(np.float32)
    gt_normals = gt_mesh.face_normals[gt_indices]

    pred_mesh = trimesh.load(pred_mesh_path)
    pred_points, pred_indices = trimesh.sample.sample_surface_even(pred_mesh, vertex_num)
    pred_points = pred_points.astype(np.float32)
    pred_normals = pred_mesh.face_normals[pred_indices]

    gt_kdtree = KDTree(gt_points)
    dist_p2g, indices_p2g = gt_kdtree.query(pred_points)

    pred_kdtree = KDTree(pred_points)
    dist_g2p, indices_g2p = pred_kdtree.query(gt_points)

    # point to point distance
    D1 = max(np.mean(dist_p2g ** 2), np.mean(dist_g2p ** 2))

    # point to point average displacement
    absD = max(np.mean(np.abs(dist_p2g)), np.mean(np.abs(dist_g2p)))

    # Hausdorff distance
    H = max(np.max(dist_p2g), np.max(dist_g2p))

    # point to plane distance
    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)
    vector_p2g = pred_points - gt_points[indices_p2g]
    normals_p2g = gt_normals[indices_p2g]
    n_dist_p2g = np.abs(np.sum(vector_p2g * normals_p2g, axis=1)) / np.linalg.norm(normals_p2g, axis=1)

    vector_g2p = gt_points - pred_points[indices_g2p]
    normals_g2p = pred_normals[indices_g2p]
    n_dist_g2p = np.abs(np.sum(vector_g2p * normals_g2p, axis=1)) / np.linalg.norm(normals_g2p, axis=1)
    D2 = max(np.mean(n_dist_p2g ** 2), np.mean(n_dist_g2p ** 2))

    # return {'D1': D1, 'D2': D2, 'H': H, 'absD': absD}
    return [D1, D2, H, absD]


def cal_d1_psnr(gt_mesh_path, pred_mesh_path, vertex_num=500000):
    # gt_points, gt_indices = gt_mesh.sample(200000, return_index=True)
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_points, gt_indices = trimesh.sample.sample_surface_even(gt_mesh, vertex_num)
    gt_points = gt_points.astype(np.float32)
    gt_normals = gt_mesh.face_normals[gt_indices]

    pred_mesh = trimesh.load(pred_mesh_path)
    pred_points, pred_indices = trimesh.sample.sample_surface_even(pred_mesh, vertex_num)
    pred_points = pred_points.astype(np.float32)
    pred_normals = pred_mesh.face_normals[pred_indices]

    gt_kdtree = KDTree(gt_points)
    dist_p2g, indices_p2g = gt_kdtree.query(pred_points)

    pred_kdtree = KDTree(pred_points)
    dist_g2p, indices_g2p = pred_kdtree.query(gt_points)

    # point to point distance
    D1 = max(np.mean(dist_p2g ** 2), np.mean(dist_g2p ** 2))

    d1_psnr = d1_mse2psnr(D1)
    # print('d1_psnr', d1_psnr)
    return d1_psnr


def my_mkdirs(path):
    if os.path.isdir(path):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(path)


def mesh_compression():
    parser = argparse.ArgumentParser(description='mesh Handle help document')
    parser.add_argument('-i', '--input_mesh_dir', type=str,
                        default='/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/watertight5/',
                        help='输入mesh的目录路径，里面的mesh全部会执行操作')
    parser.add_argument('-ob', '--output_bin_dir', type=str,
                        default='/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/output_bin',
                        help='输出2进制文件目录的路径')
    parser.add_argument('-om', '--output_mesh_dir', type=str,
                        default='/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/output_mesh/',
                        help='解码得到的mesh输出目录的路径')
    parser.add_argument('-r', '--iter', type=int, help='降分超分次数', default=2)
    parser.add_argument('-v', '--vertex_num', type=int, help='采样点数', default=500000)

    args = parser.parse_args()

    # 建好输出目录
    or_output_bin_dir = os.path.join(args.output_bin_dir, 'original')  # 压缩原始分辨率mesh
    or_output_mesh_dir = os.path.join(args.output_mesh_dir, 'original')
    dr_output_bin_dir = os.path.join(args.output_bin_dir, 'downsample')  # 压缩降分辨率mesh
    dr_output_mesh_dir = os.path.join(args.output_mesh_dir, 'downsample')
    dr_base_mesh_dir = os.path.join(dr_output_mesh_dir, 'base_mesh')
    subdiv_mesh_path = os.path.join(dr_output_mesh_dir, 'subdiv_mesh')

    my_mkdirs(or_output_bin_dir)
    my_mkdirs(or_output_mesh_dir)
    my_mkdirs(dr_output_bin_dir)
    my_mkdirs(dr_output_mesh_dir)
    my_mkdirs(dr_base_mesh_dir)
    my_mkdirs(subdiv_mesh_path)

    mesh_subdivide_method = ['midpoint', 'loop', 'butterfly', 'modified_butterfly', 'neural_subdiv']

    # list to store files
    mesh_file_list = []

    # # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is an obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
    # mesh_file_list = ['Horse_f1000.obj']

    print('The following mesh will be Handled.')
    print(mesh_file_list)

    for filename in mesh_file_list:
        or_data_matrix = []
        dr_data_matrix = []
        or_col_title = ['level', 'enc_time', 'dec_time', 'bin_size', 'D1', 'D2', 'H', 'absD']
        dr_col_title1 = ['level', 'enc_time', 'dec_time', 'bin_size', 'down_time']
        metric_method = ['subdiv_time', 'D1', 'D2', 'H', 'absD']
        for method in mesh_subdivide_method:
            dr_col_title1 += [method] * len(metric_method)
        dr_col_title2 = ['level', 'enc_time', 'dec_time', 'bin_size', 'down_time'] + metric_method * len(
            mesh_subdivide_method)
        or_data_matrix.append(or_col_title)
        dr_data_matrix.append(dr_col_title1)
        dr_data_matrix.append(dr_col_title2)

        print('current mesh is: ', filename)
        input_file_path = os.path.join(args.input_mesh_dir, filename)
        base_filename = filename[:-4]

        dr_base_mesh_filename = base_filename + '_d' + str(args.iter) + '.obj'
        dr_base_mesh_path = os.path.join(dr_base_mesh_dir, dr_base_mesh_filename)

        dr_downsample_t0 = time.time()
        mesh_downsample(input_file_path, dr_base_mesh_path, args.iter)
        dr_downsample_time = time.time() - dr_downsample_t0
        print('\t\t降分辨率完成，耗时：', dr_downsample_time)

        for level in range(0, 11, 1):
            or_data = []
            dr_data = []

            # 原始分辨率压缩 original resolution
            print('\t对原始分辨率mesh操作：')
            or_output_bin_filename = base_filename + 'level_' + str(level) + '.drc'
            or_output_mesh_filename = base_filename + 'level_' + str(level) + '_rec.obj'
            or_output_bin_path = os.path.join(or_output_bin_dir, or_output_bin_filename)
            or_output_mesh_path = os.path.join(or_output_mesh_dir, or_output_mesh_filename)
            or_enc_cmd = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/app/draco_encoder-1.5.3 -i ' + input_file_path \
                         + ' -o ' + or_output_bin_path + ' -cl ' + str(level)
            or_enc_t0 = time.time()
            os.system(or_enc_cmd)
            or_enc_time = time.time() - or_enc_t0
            print('\t\t原始分辨率mesh编码完成，耗时：', or_enc_time)

            or_bin_filesize = os.path.getsize(or_output_bin_path)

            or_dec_cmd = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/app/draco_decoder-1.5.3 -i ' + or_output_bin_path + ' -o ' \
                         + or_output_mesh_path
            or_dec_t0 = time.time()
            os.system(or_dec_cmd)
            or_dec_time = time.time() - or_dec_t0
            print('\t\t原始分辨率mesh解码完成，耗时：', or_dec_time)

            # 客观结果测试
            or_metric = distance_metric(input_file_path, or_output_mesh_path, args.iter)
            print('\t\t原始分辨率mesh测试完成，结果：', or_metric)
            or_data.append(level)
            or_data.append(or_enc_time)
            or_data.append(or_dec_time)
            or_data.append(or_bin_filesize)
            or_data += or_metric

            # 降分辨率压缩 downsample resolution
            print('\t对降分辨率mesh操作：')

            dr_output_bin_filename = base_filename + '_d' + str(args.iter) + 'level_' + str(level) + '.drc'
            dr_output_mesh_filename = base_filename + '_d' + str(args.iter) + 'level_' + str(level) + '_rec.obj'

            dr_output_bin_path = os.path.join(dr_output_bin_dir, dr_output_bin_filename)
            dr_output_mesh_path = os.path.join(dr_output_mesh_dir, dr_output_mesh_filename)

            dr_enc_cmd = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/app/draco_encoder-1.5.3 -i ' \
                         + dr_base_mesh_path + ' -o ' + dr_output_bin_path + ' -cl ' + str(level)
            # print(dr_enc_cmd)
            dr_enc_t0 = time.time()
            os.system(dr_enc_cmd)
            dr_enc_time = time.time() - dr_enc_t0
            print('\t\t降分辨率mesh编码完成，耗时：', dr_enc_time)

            dr_bin_filesize = os.path.getsize(dr_output_bin_path)

            dr_dec_cmd = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/app/draco_decoder-1.5.3 -i ' + dr_output_bin_path + ' -o ' \
                         + dr_output_mesh_path
            # print(dr_dec_cmd)
            dr_dec_t0 = time.time()
            os.system(dr_dec_cmd)
            dr_dec_time = time.time() - dr_dec_t0
            print('\t\t降分辨率mesh解码完成，耗时：', dr_dec_time)

            dr_data.append(level)
            dr_data.append(dr_enc_time)
            dr_data.append(dr_dec_time)
            dr_data.append(dr_bin_filesize)
            dr_data.append(dr_downsample_time)

            for method in mesh_subdivide_method:
                subdiv_t0 = time.time()
                output_mesh_path = mesh_subdivide(dr_output_mesh_path, subdiv_mesh_path, args.iter, method)
                subdiv_time = time.time() - subdiv_t0
                dr_metric = distance_metric(input_file_path, output_mesh_path, args.iter)
                print('\t\t当前的超分方法是：%s， 耗时：%d，测试结果：' % (method, subdiv_time), dr_metric)
                dr_data.append(subdiv_time)
                dr_data += dr_metric

            # 将测试结果加入数据矩阵
            or_data_matrix.append(or_data)
            dr_data_matrix.append(dr_data)

        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        or_sheet = book.add_sheet('original', cell_overwrite_ok=True)
        dr_sheet = book.add_sheet('downsample', cell_overwrite_ok=True)

        book_name = base_filename + '_metric.xls'
        book_path = os.path.join(args.output_mesh_dir, book_name)
        print('所有模型比较完毕，开始将数据写入文件')

        for i in range(len(or_data_matrix)):
            for j in range(len(or_data_matrix[0])):
                or_sheet.write(i, j, or_data_matrix[i][j])

        for i in range(len(dr_data_matrix)):
            for j in range(len(dr_data_matrix[0])):
                dr_sheet.write(i, j, dr_data_matrix[i][j])

        book.save(book_path)
        print('成功将数据输出到：' + book_path)


def draw_fig(anchor_x, anchor_y, test_x, test_y, fig_path, title='Anchor vs Test'):
    import matplotlib.pyplot as plt
    # plt.style.use('science')
    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.title(title)
    plt.plot(anchor_x, anchor_y, label='Anchor', marker='.')
    plt.plot(test_x, test_y, label='Test', marker='+')
    plt.xlabel('Size (byte)')
    plt.ylabel('Distance (dB)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fig_path)
    plt.show()


def bat_draw_fig():
    import matplotlib.pyplot as plt
    import scienceplots
    # plt.style.use('science')
    plt.style.use(['science', 'ieee'])

    xls_dir = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/output_mesh/'
    xls_path_list = []
    for path in os.listdir(xls_dir):
        # check if current path is an xls file
        if os.path.splitext(path)[-1] == ".xls":
            xls_path_list.append(os.path.join(xls_dir, path))

    metric_index = [0, 1, 2, 3]  # D1, D2, H, absD
    sub_dict = {'midpoint': 6, 'loop': 11, 'butterfly': 16, 'modified_butterfly': 21, 'neual_subdiv': 26}
    sheet0_col_index = [i + 4 for i in metric_index]

    for path in xls_path_list:
        book = xlrd.open_workbook(path)
        sheet0 = book.sheet_by_index(0)
        sheet1 = book.sheet_by_index(1)
        fig_dir = path[:-4] + '_figs'
        my_mkdirs(fig_dir)
        for item in sub_dict.items():
            title = 'Draco_original-vs-Draco_downsample_' + item[0]
            fig_path = os.path.join(fig_dir, title + '.png')
            # fig_path = path[:-4] + '_' + title + '.fig'
            fig = plt.figure(num=1, figsize=(20, 20))
            plt.title(title)
            plt.xlabel('Byte')
            plt.ylabel('Distance')

            sheet1_col_index = [i + item[1] for i in metric_index]

            anchor_x = sheet0.col_values(3)[1:]  # bin_size
            test_x = sheet1.col_values(3)[2:]  # bin_size

            for metric_idx in metric_index:
                anchor_y = sheet0.col_values(sheet0_col_index[metric_idx])[1:]  # D1
                test_y = sheet1.col_values(sheet1_col_index[metric_idx])[2:]  # D1

                ax = fig.add_subplot(221 + metric_idx)
                ax.plot(anchor_x, anchor_y, label='Anchor', marker='.')
                ax.plot(test_x, test_y, label='Test', marker='*')

                ax.legend(loc='lower right')
                ax.grid()
            plt.savefig(fig_path)
            plt.close(fig)


def d1_mse2psnr(mse):
    d1_psnr = 10 * math.log10(3 * pow(2, 12) * pow(2, 12) / mse)
    return d1_psnr


def bat_cal_vmesh_d1psnr():
    print('start bat_cal_vmesh_d1psnr ')
    refer_dir = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/'
    work_dir = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/subd2_net_animal-3kk_ns3_nm50x6_norm/'
    work_dir = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/subd2_mid/'
    output_file_path = work_dir + '/Camel+Lion_neural_mid_subd2_d1_psnr_number.txt'

    job_list = ['Camel', 'Lion']

    with open(output_file_path, 'w') as f:

        for job in job_list:
            print(job)
            f.write(job + '\n')
            f.write('rate_point\tD1_PSNR\n')
            refer = refer_dir + job + '_voxelized12.obj'
            for i in range(4):
                # if job == 'elephant_midpoint/subd3/':
                #     dec_mesh_path = work_dir + job + '/enc/r' + str(i + 1) + '/elephant_voxelized12_GOF_0_fr_0_rec_subdiv.obj'
                # else:
                #     dec_mesh_path = work_dir + job + '/dec/r' + str(i + 1) + '/elephant_draco_dec_subd3.obj'
                dec_mesh_path = work_dir + job + '/dec/r' + str(i + 1) + '/'+job+'_draco_dec_subd2.obj'
                #dec_mesh_path = work_dir + job + '/dec/r' + str(i + 1) + '/' + job + '_voxelized12_dec.obj'
                dec_mesh_path = work_dir + job + '/enc/r' + str(i + 1) + '/'+job+'_voxelized12_GOF_0_fr_0_rec_subdiv.obj'
                d1_pnsr = cal_d1_psnr(refer, dec_mesh_path)
                print('\tr' + str(i + 1))
                print('\t' + str(d1_pnsr) + '\n')
                f.write('r' + str(i + 1) + '\t')
                f.write(str(d1_pnsr) + '\n')


def bat_test_subd():
    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--mesh_dir', type=str, default='/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal'
                                                             '-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd0/')
    parse.add_argument('-o', '--output_dir', type=str,
                       default='./jobs/Owl_voxelized12_f5975_ns3_nm30+10_e1500/mid_test')
    parse.add_argument('-s', '--num_subd', type=int, default=3)
    args = parse.parse_args()
    # load hyper parameters
    my_mkdirs(args.output_dir)

    # load validation set

    for i in range(1, 100, 10):
        meshPath = os.path.join(args.mesh_dir, str(i).zfill(3) + '.obj')
        # out_path = os.path.join(args.output_dir, str(i).zfill(3)+'_subd'+str(args.num_subd)+'.obj')
        mesh_subdivide(meshPath, args.output_dir, args.num_subd)


def bat_cal_d1psnr():
    import numpy as np
    parse = argparse.ArgumentParser()

    parse.add_argument('-r', '--refer_mesh', type=str, default='/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/'
                                                               'animal-3kk/voxelized/Owl_voxelized12.obj')
    parse.add_argument('-t', '--test_dir', type=str, default='./jobs/Owl_voxelized12_f2987_ns3_nm30+10_e1500/test/')
    parse.add_argument('-s', '--num_subd', type=int, default=3)
    args = parse.parse_args()
    # load hyper parameters
    my_mkdirs(args.test_dir)

    # load validation set
    output_file_path = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/net_animal-3kk_ns3_nm50x6_norm/test/'
    os.path.exists(output_file_path) or os.makedirs(output_file_path)

    f = open(os.path.join(output_file_path, 'neural_d1_psnr.txt'), 'a+')  # 打开文件以追加模式
    d1_psnr_list = []

    test = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f2987_ns3_nm30/subd0/'
    mesh_name = ['Panda', 'Camel', 'Dog', 'Elephant', 'Lion']
    test = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/'

    for name in mesh_name:
        mesh_dir = name
        print(mesh_dir)
        f.write(mesh_dir + '\n')
        refer_mesh = os.path.join(test, name + '.obj')
        max_num = 50
        d1_psnr_list = []
        for i in range(1, max_num, 5):
            fill = len(str(max_num))
            # meshPath = os.path.join(args.test_dir, str(i).zfill(3)+'.obj')
            # test_path = os.path.join(args.test_dir, str(i).zfill(2)+'_subd'+str(args.num_subd)+'.obj')

            test_path = os.path.join(output_file_path, mesh_dir, str(i).zfill(fill) + '_subd3.obj')
            d1psnr = cal_d1_psnr(refer_mesh, test_path)
            d1_psnr_list.append(d1psnr)
            print('\t' + str(i).zfill(fill) + '\t' + str(d1psnr))
            f.write('\t' + str(i).zfill(fill) + '\t' + str(d1psnr) + '\n')
        f.write('\t' + 'mean\t' + str(np.mean(d1_psnr_list)) + '\n')

def write_info(directory, obj_list, output_file_path):
    f = open(output_file_path, 'a+')
    title = ['name', 'face_num', 'vertex_num', 'watertight', 'edge_manifold', 'vertex_manifold',
             'self_intersect', 'orientable']
    for t in title:
        f.write(t + '\t')
    for num, obj_file in enumerate(obj_list):
        f.write('\n')
        f.write(obj_file + '\t')
        print(num, obj_file)
        # Load the mesh using trimesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(directory, obj_file))
        f.write(str(len(mesh.triangles)) + '\t')
        f.write(str(len(mesh.vertices)) + '\t')
        f.write(str(mesh.is_watertight()) + '\t')

        f.write(str(mesh.is_edge_manifold(allow_boundary_edges=0)) + '\t')
        f.write(str(mesh.is_vertex_manifold()) + '\t')
        f.write(str(mesh.is_self_intersecting()) + '\t')
        f.write(str(mesh.is_orientable()) + '\t')
    f.close()


def filt_mesh():
    import os
    import open3d as o3d
    from multiprocessing import Pool
    import trimesh
    # Define the directory path
    directory = "/work/Users/zhuzhiwei/dataset/mesh/10k_surface"
    # output file list
    output_file_path = '/work/Users/zhuzhiwei/dataset/mesh/' \
                       '10k_surface_1component+watertight+noSelfIntersect_5550_info.txt'

    input_file = '/work/Users/zhuzhiwei/dataset/mesh/10k_surface_1component+watertight+noSelfIntersect_5550.txt'

    with open(input_file, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        # 将lines分成8组
        group_size = 500
        lines = [lines[i:i + group_size] for i in range(0, len(lines), group_size)]
        # n_processes = os.cpu_count()
        n_processes = len(lines)
        print('n_processes: ', n_processes)

        pool = Pool(processes=n_processes)  # 进程池
        for n in range(n_processes):
            output_file_name = output_file_path[:-4] + '_' + str(n) + '.txt'
            pool.apply_async(write_info, args=(directory, lines[n], output_file_name))

        pool.close()  # 使进程池不能添加新任务
        pool.join()  # 等待进程结束





    # with open(input_file, 'r') as file:
    #     lines = file.readlines()
    #     lines = [line.strip() for line in lines]
    #     count = 0
    #     k_count =0
    #     for num, obj_file in enumerate(lines):
    #         # Load the mesh using trimesh
    #         try:
    #             mesh = trimesh.load(obj_file)
    #             o3d_mesh = o3d.io.read_triangle_mesh(obj_file)
    #         except:
    #             continue
    #         # Check if the mesh has only 1 connected component
    #         try:
    #             if len(mesh.split()) != 1:
    #                 continue
    #         except:
    #             continue
    #
    #         # Check if the mesh is watertight, 其实到这就可以保证：一个边有且只有两个邻面
    #         if not mesh.is_watertight:
    #             continue
    #         # Check if the mesh has no boundary
    #         if not o3d_mesh.is_edge_manifold(allow_boundary_edges=False):
    #             continue
    #         # Check if the mesh is vertex manifold
    #         if not o3d_mesh.is_vertex_manifold():
    #             continue
    #         # Check if the mesh is self intersecting
    #         if o3d_mesh.is_self_intersecting():
    #             continue
    #         # If all checks pass, print the file path
    #         obj_base_name  = os.path.basename(obj_file)
    #         print('%d/%d:\t%s' % (num, len(lines), obj_base_name))
    #         f.write(obj_base_name + '\n')
    #         count += 1
    #         if len(mesh.faces)>1000:
    #             k_count += 1
    #             f1k.write(obj_base_name + '\n')
    #
    #     f.close()
    #     f1k.close()
    #     print('total: %d' % count)
    #     print('total 1k+face: %d' % k_count)

    # f = open(output_file_path, 'w')
    # # Get a list of all OBJ files in the directory
    # obj_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.obj')]
    # # Iterate over each OBJ file
    # count = 0
    # for num, obj_file in enumerate(obj_files):
    #     # Load the mesh using trimesh
    #     try:
    #         mesh = trimesh.load(obj_file)
    #         o3d_mesh = o3d.io.read_triangle_mesh(obj_file)
    #     except:
    #         continue
    #     # Check if the mesh has only 1 connected component
    #     try:
    #         if len(mesh.split()) != 1:
    #             continue
    #     except:
    #         continue
    #     if len(mesh.split()) != 1:
    #         continue
    #     # Check if the mesh is watertight, 其实到这就可以保证：一个边有且只有两个邻面
    #     if not mesh.is_watertight:
    #         continue
    #     # Check if the mesh has no boundary
    #     if not o3d_mesh.is_edge_manifold(allow_boundary_edges=False):
    #         continue
    #     # If all checks pass, print the file path
    #     print('%d/%d:\t%s' % (num, len(obj_files), obj_file))
    #     f.write(obj_file+'\n')
    #     count += 1
    # f.close()
    # print('total: %d' % count)

def compact_file():
    n = 12
    output_file_path = '/work/Users/zhuzhiwei/dataset/mesh/' \
                       '10k_surface_1component+watertight+noSelfIntersect_5550_info.txt'

    # Open the output file in write mode
    with open(output_file_path, "w") as outfile:
        # Iterate over each file path
        for i in range(n):
            file_path = output_file_path[:-4] + '_' + str(i) + '.txt'

            with open(file_path, "r") as infile:
                # Read the contents of the file
                contents = infile.read()
                print(i)
                if i > 0:
                    # delete the first line
                    contents = contents[contents.find('\n')+1:]


                # Write the contents to the output file
                outfile.write(contents+'\n')

if __name__ == '__main__':
    # bat_cal_d1psnr()
    # compact_file()
    bat_cal_vmesh_d1psnr()
    # filt_mesh()
    # mesh_path = '/work/Users/zhuzhiwei/vmesh/mesh_decimated_test/sold_NoCompressTexture/s2c2r2_sold_GOF_0_fr_0_decimate.ply'
    # # mesh_path = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/Panda.obj'
    # boundray(mesh_path)
    # convert_to_watertight(mesh_path)

    # mesh_geometry_voxelized()
    # bat_cal_vmesh_d1psnr()

    # refer = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12.obj'
    # neural_ref = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f2987_ns3_nm30/subd3/30.obj'
    # neural_ref_f5975 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd3/001.obj'
    # neural_test_output = '/work/Users/zhuzhiwei/neuralSubdiv-master/jobs/Owl_voxelized12_f5975_ns3_nm30+10/0_subd3.obj' #未scale,尺度不对
    # print(cal_d1_psnr(refer, neural_test_output))
    # neu3_dis = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd2/dec/Owl_voxelized12_dec.obj'
    # neu3 = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3/dec/Owl_draco_rec_subd3.obj'
    # mid3_dis = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3/dec/Owl_voxelized12_dec.obj'
    # mid3 = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3/enc/Owl_voxelized12_GOF_0_fr_0_rec_subdiv.obj'
    # suddiv_compress = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/enc/Owl_voxelized12_GOF_0_fr_0_subdiv_compress.ply'
    # intra_subdiv = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/enc/Owl_voxelized12_GOF_0_fr_0_intra_subdiv.ply'
    # intra_nsubdiv = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/enc/Owl_voxelized12_GOF_0_fr_0_intra_nsubdiv.ply'
    #
    # mid3_linear = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3_LinearLifting/dec/Owl_voxelized12_dec.obj'
    # mid3_l_sub = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3_LinearLifting/enc/Owl_voxelized12_GOF_0_fr_0_rec_subdiv.obj'
    # print(distance_metric(refer, mid3_linear, 500000))
    # print(distance_metric(refer, mid3_dis, 500000))
    # # print(distance_metric(refer, neu3, 500000))
    # # print(distance_metric(refer, mid3, 500000))
    # # a = [11.95, 12.33, 11.97, 11.964, 12.228]
    # # for ai in a:
    # #     print(cal_D1_psnr(ai))
    # mid3_dir = "/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3/"
    # mid3_linea_dir = "/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_midpoint/subd3_LinearLifting/"
    # neu3_dir = "/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3/"
    # job_list = job_list = ['Owl_midpoint_kdDisp/subd3_LinearLifting', 'Owl_midpoint_kdDisp/subd3',
    #             'Owl_midpoint/subd3_LinearLifting', 'Owl_midpoint/subd3', 'Owl_neuralsubdiv/subd3']
    #
    # for i in range(4):
    #     print('r'+str(i+1))
    #     mid3_dec = mid3_dir+'dec/r'+str(i+1)+'/Owl_voxelized12_dec.obj'
    #     mid3_linear_dec = mid3_linea_dir+'dec/r'+str(i+1)+'/Owl_voxelized12_dec.obj'
    #     neu3_dec = neu3_dir+'dec/r'+str(i+1)+'/Owl_voxelized12_dec.obj'
    #     print('mid3_dec', cal_d1_psnr(refer, mid3_dec))
    #     print('mid3_linear_dec', cal_d1_psnr(refer, mid3_linear_dec))
    #     print('neu3_dec', cal_d1_psnr(refer, neu3_dec))
    # bat_cal_vmesh_d1psnr()

    # refer_mesh = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12.obj'
    # # # subd3_2987 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f2987_ns3_nm10/subd3/01.obj'
    # # # subd3_5975 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm10/subd3/01.obj'
    # # # neural = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3_Owl_f5975_ofit/dec/r2/Owl_draco_dec_subd3.obj'
    # # # cal_d1_psnr(refer_mesh, neural)
    # # output = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3_Owl_f5975_ofit/dec/r2/'
    # # output2 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd0/'
    # # r2_draco_dec = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3_Owl_f5975_ofit/dec/r2/Owl_draco_dec.obj'
    # neural_train_subd0 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd0/001.obj'
    # r4_draco_dec = '/work/Users/zhuzhiwei/vmesh/neuralSubdiv_experiment/animal-3kk/Owl_neuralsubdiv/subd3_Owl_f2987/dec/r4/Owl_draco_dec.obj'
    # # neural_subd3 = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/animal-3kk/voxelized/Owl_voxelized12_f5975_ns3_nm100/subd0/001_subd3.obj'
    # # mesh_subdivide(neural_train_subd0, output, iters=3)
    # # output_path = output + '001_mid3.obj'
    # # cal_d1_psnr(refer_mesh, output_path)
    # # cal_d1_psnr(refer_mesh, neural_subd3)
    # # cal_d1_psnr(refer_mesh, subd3_5975)
    # cal_d1_psnr(refer_mesh, r4_draco_dec)
    # cal_d1_psnr(refer_mesh, neural_train_subd0)
    # bat_draw_fig()
    # mesh_compression()
    # mesh_downsample()
    # mesh_subdivide()
    # mesh_normalize()
    # mesh_scale()
    # mesh_metric()

    # bat_test_subd()
    # bat_cal_d1psnr()

    # mesh_subdivide_butterfly()
    # mesh_metric()
    # gt_mesh = '/work/Users/zhuzhiwei/dataset/mesh/free3d-animal/watertight5/elephant_overfit.obj'
    # test_mesh_dir = '/work/Users/zhuzhiwei/neuralSubdiv/data_meshes/cartoon_elephant_200/'
    #
    # for i in range(3):
    #     test_mesh = test_mesh_dir + 'subd'+str(i)+'/200.obj'
    #     dist_dict = distance_metric(gt_mesh, test_mesh, 500000)
    #     dist_dict.insert(0, i)
    #     print(dist_dict)
    # input_mesh_dir = '/work/Users/zhuzhiwei/draco/test_neural_subdiv_mesh/output_bin/'
    # input_dir1 = input_mesh_dir + 'original'
    # input_dir2 = input_mesh_dir + 'downsample'
    #
    # list1 = []
    # list2 = []
    #
    # for path in os.listdir(input_dir1):
    #
    #     # check if current path is an obj file
    #     if os.path.splitext(path)[-1] == ".drc":
    #         l1 = []
    #         l1.append(path)
    #         path = os.path.join(input_dir1,path)
    #         l1.append(os.path.getsize(path))
    #         list1.append(l1)
    #
    # for path in os.listdir(input_dir2):
    #
    #     # check if current path is an obj file
    #     if os.path.splitext(path)[-1] == ".drc":
    #         l1 = []
    #         l1.append(path)
    #         path = os.path.join(input_dir2, path)
    #         l1.append(os.path.getsize(path))
    #         list2.append(l1)
    # print(list1)
    # print(list2)
    # input_mesh_dir = '/work/Users/zhuzhiwei/dataset/mesh/mpeg-mesh-content/'
    # for path in os.listdir(input_mesh_dir):
    #     if path[-4:] != '.zip':
    #         dir_path = os.path.join(input_mesh_dir, path)
    #         count = 0
    #         for i_path in os.listdir(dir_path) :
    #             if (i_path[-4:] == '.obj') and (count<1):
    #                 count+=1
    #                 file_path = os.path.join(dir_path, i_path)
    #                 print(file_path)
    #                 base_mesh = o3d.io.read_triangle_mesh(file_path)
    #                 print(base_mesh.is_watertight())
    #                 print(base_mesh.is_vertex_manifold())
    #                 print(base_mesh.is_edge_manifold())

    # 将有边界的网格处理成无边界的网格
    # mesh = o3d.io.read_triangle_mesh("/work/Users/zhuzhiwei/dataset/mesh/mpeg-mesh-content/airplane/airplane_0001.obj")
