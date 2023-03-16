import open3d as o3d
import argparse
import os.path

def mesh_subdivide():
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
    mesh_file_list = []
    # Iterate directory
    for path in os.listdir(args.input_mesh_dir):
        # check if current path is a obj file
        if os.path.splitext(path)[-1] == ".obj":
            mesh_file_list.append(path)
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

def mesh_downsample():
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


if __name__ == '__main__':
    # mesh_downsample()
    mesh_subdivide()

