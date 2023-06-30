# ------------------------------------------------------------------------------------------
# Copyright (c) 2023 Zhu Zhiwei
# This code is modified from :
# Gaussian-Product Subdivision Surfaces - Python Demo
# Copyright (c) 2020 Reinhold Preiner
#
# This code is licensed under the MIT license.
# https://opensource.org/licenses/mit-license.html
# ------------------------------------------------------------------------------------------

import igl
import numpy as np
import os

class Mesh:
    v = []
    f = []
def loop_subd(i_mesh, o_path, numSubd):
    lmesh = i_mesh
    for _ in range(numSubd):
        lmesh.v, lmesh.f = igl.loop(lmesh.v, lmesh.f)

    igl.write_triangle_mesh(o_path, lmesh.v, lmesh.f)
    print("Wrote Loop mesh to", o_path)
def gps_subd(i_mesh, o_path, numSubd):

    # ------------------------------------------------------------------------------------------
    # Gaussian Inference via product of face Gaussians (Eq. 16 and 17)
    cm = Mesh()
    cm.v = i_mesh.v.copy()
    cm.f = i_mesh.f.copy()

    # create list of empty (inverse) covariances
    icov = np.zeros((len(i_mesh.v), 3, 3))

    # 1. infer face covs and inverse vertex covs
    for f in cm.f:
        # compute covariance of face vertices
        cov = np.zeros((3, 3))
        sum = np.zeros(3)
        for j in [1, 2]:
            v = cm.v[f[j]] - cm.v[f[0]]
            sum += v
            cov += np.outer(v, v)  # outer product
        cov = cov / 3 - np.outer(sum, sum) / 9
        # bias covariance by some fraction of its dominant eigenvalue
        bias = np.linalg.eigvalsh(cov)[2] * 0.05
        cov += np.identity(3) * bias
        # inverse cov at vertices is given by the sum of inverse of surrounding face covs
        for fv in f:
            icov[fv] += np.linalg.inv(cov)

    # 2. transform to 9D dual-space vertices
    qq1 = np.zeros((len(cm.v), 3))
    qq2 = np.zeros((len(cm.v), 3))
    qlin = np.zeros((len(cm.v), 3))
    for i, ic in enumerate(icov):
        icf = ic.flatten()
        qq1[i] = [icf[0], icf[1], icf[2]]
        qq2[i] = [icf[4], icf[5], icf[8]]
        qlin[i] = ic @ cm.v[i]

    # 3. perform Gaussian-product subdivision
    #    note: igl.loop only handles 3D subdivs -> split the 9D mesh into three 3D ones
    for _ in range(numSubd):
        qq1, f = igl.loop(qq1, cm.f)
        qq2, f = igl.loop(qq2, cm.f)
        qlin, cm.f = igl.loop(qlin, cm.f)

    # 4. transform back to 3D
    cm.v = np.zeros((len(qlin), 3))
    for i, ql in enumerate(qlin):
        icov = [qq1[i],
                [qq1[i][1], qq2[i][0], qq2[i][1]],
                [qq1[i][2], qq2[i][1], qq2[i][2]]]
        cm.v[i] = np.linalg.inv(icov) @ ql

    # ------------------------------------------------------------------------------------------
    # save meshes to  file

    igl.write_triangle_mesh(o_path, cm.v, cm.f)
    print("Wrote GPS mesh to", o_path)


if __name__ == "__main__":
    # load input triangle mesh
    # i_mesh_file = "data_meshes/bob_f600_ns3_nm20/subd0/01.obj"
    # mesh = Mesh()
    # mesh.v, mesh.f = igl.read_triangle_mesh(i_mesh_file)
    # loop_subd(mesh, "/work/Users/zhuzhiwei/neuralSubdiv-master/test/gps/bob_01_loop.obj", 3)
    # #gps_subd(mesh, "/work/Users/zhuzhiwei/neuralSubdiv-master/test/gps/bob_01_gps.obj", 3)
    numSubd = 3
    output_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv-master/test/'

    mesh_root_dir = '/work/Users/zhuzhiwei/neuralSubdiv/data_meshes/'
    mesh_list = ['jaw_f1000_ns3_nm200', 'Horse_f1000_ns3_nm200', 'horse_f600_ns3_nm200']
    for mesh_name in mesh_list:
        for i in range(1, 200, 20):
            mesh_dir = os.path.join(mesh_root_dir, mesh_name, 'subd0')
            meshPath = [os.path.join(mesh_dir, str(i).zfill(3) + '.obj')]
            mesh = Mesh()
            mesh.v, mesh.f = igl.read_triangle_mesh(meshPath[0])

            output_dir_gps = os.path.join(output_root_dir, 'gps',mesh_name)
            os.path.exists(output_dir_gps) or os.makedirs(output_dir_gps)
            out_path_gps = os.path.join(output_dir_gps, str(i).zfill(3) + '_subd' + str(numSubd) + '.obj')
            gps_subd(mesh, out_path_gps, numSubd)

            output_dir_loop = os.path.join(output_root_dir, 'loop', mesh_name)
            os.path.exists(output_dir_loop) or os.makedirs(output_dir_loop)
            out_path_loop = os.path.join(output_dir_loop, str(i).zfill(3) + '_subd' + str(numSubd) + '.obj')
            loop_subd(mesh, out_path_loop, numSubd)
