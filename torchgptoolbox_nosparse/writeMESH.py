import numpy as np
import torch
import open3d as o3d

def writeMESH(fileName,V_torch,F_torch):
    '''

    Input:
      filepath a string of mesh file path
      V (|V|,3) torch tensor of vertex positions (float)
	  F (|F|,3) torch tensor of face indices (long)
    '''
    V = V_torch.data.numpy()
    F = F_torch.data.numpy()
    if fileName.endswith(".obj"):
        f = open(fileName, 'w')
        for ii in range(V.shape[0]):
            string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
            f.write(string)
        Ftemp = F + 1
        for ii in range(F.shape[0]):
            string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
            f.write(string)
        f.close()
    elif fileName.endswith(".ply"):
        ply = o3d.geometry.TriangleMesh()
        ply.vertices = o3d.utility.Vector3dVector(V)
        ply.triangles = o3d.utility.Vector3iVector(F)
        o3d.io.write_triangle_mesh(fileName, ply)