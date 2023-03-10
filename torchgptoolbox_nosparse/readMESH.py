import numpy as np
import torch
import open3d as o3d

def readMESH(filepath):
    """
    Input:
      filepath a string of mesh file path
    Output:
      V (|V|,3) torch tensor of vertex positions (float)
	  F (|F|,3) torch tensor of face indices (long)
    """

    V = []
    F = []
    # if filepath.endswith(".obj"):
    #     with open(filepath, "r") as f:
    #         lines = f.readlines()
    #     while True:
    #         for line in lines:
    #             if line == "":
    #                 break
    #             elif line.strip().startswith("vn"):
    #                 continue
    #             elif line.strip().startswith("vt"):
    #                 continue
    #             elif line.strip().startswith("v"):
    #                 vertices = line.replace("\n", "").split(" ")[1:]
    #                 vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
    #                 V.append(list(map(float, vertices)))
    #             elif line.strip().startswith("f"):
    #                 t_index_list = []
    #                 for t in line.replace("\n", "").split(" ")[1:]:
    #                     t_index = t.split("/")[0]
    #                     try:
    #                         t_index_list.append(int(t_index) - 1)
    #                     except ValueError:
    #                         continue
    #                 F.append(t_index_list)
    #             else:
    #                 continue
    #         break
    #     V = np.asarray(V)
    #     F = np.asarray(F)
    #
    # elif filepath.endswith(".ply"):
    ply = o3d.io.read_triangle_mesh(filepath)
    V = np.asarray(ply.vertices)
    F = np.asarray(ply.triangles)

    V_th = torch.from_numpy(V).float()
    F_th = torch.from_numpy(F).long()
    return V_th, F_th

