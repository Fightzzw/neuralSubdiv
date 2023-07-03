import numpy as np
from trimesh.geometry import faces_to_edges
from trimesh.constants import tol
from trimesh import util
from trimesh import grouping
from trimesh import graph

import open3d as o3d

def creat_half_edge(edges, vertices, edges_face):
    half_edge_list = []

    for i in range(len(edges)):
        half_edge = o3d.geometry.HalfEdge()
        half_edge.next = edges[(i+1)%3]
        half_edge.triangle_index = edges_face[i]
        v = vertices[i]




def subdivide_modified_butterfly( vertices,
                   faces,
                   iterations=None):
    """
    Subdivide a mesh by dividing each triangle into four triangles
    and approximating their smoothed surface (loop subdivision).
    This function is an array-based implementation of butterfly subdivision,
    which avoids slow for loop and enables faster calculation.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    iterations : int
          Number of iterations to run subdivision

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices

    """



    try:
        from itertools import zip_longest
    except BaseException:
        # python2
        from itertools import izip_longest as zip_longest

    if iterations is None:
        iterations = 1

    def _subdivide(vertices, faces):
        # find the unique edges of our faces
        edges, edges_face = faces_to_edges(
            faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)

        # set interior edges if there are two edges and boundary if there is
        # one.
        edge_inter = np.sort(
            grouping.group_rows(
                edges,
                require_count=2),
            axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)
        # make sure that one edge is shared by only one or two faces.
        if not len(edge_inter) * 2 + len(edge_bound) == len(edges):
            # we have multiple bodies it's a party!
            # edges shared by 2 faces are "connected"
            # so this connected components operation is
            # essentially identical to `face_adjacency`
            faces_group = graph.connected_components(
                edges_face[edge_inter])

            if len(faces_group) == 1:
                raise ValueError('Some edges are shared by more than 2 faces')

            # collect a subdivided copy of each body
            seq_verts = []
            seq_faces = []
            # keep track of vertex count as we go so
            # we can do a single vstack at the end
            count = 0
            # loop through original face indexes
            for f in faces_group:
                # a lot of the complexity in this operation
                # is computing vertex neighbors so we only
                # want to pass forward the referenced vertices
                # for this particular group of connected faces
                unique, inverse = grouping.unique_bincount(
                    faces[f].reshape(-1), return_inverse=True)

                # subdivide this subset of faces
                cur_verts, cur_faces = _subdivide(
                    vertices=vertices[unique],
                    faces=inverse.reshape((-1, 3)))

                # increment the face references to match
                # the vertices when we stack them later
                cur_faces += count
                # increment the total vertex count
                count += len(cur_verts)
                # append to the sequence
                seq_verts.append(cur_verts)
                seq_faces.append(cur_faces)

            # return results as clean (n, 3) arrays
            return np.vstack(seq_verts), np.vstack(seq_faces)

        # set interior, boundary mask for unique edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask

        # find vertex neighbors of each vertex
        neighbors = graph.neighbors(
            edges=edges[unique],
            max_index=len(vertices))

        # number of neighbors
        k = [len(i) for i in neighbors]



        # odd vertices
        # boundary vertices
        e_bound = edges[unique[edge_bound_mask]]
        odd_bound_vertices = []

        for i in range(e_bound):
            v0_idx = e_bound[i, 0]
            v1_idx = e_bound[i, 1]
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            v0_neighbors = neighbors[v0_idx]
            v1_neighbors = neighbors[v1_idx]

            k0 = k[v0_idx]
            k1 = k[v1_idx]
            assert (k0 == 2)
            assert (k1 == 2)
            if v0_neighbors[0] == v1_idx:
                v2_idx = v0_neighbors[1]
            else:
                v2_idx = v0_neighbors[0]

            if v1_neighbors[0] == v0_idx:
                v3_idx = v1_neighbors[1]
            else:
                v3_idx = v1_neighbors[0]
            v2 = vertices[v2_idx]
            v3 = vertices[v3_idx]
            odd = (9 * (v0 + v1) -(v2 + v3))/16
            odd_bound_vertices.append(odd)

        # interior vertices
        e_inter = edges[unique[edge_inter_mask]]

        odd_inter_vertices = []

        for i in range(e_inter):
            v0_idx = e_inter[i, 0]
            v1_idx = e_inter[i, 1]
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            v0_neighbors = neighbors[v0_idx]
            v0_neighbors.remove(v1_idx)
            v1_neighbors = neighbors[v1_idx]
            v1_neighbors.remove(v0_idx)
            k0 = k[v0_idx]
            k1 = k[v1_idx]
            odd = 0
            if k0 == 6 & k1 == 6:
                b_list = [x for x in v0_neighbors if x in v1_neighbors]
                assert len(b_list) == 2
                c_idx = b_list[0]
                d_idx = b_list[1]
                c_neighbors = neighbors[c_idx]
                d_neighbors = neighbors[d_idx]
                e_list = [x for x in v0_neighbors if x in c_neighbors]
                f_list = [x for x in v1_neighbors if x in c_neighbors]
                g_list = [x for x in v0_neighbors if x in d_neighbors]
                h_list = [x for x in v1_neighbors if x in d_neighbors]
                c = vertices[c_idx]
                d = vertices[d_idx]
                e = vertices[e_list[0]]
                f = vertices[f_list[0]]
                g = vertices[g_list[0]]
                h = vertices[h_list[0]]

                odd = (8 * (v0 + v1) + 2 * (c + d) - (e + f + g + h))/16

            elif k0 == 6 & k1 != 6:
                odd = 0.75 * v1
                if k1 == 3:
                    e_list = [x for x in v1_neighbors if x != v0_idx]
                    assert len(e_list) == 2
                    e1 = vertices[e_list[0]]
                    e2 = vertices[e_list[1]]
                    odd += (5 * v0 - e1 - e2) / 12
                elif k1 == 4:
                    e2 = v1_neighbors[2]
                    odd += (3 * v0 - e2) / 8
                elif k1 >= 5:
                    for j in range(len(v1_neighbors)):
                        odd += vertices[v1_neighbors[j]] * \
                               (1 + 4 * np.cos(2 * np.pi * j / k1) + 2 * np.cos(4 * np.pi * j / k1)) / 4 / k1
            elif k0 != 6 & k1 == 6:
                odd = 0.75 * v0
                if k1 == 3:
                    odd += (5 * v1 - e1 - e2) / 12
                elif k1 == 4:
                    odd += (3 * v1 - e2) / 8
                elif k1 >= 5:
                    for j in range(len(v0_neighbors)):
                        odd += vertices[v0_neighbors[j]] * \
                               (1 + 4 * np.cos(2 * np.pi * j / k0) + 2 * np.cos(4 * np.pi * j / k0)) / 4 / k0
            elif k0 != 6 & k1 != 6:
                odd1 = 0.75 * v1
                if k1 == 3:
                    odd1 += (5 * v0 - e1 - e2) / 12
                elif k1 == 4:
                    odd1 += (3 * v0 - e2) / 8
                elif k1 >= 5:
                    for j in range(len(v1_neighbors)):
                        odd1 += vertices[v1_neighbors[j]] * \
                               (1 + 4 * np.cos(2 * np.pi * j / k1) + 2 * np.cos(4 * np.pi * j / k1)) / 4 / k1
                odd0 = 0.75 * v0
                if k1 == 3:
                    odd0 += (5 * v1 - e1 - e2) / 12
                elif k1 == 4:
                    odd0 += (3 * v1 - e2) / 8
                elif k1 >= 5:
                    for j in range(len(v0_neighbors)):
                        odd0 += vertices[v0_neighbors[j]] * \
                               (1 + 4 * np.cos(2 * np.pi * j / k0) + 2 * np.cos(4 * np.pi * j / k0)) / 4 / k0
                odd = (odd0 + odd1)/2

            odd_inter_vertices.append(odd)
        odd_vertices = []
        bound_count = 0
        inter_count = 0
        for i in range(unique.shape[0]):
            if edge_bound_mask[i]:
                odd_vertices[i] = odd_bound_vertices[bound_count]
                bound_count += 1
            else:
                odd_vertices[i] = odd_inter_vertices[inter_count]
                inter_count += 1



        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack([
            faces[:, 0],
            odd_idx[:, 0],
            odd_idx[:, 2],
            odd_idx[:, 0],
            faces[:, 1],
            odd_idx[:, 1],
            odd_idx[:, 2],
            odd_idx[:, 1],
            faces[:, 2],
            odd_idx[:, 0],
            odd_idx[:, 1],
            odd_idx[:, 2]]).reshape((-1, 3))

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((vertices, odd_vertices))

        return new_vertices, new_faces

    for _ in range(iterations):
        vertices, faces = _subdivide(vertices, faces)

    if tol.strict or True:
        assert np.isfinite(vertices).all()
        assert np.isfinite(faces).all()
        # should raise if faces are malformed
        assert np.isfinite(vertices[faces]).all()

        # none of the faces returned should be degenerate
        # i.e. every face should have 3 unique vertices
        assert (faces[:, 1:] != faces[:, :1]).all()

    return vertices, faces

if __name__ == '__main__':
    mesh_path = '/work/Users/zhuzhiwei/neuralSubdiv-master/data_meshes/objs/fish.obj'
    mesh = trimesh.load(mesh_path)
    mesh.subdivide_loop(1)