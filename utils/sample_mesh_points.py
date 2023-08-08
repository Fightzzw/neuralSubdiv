import os
import numpy as np
import torch
import random

def slow_samplePointsOnMesh(V, F, numSamplePoints):
    """
    :param V: 顶点坐标 tensor (nV, 3)
    :param F: 面索引 tensor (nF, 3)
    :param numSamplePoints: 采样点数 int
    :return: 采样点坐标 tensor (numSamplePoints, 3)
    """

    # Compute face areas
    face_areas = torch.zeros(len(F), dtype=torch.float32)
    for i, face in enumerate(F):
        v0, v1, v2 = face
        edge1 = V[v1] - V[v0]
        edge2 = V[v2] - V[v0]
        face_areas[i] = torch.norm(torch.cross(edge1, edge2)) / 2.0
    # Normalize face areas to probabilities
    probabilities = face_areas / torch.sum(face_areas)
    # Sample points on the mesh
    sampled_points = torch.zeros(numSamplePoints, 3, dtype=torch.float32)
    for i in range(numSamplePoints):
        # Randomly select a face based on the probabilities
        face_index = torch.multinomial(probabilities, 1).item()
        v0, v1, v2 = F[face_index]
        # Generate random barycentric coordinates
        r1, r2 = sorted([random.random(), random.random()])
        r0 = 1 - r1 - r2
        # Compute the sampled point on the face
        sampled_points[i] = r0 * V[v0] + r1 * V[v1] + r2 * V[v2]
    return sampled_points

def save_obj(filename, vertices, faces):
    """
    Args:
        filename: str
        vertices: tensor(float), shape (num_vertices, 3)
        faces: tensor(long), shape (num_faces, 3)
    """
    assert filename.endswith('.obj'), 'file name must end with .obj'
    with open(filename, 'w') as f:
        for vert in vertices:
            f.write('v %f %f %f\n' % tuple(vert))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face+1))


def obj2nvc(vertices, faces):
    """
    Args:
        vertices: tensor(float), shape (num_vertices, 3)
        faces: tensor(long), shape (num_faces, 3)
    Returns:
        mesh: tensor(float), shape (num_faces, 3, 3), (num_faces, 3 vertices, xyz coordinates)
    """
    mesh = vertices[faces.flatten()].reshape(faces.size()[0], 3, 3)
    return mesh.contiguous()


def nvc2obj(mesh):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
    Returns:
        vertices: tensor(float), shape (num_vertices, 3)
        faces: tensor(long), shape (num_faces, 3)
    """
    unique_v, idx = np.unique(mesh.view(-1, 3).cpu(), axis=0, return_inverse=True)
    vertices = torch.from_numpy(unique_v)
    faces = torch.from_numpy(idx).view(-1, 3)
    return vertices, faces


def normalize_mesh(mesh):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
    Returns:
        mesh: tensor(float), shape (num_faces, 3, 3)
    """
    mesh = mesh.reshape(-1, 3)

    mesh_max = torch.max(mesh, dim=0)[0]
    mesh_min = torch.min(mesh, dim=0)[0]
    mesh_center = (mesh_max + mesh_min) / 2.0
    mesh = mesh - mesh_center

    max_length = torch.sqrt(torch.max(torch.sum(mesh**2, dim=-1)))
    mesh /= max_length

    mesh = mesh.reshape(-1, 3, 3)
    return mesh




def face_normals(mesh):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
    Returns:
        normals: tensor(float), shape (num_faces, 3)
    """
    vec_a = mesh[:, 0] - mesh[:, 1]  # 每个三角形面的第0个顶点坐标-第1个顶点坐标=矢量10
    vec_b = mesh[:, 1] - mesh[:, 2]  # 每个三角形面的第1个顶点坐标-第2个顶点坐标=矢量21
    normals = torch.cross(vec_a, vec_b)  # 计算两个矢量的叉乘得到法线
    return normals


def area_weighted_distribution(mesh, normals=None):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
        normals: tensor(float), shape (num_faces, 3)
    Returns:
        distrib: distribution
    """
    if normals is None:
        normals = face_normals(mesh)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10
    distrib = torch.distributions.Categorical(areas.view(-1))
    return distrib


def sample_uniformly(mesh, num_samples):
    """
    sample uniformly in [-1,1] bounding volume.
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
        num_samples: int
    Returns:
        samples: tensor(float), shape (num_samples, 3)
    """
    samples = (torch.rand(num_samples, 3) - 0.5) * 1.1
    samples = samples.to(mesh.device)
    return samples


def sample_on_surface(mesh, num_samples, normals=None, distrib=None):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
        num_samples: int
        normals: tensor(float), shape (num_faces, 3)
        distrib: distribution
    Returns:
        samples: tensor(float), shape (num_samples, 3)
        normals: tensor(float), shape (num_samples, 3)
    """
    if normals is None:
        normals = face_normals(mesh)
    if distrib is None:
        distrib = area_weighted_distribution(mesh, normals)
    idx = distrib.sample([num_samples])
    selected_faces = mesh[idx]
    selected_normals = normals[idx]
    u = torch.sqrt(torch.rand(num_samples)).to(mesh.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(mesh.device).unsqueeze(-1)
    samples = (1 - u) * selected_faces[:,0,:] + (u * (1 - v)) * selected_faces[:,1,:] + u * v * selected_faces[:,2,:]
    return samples, selected_normals


def sample_near_surface(mesh, num_samples, normals=None, distrib=None):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
        num_samples: int
        normals: tensor(float), shape (num_faces, 3)
        distrib: distribution
    Returns:
        samples: tensor(float), shape (num_samples, 3)
    """
    samples = sample_on_surface(mesh, num_samples, normals, distrib)[0]
    samples += torch.randn_like(samples) * 0.01
    return samples


def sample_points(mesh, num_samples_and_method, normals=None, distrib=None):
    """
    Args:
        mesh: tensor(float), shape (num_faces, 3, 3)
        num_samples_and_method: [tuple(int, str)]
        normals: tensor(float), shape (num_faces, 3)
        distrib: distribution
    Returns:
        samples: tensor(float), shape (num_samples, 3)
    """
    if normals is None:
        normals = face_normals(mesh)
    if distrib is None:
        distrib = area_weighted_distribution(mesh, normals)
    samples = []
    for num_samples, method in num_samples_and_method:
        if method == 'uniformly':
            samples.append(sample_uniformly(mesh, num_samples))
        elif method == 'surface':
            samples.append(sample_on_surface(mesh, num_samples, normals, distrib)[0])
        elif method == 'near':
            samples.append(sample_near_surface(mesh, num_samples, normals, distrib))
    samples = torch.cat(samples, dim=0)
    return samples



