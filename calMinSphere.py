from scipy.spatial import Delaunay, SphericalVoronoi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import open3d as o3d
import random
def show_sphere(center, radius):
    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.25)
    # Plot the center
    ax.scatter(center[0], center[1], center[2], c='r', s=100, marker='o')
    plt.show()


def show_tetrahedron(points):
    # # Define the four points
    # p1 = np.array([0, 0, 0])
    # p2 = np.array([0, 4, 0])
    # p3 = np.array([4, 0, 0])
    # p4 = np.array([0, 0, 4])
    # Define the vertices and faces of the tetrahedron
    vertices = points
    vertices = np.random.rand(4, 3)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Add the tetrahedron to the plot
    tetra = Poly3DCollection(vertices[faces], facecolors='b', alpha=0.25, linewidths=0.5, edgecolors='k')
    ax.add_collection3d(tetra)
    # add edges
    edge = Line3DCollection(vertices[faces], linewidths=1, edgecolors='k', linestyles=':')
    ax.add_collection3d(edge)
    # Set the limits of the plot
    ax.set_xlim3d([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim3d([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim3d([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
    # Show the plot
    print('Showing the plot...')
    plt.show()


def o3d_bounding_sphere(pcd):
    # Load point cloud from file
    # Compute minimum bounding sphere
    bounding_sphere = pcd.get_axis_aligned_bounding_box().get_sphere()
    # Draw the bounding sphere
    bounding_sphere.color = (1, 0, 0)
    o3d.visualization.draw_geometries([pcd, bounding_sphere])
    # Print center and radius of the bounding sphere
    print("Center:", bounding_sphere.center)
    print("Radius:", bounding_sphere.radius)

def trimesh_4points_bs(points):
    # 计算的是外接球
    import trimesh
    # 用points作为mesh的顶点
    mesh = trimesh.Trimesh(vertices=points)
    # Compute the minimum bounding sphere
    center, radius = trimesh.nsphere.minimum_nsphere(mesh.vertices)
    # Print the center and radius of the bounding sphere
    print("Center:", center)
    print("Radius:", radius)
    return center, radius


def naive_bs(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    # Define an initial sphere that encloses the first three points
    center = np.mean([p1, p2, p3], axis=0)
    radius = np.max(np.linalg.norm([p1, p2, p3], axis=1))
    # Check if the fourth point lies outside the sphere
    if np.linalg.norm(p4 - center) > radius:
        # Remove the point that is farthest from the center of the sphere
        farthest_point = np.argmax(np.linalg.norm([p1, p2, p3] - center, axis=1))
        if farthest_point == 0:
            p1 = p2
            p2 = p3
        elif farthest_point == 1:
            p2 = p3
        # Recompute the center and radius of the sphere
        center = np.mean([p1, p2, p3], axis=0)
        radius = np.max(np.linalg.norm([p1, p2, p3], axis=1))
    # Check if the fourth point lies outside the sphere
    if np.linalg.norm(p4 - center) > radius:
        # Remove the point that is farthest from the center of the sphere
        farthest_point = np.argmax(np.linalg.norm([p1, p2, p4] - center, axis=1))
        if farthest_point == 0:
            p1 = p2
            p2 = p4
        elif farthest_point == 1:
            p2 = p4
        # Recompute the center and radius of the sphere
        center = np.mean([p1, p2, p4], axis=0)
        radius = np.max(np.linalg.norm([p1, p2, p4], axis=1))
    # Check if the fourth point lies outside the sphere
    if np.linalg.norm(p4 - center) > radius:
        # Remove the point that is farthest from the center of the sphere
        farthest_point = np.argmax(np.linalg.norm([p1, p3, p4] - center, axis=1))
        if farthest_point == 0:
            p1 = p3
            p3 = p4
        # Recompute the center and radius of the sphere
        center = np.mean([p1, p3, p4], axis=0)
        radius = np.max(np.linalg.norm([p1, p3, p4], axis=1))
    # Check if the fourth point lies outside the sphere
    if np.linalg.norm(p4 - center) > radius:
        raise ValueError("Points are not enclosed by a sphere")
    print("Center:", center)
    print("Radius:", radius)
    return center, radius


def bs_objective(points):
    from scipy.optimize import minimize
    import numpy as np
    # Define the four points in 3D space
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    # Define an initial sphere that encloses the first three points
    center = np.mean([p1, p2, p3], axis=0)
    radius = np.max(np.linalg.norm([p1, p2, p3], axis=1))

    # Define the objective function to minimize the volume of the sphere
    def objective(x):
        return 4 / 3 * np.pi * x[3] ** 3

    # Define the constraint that the sphere must enclose all four points
    def constraint(x):
        return np.linalg.norm(p1 - x[:3]) - x[3], \
               np.linalg.norm(p2 - x[:3]) - x[3], \
               np.linalg.norm(p3 - x[:3]) - x[3], \
               np.linalg.norm(p4 - x[:3]) - x[3]

    # Use an optimization algorithm to find the center and radius of the minimum volume enclosing sphere
    result = minimize(objective, [center[0], center[1], center[2], radius],
                      constraints={'fun': constraint, 'type': 'ineq'})
    # Check if the optimization was successful
    if not result.success:
        raise ValueError("Optimization failed")
    # Return the center and radius of the minimum volume enclosing sphere
    center = result.x[:3]
    radius = result.x[3]
    print("Center:", center)
    print("Radius:", radius)
    return center, radius


def get_min_sphere_4points(points):
    """
    Get the minimum radius of a circumscribed sphere that encloses all the points
    """
    def minimum_enclosing_sphere_3points(triangle):
        # Compute the circumcenter of the triangle
        a, b, c = triangle
        ab = b - a
        ac = c - a
        ab_cross_ac = np.cross(ab, ac)
        ab_cross_ac_norm_sq = np.dot(ab_cross_ac, ab_cross_ac)
        if ab_cross_ac_norm_sq == 0:
            # Points are colinear, return a point and radius of infinity
            return a, np.inf
        ab_norm_sq = np.dot(ab, ab)
        ac_norm_sq = np.dot(ac, ac)
        circumcenter = a + (np.cross(ab_norm_sq * ac - ac_norm_sq * ab, ab_cross_ac) / (2 * ab_cross_ac_norm_sq))
        # Calculate the radius of the circumcircle
        radius = np.linalg.norm(circumcenter - a)
        # Check if the circumcenter lies inside the triangle
        if np.all(np.logical_and(circumcenter >= a, circumcenter <= c)):
            return circumcenter, radius
        # Otherwise, the minimum enclosing sphere is the circumcircle
        else:
            center = np.mean(triangle, axis=0)
            radius = np.max(np.linalg.norm(triangle - center, axis=1))
            return center, radius
    def _min_sphere(points, center, radius):
        if len(points) == 0 or len(center) == 3:
            if len(center) == 3:
                # c1, c2, c3 = center
                # return np.array([(c1 + c2 + c3) / 3]), 0
                return minimum_enclosing_sphere_3points(center)
            elif len(center) == 2:
                c1, c2 = center
                return (c1 + c2) / 2, np.linalg.norm(c1 - c2) / 2
            elif len(center) == 1:
                return center[0], 0
            else:
                return None, 0
        else:
            p = points[0]
            points = points[1:]
            c, r = _min_sphere(points, center, radius)
            if c is None or np.linalg.norm(p - c) > r:
                center.append(p)
                c, r = _min_sphere(points, center, radius)
                center.pop()
            return c, r

    if len(points) < 4:
        raise ValueError("At least 4 points are required.")
    np.random.shuffle(points)
    center, radius = _min_sphere(points, [], 0)
    return center, radius

def min_sphere(points):
    def _min_sphere(points, center, radius):
        if len(points) == 0 or len(center) == 3:
            if len(center) == 3:
                # c1, c2, c3 = center
                # return np.array([(c1 + c2 + c3) / 3]), 0
                return minimum_enclosing_sphere_3points(center)
            elif len(center) == 2:
                c1, c2 = center
                return (c1 + c2) / 2, np.linalg.norm(c1 - c2) / 2
            elif len(center) == 1:
                return center[0], 0
            else:
                return None, 0
        else:
            p = points[0]
            points = points[1:]
            c, r = _min_sphere(points, center, radius)
            if c is None or np.linalg.norm(p - c) > r:
                center.append(p)
                c, r = _min_sphere(points, center, radius)
                center.pop()
            return c, r

    if len(points) < 4:
        raise ValueError("At least 4 points are required.")
    np.random.shuffle(points)
    center, radius = _min_sphere(points, [], 0)
    print("Center:", center)
    print("Radius:", radius)
    return center, radius



def minimum_enclosing_sphere_3points(triangle):
    # Compute the circumcenter of the triangle
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ab_cross_ac = np.cross(ab, ac)
    ab_cross_ac_norm_sq = np.dot(ab_cross_ac, ab_cross_ac)
    if ab_cross_ac_norm_sq == 0:
        # Points are colinear, return a point and radius of infinity
        return a, np.inf
    ab_norm_sq = np.dot(ab, ab)
    ac_norm_sq = np.dot(ac, ac)
    circumcenter = a + (np.cross(ab_norm_sq * ac - ac_norm_sq * ab, ab_cross_ac) / (2 * ab_cross_ac_norm_sq))
    # Calculate the radius of the circumcircle
    radius = np.linalg.norm(circumcenter - a)
    # Check if the circumcenter lies inside the triangle
    if np.all(np.logical_and(circumcenter >= a, circumcenter <= c)):
        return circumcenter, radius
    # Otherwise, the minimum enclosing sphere is the circumcircle
    else:
        center = np.mean(triangle, axis=0)
        radius = np.max(np.linalg.norm(triangle - center, axis=1))
        return center, radius
def welzl2(points):
    if len(points) == 0 or len(points) > 4:
        return None, 0
    if len(points) == 1:
        return points[0], 0
    if len(points) == 2:
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[0] - center)
        return center, radius
    if len(points) == 3:
        return minimum_enclosing_sphere_3points(points)
    # Randomly select a point from the set
    p = random.choice(points)
    # Recursively compute the minimum enclosing sphere of the remaining points
    center, radius = welzl2([q for q in points if np.linalg.norm(q - p) > 0])
    # Check if the randomly selected point is inside the minimum enclosing sphere
    if np.linalg.norm(p - center) <= radius:
        return center, radius
    # If not, add the point to the set and compute the minimum enclosing sphere
    new_points = [q for q in points if np.linalg.norm(q - p) > radius]
    new_points.append(p)
    center, radius = welzl2(new_points)

    return center, radius


def fit_sphere_4points(array, tol=1e-6):
    # Check if the the points are co-linear
    D12 = array[1] - array[0]
    D12 = D12 / np.linalg.norm(D12)
    D13 = array[2] - array[0]
    D13 = D13 / np.linalg.norm(D13)
    D14 = array[3] - array[0]
    D14 = D14 / np.linalg.norm(D14)

    chk1 = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)  # 如果共线，chk1=1
    chk2 = np.clip(np.abs(np.dot(D12, D14)), 0., 1.)
    # 求的是反余弦值，如果是1，反余弦值为0（弧度），乘以180/pi，就是0（度），说明共线
    if np.arccos(chk1) / np.pi * 180 < tol or np.arccos(chk2) / np.pi * 180 < tol:
        R = np.inf
        C = np.full(3, np.nan)
        return R, C

    # Check if the the points are co-planar
    n1 = np.linalg.norm(np.cross(D12, D13))
    n2 = np.linalg.norm(np.cross(D12, D14))

    chk = np.clip(np.abs(np.dot(n1, n2)), 0., 1.)
    if np.arccos(chk) / np.pi * 180 < tol:
        R = np.inf
        C = np.full(3, np.nan)
        return R, C

    # Centroid of the sphere
    A = 2 * (array[1:] - np.full(len(array) - 1, array[0]))
    b = np.sum((np.square(array[1:]) - np.square(np.full(len(array) - 1, array[0]))), axis=1)
    C = np.transpose(np.linalg.solve(A, b))

    # Radius of the sphere
    R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))
    print("Center:", C)
    print("Radius:", R)
    print(type(R))

    return C, R


if __name__ == '__main__':

    # # Define the four points
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 4, 0])
    p3 = np.array([4, 0, 0])
    p4 = np.array([1, 2, 0])

    points1 = np.array([p1, p2, p3, p4])
    # points1 = np.array([[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
    #                     [3.4941856e+01, 0.0000000e+00, -3.9401650e-04],
    #                     [3.4963669e+01, 3.0126572e-03, -4.4547096e-03],
    #                     [3.9242031e+01, -2.7605583e+01, -3.7195000e+01]]
    #                    )
    #points1 = np.random.rand(4, 3)
    # show_tetrahedron(points1)
    center0, radius0 = min_sphere(points1)
    # center1, radius1 = fit_sphere_4points(points1)
    center2, radius2 = welzl2(points1)
    print("Center:", center2)
    print("Radius:", radius2)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='b')
    # plot the tetrahedron
    ax.plot(points1[:, 0], points1[:, 1], points1[:, 2], c='b')




    # Plot the sphere1
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = center0[0] + radius0 * np.cos(u) * np.sin(v)
    y = center0[1] + radius0 * np.sin(u) * np.sin(v)
    z = center0[2] + radius0 * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # Plot the sphere2
    # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    # x = center1[0] + radius1 * np.cos(u) * np.sin(v)
    # y = center1[1] + radius1 * np.sin(u) * np.sin(v)
    # z = center1[2] + radius1 * np.cos(v)
    # ax.plot_wireframe(x, y, z, color="g")

    # Plot the sphere3
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = center2[0] + radius2 * np.cos(u) * np.sin(v)
    y = center2[1] + radius2 * np.sin(u) * np.sin(v)
    z = center2[2] + radius2 * np.cos(v)
    ax.plot_wireframe(x, y, z, color="y")

    # Set the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # Show the plot
    print('Showing the plot...')
    plt.show()
