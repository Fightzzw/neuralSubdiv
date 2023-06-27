import numpy as np
import matplotlib.pyplot as plt


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
    print("Center:", center)
    print("Radius:", radius)
    return center, radius


def fit_circumscribed_sphere_4points(array, tol=1e-6):
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

    return C, R


if __name__ == '__main__':

    # # Define the four points
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 4, 0])
    p3 = np.array([4, 0, 0])
    p4 = np.array([1, 2, 0])

    points1 = np.array([p1, p2, p3, p4])

    points1 = np.random.rand(4, 3)
    # show_tetrahedron(points1)
    center0, radius0 = fit_circumscribed_sphere_4points(points1)

    center1, radius1 = get_min_sphere_4points(points1)


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
    ax.plot_wireframe(x, y, z, color="g")

    # Plot the sphere2
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = center1[0] + radius1 * np.cos(u) * np.sin(v)
    y = center1[1] + radius1 * np.sin(u) * np.sin(v)
    z = center1[2] + radius1 * np.cos(v)
    ax.plot_wireframe(x, y, z, color="y")

    # Set the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # Show the plot
    print('Showing the plot...')
    plt.show()