import numpy as np

def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    # Compute translation vectors for each camera
    translation_vector1 = -camera_rotation1.T @ camera_position1
    translation_vector2 = -camera_rotation2.T @ camera_position2

    # Compute projection matrices for each camera
    projection_matrix1 = camera_matrix @ np.hstack((camera_rotation1.T, translation_vector1.reshape(3, 1)))
    projection_matrix2 = camera_matrix @ np.hstack((camera_rotation2.T, translation_vector2.reshape(3, 1)))

    # Number of points
    num_points = image_points1.shape[0]

    # Array to store the 3D points
    points_3d = np.zeros((num_points, 3))

    # For each pair of corresponding points
    for i in range(num_points):
        # Get the coordinates in homogeneous form
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]

        # Construct the system of linear equations
        A = np.array([
            (- x1 * projection_matrix1[2, :] + projection_matrix1[0, :]),
            (  y1 * projection_matrix1[2, :] - projection_matrix1[1, :]),
            (- x2 * projection_matrix2[2, :] + projection_matrix2[0, :]),
            (  y2 * projection_matrix2[2, :] - projection_matrix2[1, :])
        ])

        # Solve the system using least squares
        _, _, V = np.linalg.svd(A)
        print(V)
        X = V[-1]
        X /= X[-1]  # Convert to non-homogeneous coordinates

        # Store the result in the array
        points_3d[i] = X[:3]

    return points_3d
