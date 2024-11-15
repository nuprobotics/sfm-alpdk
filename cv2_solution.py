import numpy as np
import cv2
import typing
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml


# Task 2
def get_matches(image1, image2) -> typing.Tuple[
    typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.DMatch]]:
    sift = cv2.SIFT_create()
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kp1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()

    matches_1_to_2: typing.Sequence[typing.Sequence[cv2.DMatch]] = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches_1_to_2 = []
    for m, n in matches_1_to_2:
        if m.distance < 0.75 * n.distance:
            good_matches_1_to_2.append(m)

    matches_2_to_1 = bf.knnMatch(descriptors2, descriptors1, k=2)
    good_matches_2_to_1 = []
    for m, n in matches_2_to_1:
        if m.distance < 0.75 * n.distance:
            good_matches_2_to_1.append(m)

    final_matches = []
    for match1 in good_matches_1_to_2:
        for match2 in good_matches_2_to_1:
            if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                final_matches.append(match1)
                break

    return kp1, kp2, final_matches


def get_second_camera_position(kp1, kp2, matches, camera_matrix):
    coordinates1 = np.array([kp1[match.queryIdx].pt for match in matches])
    coordinates2 = np.array([kp2[match.trainIdx].pt for match in matches])
    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, camera_matrix)
    _, R, t, mask = cv2.recoverPose(E, coordinates1, coordinates2, camera_matrix)
    return R, t, E


# Task 3
def triangulation(
        camera_matrix: np.ndarray,
        camera1_translation_vector: np.ndarray,
        camera1_rotation_matrix: np.ndarray,
        camera2_translation_vector: np.ndarray,
        camera2_rotation_matrix: np.ndarray,
        kp1: typing.Sequence[cv2.KeyPoint],
        kp2: typing.Sequence[cv2.KeyPoint],
        matches: typing.Sequence[cv2.DMatch]
):
    # Compute projection matrices for each camera
    projection_matrix1 = camera_matrix @ np.hstack((camera1_rotation_matrix, camera1_translation_vector.reshape(3, 1)))
    projection_matrix2 = camera_matrix @ np.hstack((camera2_rotation_matrix, camera2_translation_vector.reshape(3, 1)))

    points1 = np.array([kp1[match.queryIdx].pt for match in matches], dtype=np.float32)
    points2 = np.array([kp2[match.trainIdx].pt for match in matches], dtype=np.float32)

    # Number of points
    num_points = points1.shape[0]

    # Array to store the 3D points
    points_3d = np.zeros((num_points, 3))

    # For each pair of corresponding points
    for i in range(num_points):
        # Get the coordinates in homogeneous form
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        # Construct the system of linear equations
        A = np.array([
            (- x1 * projection_matrix1[2, :] + projection_matrix1[0, :]),
            (y1 * projection_matrix1[2, :] - projection_matrix1[1, :]),
            (- x2 * projection_matrix2[2, :] + projection_matrix2[0, :]),
            (y2 * projection_matrix2[2, :] - projection_matrix2[1, :])
        ])

        # Solve the system using least squares
        _, _, V = np.linalg.svd(A)
        print(V)
        X = V[-1]
        X /= X[-1]  # Convert to non-homogeneous coordinates

        # Store the result in the array
        points_3d[i] = X[:3]

    return points_3d


# Task 4
def resection(
        image1: np.ndarray,
        image2: np.ndarray,
        camera_matrix: np.ndarray,
        matches: typing.Sequence[cv2.DMatch],
        points_3d: np.ndarray
):
    kp1, kp2, matches2 = get_matches(image1, image2)

    final_matches = []
    final_points_3d = []

    i = 0

    for match2 in matches2:
        if any(match2.queryIdx == match.queryIdx for match in matches):
            final_matches.append(match2)
            final_points_3d.append(points_3d[i])
        i += 1

    final_points_3d = np.array(final_points_3d)

    # for match in final_matches:
    #     print(f"Query Index: {match.queryIdx}, Train Index: {match.trainIdx}, Distance: {match.distance}")

    # Collect the 2D coordinates from both images
    # points_2d_image1 = np.array([kp1[match.queryIdx].pt for match in final_matches], dtype=np.float32)
    points_2d_image2 = np.array([kp2[match.trainIdx].pt for match in final_matches], dtype=np.float32)

    # print(points_2d_image1.shape)
    # print(points_2d_image2.shape)
    # print(final_points_3d.shape)

    A = []
    for i in range(len(points_2d_image2)):
        X, Y, Z = final_points_3d[i]
        # x1, y1 = points_2d_image1[i]
        x2, y2 = points_2d_image2[i]

        # A.append([X, Y, Z, 1,  0,  0,  0,  0, -x1 * X, -x1 * Y, -x1 * Z, -x1])
        # A.append([0, 0, 0, 0, -X, -Y, -Z, -1,  y1 * X,  y1 * Y,  y1 * Z,  y1])

        A.append([X, Y, Z, 1,  0,  0,  0,  0, -x2 * X, -x2 * Y, -x2 * Z, -x2])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1,  y2 * X,  y2 * Y,  y2 * Z,  y2])

    A = np.array(A)

    _, _, VT = np.linalg.svd(A)

    P = VT[-1].reshape(3, 4)

    M = np.dot(np.linalg.inv(camera_matrix), P)

    R = M[:, :3]

    t = M[:, 3]

    t = t.reshape((3, 1))

    R = R.T
    t = -np.dot(np.linalg.inv(R.T), t)

    print(R)
    print(t)

    return R, t

def convert_to_world_frame(translation_vector, rotation_matrix):
    world_rotation = rotation_matrix.T
    world_position = -np.dot(world_rotation, translation_vector)

    return world_position, world_rotation


def visualisation(
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        camera_position3: np.ndarray,
        camera_rotation3: np.ndarray,
):

    def plot_camera(ax, position, direction, label):
        color_scatter = 'blue' if label != 'Camera 3' else 'green'
        # print(position)
        ax.scatter(position[0][0], position[1][0], position[2][0], color=color_scatter, s=100)
        color_quiver = 'red' if label != 'Camera 3' else 'magenta'

        ax.quiver(position[0][0], position[1][0], position[2][0], direction[0], direction[1], direction[2],
                  length=1, color=color_quiver, arrow_length_ratio=0.2)
        ax.text(position[0][0], position[1][0], position[2][0], label, color='black')


    camera_positions = [camera_position1, camera_position2, camera_position3]
    camera_directions = [camera_rotation1[:, 2], camera_rotation2[:, 2], camera_rotation3[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_camera(ax, camera_positions[0], camera_directions[0], 'Camera 1')
    plot_camera(ax, camera_positions[1], camera_directions[1], 'Camera 2')
    plot_camera(ax, camera_positions[2], camera_directions[2], 'Camera 3')

    initial_elev = 0
    initial_azim = 270

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=initial_elev, azim=initial_azim)

    ax.set_xlim([-1.50, 2.0])
    ax.set_ylim([-.50, 3.0])
    ax.set_zlim([-.50, 3.0])

    ax_elev_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    elev_slider = Slider(ax_elev_slider, 'Elev', 0, 360, valinit=initial_elev)

    ax_azim_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    azim_slider = Slider(ax_azim_slider, 'Azim', 0, 360, valinit=initial_azim)


    def update(val):
        elev = elev_slider.val
        azim = azim_slider.val
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)

    plt.show()


def main():
    image1 = cv2.imread('./images/image0.jpg')
    image2 = cv2.imread('./images/image1.jpg')
    image3 = cv2.imread('./images/image2.jpg')
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')

    key_points1, key_points2, matches_1_to_2 = get_matches(image1, image2)
    R2, t2, E = get_second_camera_position(key_points1, key_points2, matches_1_to_2, camera_matrix)
    triangulated_points = triangulation(
        camera_matrix,
        np.array([0, 0, 0]).reshape((3,1)),
        np.eye(3),
        t2,
        R2,
        key_points1,
        key_points2,
        matches_1_to_2
    )

    R3, t3 = resection(image1, image3, camera_matrix, matches_1_to_2, triangulated_points)
    camera_position1, camera_rotation1 = convert_to_world_frame(np.array([0, 0, 0]).reshape((3,1)), np.eye(3))
    camera_position2, camera_rotation2 = convert_to_world_frame(t2, R2)
    camera_position3, camera_rotation3 = convert_to_world_frame(t3, R3)
    visualisation(
        camera_position1,
        camera_rotation1,
        camera_position2,
        camera_rotation2,
        camera_position3,
        camera_rotation3
    )

if __name__ == "__main__":
    main()
