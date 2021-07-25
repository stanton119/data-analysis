import numpy as np


def get_unique_eye_coordinates():
    # left, right
    eye_coordinates = [
        [
            (263, 249),
            (249, 390),
            (390, 373),
            (373, 374),
            (374, 380),
            (380, 381),
            (381, 382),
            (382, 362),
            (263, 466),
            (466, 388),
            (388, 387),
            (387, 386),
            (386, 385),
            (385, 384),
            (384, 398),
            (398, 362),
        ],
        [
            (33, 7),
            (7, 163),
            (163, 144),
            (144, 145),
            (145, 153),
            (153, 154),
            (154, 155),
            (155, 133),
            (33, 246),
            (246, 161),
            (161, 160),
            (160, 159),
            (159, 158),
            (158, 157),
            (157, 173),
            (173, 133),
        ],
    ]
    unique_eye_coordinates = list()
    for eye in eye_coordinates:
        unique_coordinates = list()
        for point in eye:
            unique_coordinates += [point[0], point[1]]
        unique_eye_coordinates.append(list(set(unique_coordinates)))
    return unique_eye_coordinates


UNIQUE_EYE_COORDINATES = get_unique_eye_coordinates()


def get_avg_eye_coordinates(points, unique_eye_coordinates):
    avg_coordinates = list()
    for unique_coordinates in unique_eye_coordinates:
        _avg_coordinates = list()
        for point in unique_coordinates:
            _avg_coordinates.append(
                [points[point].x, points[point].y, points[point].z]
            )
        _avg_coordinates = np.array(_avg_coordinates)
        _avg_coordinates = np.mean(_avg_coordinates, axis=0)
        avg_coordinates.append(_avg_coordinates)
    return avg_coordinates


from collections import namedtuple


def get_landmarks_from_np(coordinates):
    LandmarkPoint = namedtuple("Point", ["x", "y", "z"])
    landmarks = list()
    for coordinate in coordinates:
        landmarks.append(
            LandmarkPoint(coordinate[0], coordinate[1], coordinate[2])
        )
    return landmarks


def get_avg_landmarks_from_results(results, unique_eye_coordinates):
    # get 1st face only
    points = results.multi_face_landmarks[0].landmark
    avg_eye_coordinates = get_avg_eye_coordinates(
        points, unique_eye_coordinates
    )
    landmarks = get_landmarks_from_np(avg_eye_coordinates)
    return landmarks


def add_landmarks_to_image(image, landmarks, area=10):
    for landmark in landmarks:
        c_x = landmark.x * image.shape[1]
        c_y = landmark.y * image.shape[0]
        c_x = (int(np.round(c_x)) - area, int(np.round(c_x)) + area)
        c_y = (int(np.round(c_y)) - area, int(np.round(c_y)) + area)
        for i in range(3):
            image[c_y[0] : c_y[1], c_x[0] : c_x[1], i] = (
                255 - (255 - image[c_y[0] : c_y[1], c_x[0] : c_x[1], i]) * 0.8
            )
    return image


def get_eye_distance(landmarks):
    points = np.array(landmarks)
    return np.sqrt(np.sum(np.power(points[0] - points[1], 2)))


def add_distance_to_image(image, distance, scale=5, area=15):
    scaled_distance = int(np.round(distance * image.shape[1] * scale))
    for i in range(3):
        image[0:area, 0:scaled_distance, i] = 255
    return image
