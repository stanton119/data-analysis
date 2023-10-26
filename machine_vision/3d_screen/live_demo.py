# %%
"""
# 3D screen test

Draw a square in front of another one.
Move the front square based on the position of the viewer's face.
Amount it moves depends on the depth between the squares.

Build with a graphics engine?
"""
# %%
# draw squares

from PIL import Image
import numpy as np


def blank_screen():
    shape = (720, 1280, 3)
    return np.ones(shape=shape, dtype=np.uint8) * 255


def draw_square(image, x_per=None, y_per=None, colour=None):
    if x_per is None:
        x_per = (0.4, 0.6)
    x = (int(x_per[0] * image.shape[1]), int(x_per[1] * image.shape[1]))

    if y_per is None:
        y_per = (0.4, 0.6)
    y = (int(y_per[0] * image.shape[0]), int(y_per[1] * image.shape[0]))

    if colour is None:
        colour = (255, 0, 0)

    for idx in range(3):
        image[y[0] : y[1], x[0] : x[1], idx] = colour[idx]
    return image


def get_square_position_from_landmarks(landmarks, depth=1.0):
    eye_left = landmarks[0]
    eye_right = landmarks[1]

    # average left/right eye positions
    x_avg = 0.5 * (eye_left.x + eye_right.x)
    y_avg = 0.5 * (eye_left.y + eye_right.y)

    x_size = 0.2 / 2 * depth
    y_size = 0.2 / 2 * depth

    x_pos = 0.5 - x_avg
    y_pos = 0.5 - y_avg
    x_per = (0.5 + x_pos * 1 / depth - x_size, 0.5 + x_pos * 1 / depth + x_size)
    y_per = (0.5 + y_pos * 1 / depth - y_size, 0.5 + y_pos * 1 / depth + y_size)

    # as z reduces, make square bigger
    # mid = 0.5, 0.5

    return x_per, y_per


image = blank_screen()
image = draw_square(image, x_per=(0.3, 0.7), y_per=(0.3, 0.7), colour=(0, 255, 0))
image = draw_square(image, x_per=(0.4, 0.6), y_per=(0.4, 0.6), colour=(255, 0, 0))
Image.fromarray(image)


# %%
# live demo
import cv2
import mediapipe as mp
import utilities
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

depths = [1, 0.8, 0.5, 0.4]
colours = np.random.randint(low=0, high=255, size=(len(depths), 3))

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            # convert fash mesh to squares
            landmarks = utilities.get_avg_landmarks_from_results(
                results, utilities.UNIQUE_EYE_COORDINATES
            )

            image = blank_screen()
            image = draw_square(
                image, x_per=(0.3, 0.7), y_per=(0.3, 0.7), colour=(0, 255, 0)
            )

            for depth, colour in zip(depths, colours):
                x_per, y_per = get_square_position_from_landmarks(
                    landmarks, depth=depth
                )
                image = draw_square(
                    image,
                    x_per=x_per,
                    y_per=y_per,
                    colour=colour,
                )

        cv2.imshow("3D Screen", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
