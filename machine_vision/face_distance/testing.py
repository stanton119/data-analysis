# %%
# live demo
import mediapipe as mp
from PIL import Image
import numpy as np
import utilities

from pathlib import Path
image = Image.open(Path(__file__).parent / "face_orig.jpeg")
image = np.array(image)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        if results.multi_face_landmarks:
            landmarks = utilities.get_avg_landmarks_from_results(
                results, utilities.UNIQUE_EYE_COORDINATES
            )
            image = utilities.add_landmarks_to_image(image, landmarks)
            distance = utilities.get_eye_distance(landmarks)
            image = utilities.add_distance_to_image(image, distance)


image = Image.fromarray(image)
image.save(Path(__file__).parent / "face_proc.jpeg")
