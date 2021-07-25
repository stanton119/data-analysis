# %%
# live demo
import cv2
import mediapipe as mp
import utilities

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


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
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            landmarks = utilities.get_avg_landmarks_from_results(
                results, utilities.UNIQUE_EYE_COORDINATES
            )
            image = utilities.add_landmarks_to_image(image, landmarks)
            distance = utilities.get_eye_distance(landmarks)
            image = utilities.add_distance_to_image(image, distance)

        cv2.imshow("MediaPipe FaceMesh", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        results = face_mesh.process(image)

cap.release()
