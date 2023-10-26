import cv2
from PIL import Image
from pathlib import Path

save_folder = Path(__file__).parent / "face_pics"
save_folder.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)
n_img = 3
for idx in range(n_img + 1):
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    if idx == 0:
        # low brightness on first frame
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    # save to disk
    im = Image.fromarray(image)
    im.save(str(save_folder / f"{idx}.jpeg"))

    cv2.imshow("Captured image", image)
    if cv2.waitKey(500) & 0xFF == 27:
        break

cap.release()
