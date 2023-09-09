import cv2
import numpy as np
from PIL import ImageTk, Image
from keras.utils import img_to_array

def preprocess_image(image):
    # Convert the image to RGB format and rescale pixel values to the range [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    return image

def update_frame(cap, model, emotion_dict, video_label, window):
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        return
    # Load the pre-trained face detection cascade
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Preprocess the cropped region
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = preprocess_image(cropped_img)
        cropped_img = np.expand_dims(cropped_img, axis=0)

        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert the OpenCV image to PIL format
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image to fit the label
    img = img.resize((800, 600), Image.LANCZOS)

    # Convert the PIL image to Tkinter format
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the label with the new image
    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)

    # Schedule the next update
    window.after(10, update_frame, cap, model, emotion_dict, video_label, window)