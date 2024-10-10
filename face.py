import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory to store the training images
TRAINING_DATA_DIR = 'training_images'

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        # Create start and stop buttons
        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition, bg='green', fg='white')
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Recognition", command=self.stop_recognition, bg='red', fg='white')
        self.stop_button.pack(pady=10)

        # Create button for face training
        self.train_button = tk.Button(root, text="Add New Face", command=self.train_new_face, bg='blue', fg='white')
        self.train_button.pack(pady=10)

        self.label = tk.Label(root)
        self.label.pack()

        self.video_capture = None
        self.running = False

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.video_capture = cv2.VideoCapture(0)
            self.detect_faces()

    def stop_recognition(self):
        self.running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        cv2.destroyAllWindows()

    def detect_faces(self):
        if self.running:
            ret, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            self.root.after(10, self.detect_faces)

    def train_new_face(self):
        person_name = simpledialog.askstring("Input", "Enter the name of the person:")

        if person_name:
            self.capture_images_for_training(person_name)
            messagebox.showinfo("Info", f"Training data captured for {person_name}. Now training the model...")
            self.train_model()

    def capture_images_for_training(self, person_name):
        if not os.path.exists(TRAINING_DATA_DIR):
            os.makedirs(TRAINING_DATA_DIR)

        person_dir = os.path.join(TRAINING_DATA_DIR, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        video_capture = cv2.VideoCapture(0)
        count = 0

        while count < 20:  # Capture 20 images
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                # Save the captured image to the training folder
                cv2.imwrite(f"{person_dir}/face_{count}.jpg", face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Capturing Training Data', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if count >= 20:  # Stop after capturing 20 images
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        labels = []
        label_count = 0

        for person_name in os.listdir(TRAINING_DATA_DIR):
            person_dir = os.path.join(TRAINING_DATA_DIR, person_name)
            for image_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_count)
            label_count += 1

        # Train the recognizer and save the model
        recognizer.train(faces, np.array(labels))
        recognizer.save('training_data.yml')

        messagebox.showinfo("Info", "Training completed and model saved!")


# Initialize Tkinter root
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()
