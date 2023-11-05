
import os
import face_recognition
import pickle
from PIL import Image
from Augmentor import Pipeline

# Define the path to the folder containing student images
students_folder = 'student_images'

# Initialize a list to store face encodings and names
encodings = []
names = []

# Loop through each student folder
for student_folder in os.listdir(students_folder):
    student_name = student_folder
    student_folder_path = os.path.join(students_folder, student_folder)

    # Initialize an Augmentor pipeline
    p = Pipeline(student_folder_path)

    # Define augmentation operations (you can customize these as needed)
    p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)

    try:
        # Sample 10 augmented images per original image
        p.sample(10)
    except Exception as e:
        print(f"Error processing images for {student_name}: {e}")
        continue

    # Loop through the augmented images
    for augmented_image in os.listdir(student_folder_path + '/output'):
        image_path = os.path.join(student_folder_path, 'output', augmented_image)

        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                encodings.append(encoding)
                names.append(student_name)
            else:
                print(f"No face detected in {augmented_image} for {student_name}")
        except Exception as e:
            print(f"Error processing image {augmented_image} for {student_name}: {e}")

# Save the encodings and names
with open('encodings.pkl', 'wb') as f:
    pickle.dump((encodings, names), f)

# Provide a success message after the training
print("Training completed successfully.")
