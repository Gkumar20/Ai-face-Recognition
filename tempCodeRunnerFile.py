

# # face recognization from the photo 

# import cv2
# import face_recognition
# import pickle

# # Load the face encodings and names
# with open('encodings.pkl', 'rb') as f:
#     encodings, names = pickle.load(f)

# # Define the path to the group photo
# group_photo_path = 'test_images/gp4.jpg'

# # Load the group photo
# group_photo = face_recognition.load_image_file(group_photo_path)

# # Find all face locations and face encodings in the group photo
# face_locations = face_recognition.face_locations(group_photo)
# face_encodings = face_recognition.face_encodings(group_photo, face_locations)

# # Initialize a set to keep track of recorded students
# recorded_students = set()

# for encoding in face_encodings:
#     matches = face_recognition.compare_faces(encodings, encoding)

#     for i, match in enumerate(matches):
#         if match:
#             student_name = names[i]

#             # Check if attendance has already been recorded for this student
#             if student_name not in recorded_students:
#                 recorded_students.add(student_name)
#                 print(f'Attendance recorded for {student_name}')

# # Release any resources if needed (not necessary for reading a photo)
# cv2.destroyAllWindows()
