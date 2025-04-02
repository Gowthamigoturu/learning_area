import cv2
import face_recognition
import numpy as np

# load known face encodings and names
known_face_encodings = []
known_face_names = []

# load known faces and their names here
known_person1_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person1.jpg")
known_person2_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person2.jpg")
known_person3_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person3.jpg")
known_person4_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person4.jpg")
known_person5_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person5.jpg")
known_person7_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person7.jpg")
known_person8_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person8.jpg")
known_person9_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person9.jpg")
known_person10_image = face_recognition.load_image_file("C:\\Users\\DELL\\Pictures\\person10.jpg")

known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding = face_recognition.face_encodings(known_person4_image)[0]
known_person5_encoding = face_recognition.face_encodings(known_person5_image)[0]
known_person7_encoding = face_recognition.face_encodings(known_person7_image)[0]
known_person8_encoding = face_recognition.face_encodings(known_person8_image)[0]
known_person9_encoding = face_recognition.face_encodings(known_person9_image)[0]
known_person10_encoding = face_recognition.face_encodings(known_person10_image)[0]

known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)
known_face_encodings.append(known_person4_encoding)
known_face_encodings.append(known_person5_encoding)
known_face_encodings.append(known_person7_encoding)
known_face_encodings.append(known_person8_encoding)
known_face_encodings.append(known_person9_encoding)
known_face_encodings.append(known_person10_encoding)

known_face_names.append("Gowthami ")
known_face_names.append("Bhargavi mam")
known_face_names.append("Swathi mam")
known_face_names.append("Harinath sir")
known_face_names.append("Poojitha")
known_face_names.append("Bhargavi")
known_face_names.append("Hasini")
known_face_names.append("Sneha")
known_face_names.append("Meghana")

#initialize webcam
Video_Capture = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    ret, frame = Video_Capture.read()
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # loop through each face found in the frame
    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom,left =face_location
        # check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding) 
        name = "unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
        # Draw a box around the face and label with name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # break 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
Video_Capture.release()
cv2.destroyAllWindows()
