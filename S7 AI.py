import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
abhiram_image = face_recognition.load_image_file("Students/Abhiram_G_Krishnan.jpg")
abhiram_encoding = face_recognition.face_encodings(abhiram_image)[0]

abhishek_image = face_recognition.load_image_file("Students/Abhishek_C.jpg")
abhishek_encoding = face_recognition.face_encodings(abhishek_image)[0]

adithya_krishna_image = face_recognition.load_image_file("Students/Adithya_Krishna.jpg")
adithya_krishna_encoding = face_recognition.face_encodings(adithya_krishna_image)[0]

ABC_image = face_recognition.load_image_file("Students/ABC.jpg")
ABC_encoding = face_recognition.face_encodings(ABC_image)[0]

adith_selvan_image = face_recognition.load_image_file("Students/Adith_Selvan.jpg")
adith_selvan_encoding = face_recognition.face_encodings(adith_selvan_image)[0]

aiswarya_image = face_recognition.load_image_file("Students/Aiswarya_A.jpg")
aiswarya_encoding = face_recognition.face_encodings(aiswarya_image)[0]

akhilesh_image = face_recognition.load_image_file("Students/Akhilesh_Krishnan.jpg")
akhilesh_encoding = face_recognition.face_encodings(akhilesh_image)[0]

akil_abdul_image = face_recognition.load_image_file("Students/Akil_Abdul.jpg")
akil_abdul_encoding = face_recognition.face_encodings(akil_abdul_image)[0]

anupama_image = face_recognition.load_image_file("Students/Anupama_A.jpg")
anupama_encoding = face_recognition.face_encodings(anupama_image)[0]

dayal_image = face_recognition.load_image_file("Students/Dayal_Naron.jpg")
dayal_encoding = face_recognition.face_encodings(dayal_image)[0]

joel_image = face_recognition.load_image_file("Students/Joel.jpg")
joel_encoding = face_recognition.face_encodings(joel_image)[0]

kasinath_a_image = face_recognition.load_image_file("Students/Kasinath_A.jpg")
kasinath_a_encoding = face_recognition.face_encodings(kasinath_a_image)[0]

kasinath_ms_image = face_recognition.load_image_file("Students/Kasinath_M_S.jpg")
kasinath_ms_encoding = face_recognition.face_encodings(kasinath_ms_image)[0]

nandana_image = face_recognition.load_image_file("Students/Nandana.jpg")
nandana_encoding = face_recognition.face_encodings(nandana_image)[0]

pranav_image = face_recognition.load_image_file("Students/Pranav_Rajesh.jpg")
pranav_encoding = face_recognition.face_encodings(pranav_image)[0]

priyanka_image = face_recognition.load_image_file("Students/Priyanka_Rajeev.jpg")
priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]

sreelakshmi_image = face_recognition.load_image_file("Students/R_Sreelakshmi.jpg")
sreelakshmi_encoding = face_recognition.face_encodings(sreelakshmi_image)[0]

rithik_image = face_recognition.load_image_file("Students/Rithik_Keshav.jpg")
rithik_encoding = face_recognition.face_encodings(rithik_image)[0]

sulekha_image = face_recognition.load_image_file("Students/Sulekha_R.jpg")
sulekha_encoding = face_recognition.face_encodings(sulekha_image)[0]

vaibhav_image = face_recognition.load_image_file("Students/Vaibhav_Nair.jpg")
vaibhav_encoding = face_recognition.face_encodings(vaibhav_image)[0]

vinayak_m_image = face_recognition.load_image_file("Students/Vinayak_Madhu.jpg")
vinayak_m_encoding = face_recognition.face_encodings(vinayak_m_image)[0]

vinayak_t_image = face_recognition.load_image_file("Students/Vinayak_T_Nair.jpg")
vinayak_t_encoding = face_recognition.face_encodings(vinayak_t_image)[0]

yadhukrishnan_image = face_recognition.load_image_file("Students/Yadhukrishnan_M.jpg")
yadhukrishnan_encoding = face_recognition.face_encodings(yadhukrishnan_image)[0]

yedukrishnan_image = face_recognition.load_image_file("Students/Yedukrishnan_H.jpg")
yedukrishnan_encoding = face_recognition.face_encodings(yedukrishnan_image)[0]

known_face_encodings = [
    abhiram_encoding, abhishek_encoding, adithya_krishna_encoding, ABC_encoding, 
    adith_selvan_encoding, aiswarya_encoding, akhilesh_encoding, akil_abdul_encoding, 
    anupama_encoding, dayal_encoding, joel_encoding, kasinath_a_encoding, kasinath_ms_encoding, 
    nandana_encoding, pranav_encoding, priyanka_encoding, sreelakshmi_encoding, rithik_encoding, 
    sulekha_encoding, vaibhav_encoding, vinayak_m_encoding, vinayak_t_encoding, yadhukrishnan_encoding, 
    yedukrishnan_encoding
]

known_face_names = [
    "Abhiram G Krishnan", "Abhishek C", "Adithya Krishna", "Adithya G", "Adith Selvan", 
    "Aiswarya A", "Akhilesh Krishnan", "Akil Abdul", "Anupama A", "Dayal Naron", "Joel", 
    "Kasinath A", "Kasinath M S", "Nandana", "Pranav Rajesh", "Priyanka Rajeev", 
    "R Sreelakshmi", "Rithik Keshav", "Sulekha R", "Vaibhav Nair", "Vinayak Madhu", 
    "Vinayak T Nair", "Yadhukrishnan M", "Yedukrishnan H"
]

# List of expected students

students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 
    
    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()
