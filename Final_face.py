import cv2
import face_recognition

# to open or acess webcam
cap = cv2.VideoCapture(0)

# To load trainning image
img1 = face_recognition.load_image_file('Sandesh.jpg')

# To change bgr format to rgb
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# to loacte the face present in the image
face =face_recognition.face_locations(img1)[0]

#for encoding the face
encodeFace = face_recognition.face_encodings(img1)[0]

# to create bounding box
cv2.rectangle(img1, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), 5)

while True:
    ret, frame1 = cap.read()
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame = frame1[:, :,::-1]
    # to find the location face on testing image
    faceTest = face_recognition.face_locations(frame)[0]

    # encoding of testion image
    encodefaceTest = face_recognition.face_encodings(frame)[0]

    #to create bounding boxes on testing face
    cv2.rectangle(frame1, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (0, 255, 0), 5)

    # loading of testing image
    img2 = face_recognition.load_image_file('test1.jpg')

    # comparision between trained face and testing face
    result = face_recognition.compare_faces([encodeFace], encodefaceTest)
    #distance between the trained face and testing face
    face_distance = face_recognition.face_distance([encodeFace], encodefaceTest)
    print(type(result))

    if (result[0] == True):
        cv2.putText(frame1, "Sandesh", (faceTest[3], faceTest[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(frame1, "Unknown", (faceTest[3], faceTest[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("test", frame1)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

