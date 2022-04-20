import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        # nameList = []
        # dayStringList = []
        # subjectList = []
        other_data = set()

        for line in myDataList:
            entry = line.strip().split(',')
            other_data.add((entry[0], entry[2], entry[3]))

            # f.writeline(...)
        # for line in myDataList:
        #     entry = line.split(',')
        #     nameList.append(entry[0])
        #     dayStringList.append(entry[2])
        #     subjectList.append(entry[3])

        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dayString = now.strftime("%A")
        if ((dtString >= "10:00:00") and (dtString <= "11:30:00")):
            subject = "Maths"
        elif ((dtString >= "14:00:00") and (dtString <= "19:00:00")):
            subject = "English"
        # print(f'\n{name},{dtString},{dayString},{subject}')
        # print(nameList, dayStringList, subjectList)
        # if ((name not in nameList) and (dayString not in dayStringList) and (subject not in subjectList)):
        #     f.writelines(f'\n{name},{dtString},{dayString},{subject}')

        current_data = (name, dayString, subject)
        # print(other_data)
        if current_data not in other_data:
            f.writelines(f'\n{name},{dtString},{dayString},{subject}')


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
#
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(100)
