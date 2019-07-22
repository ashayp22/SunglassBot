import cv2, json
import numpy as np


#load cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def getExtension():
    with open('info.json') as json_file:
        data = json.load(json_file)
        return data["data"]["extension"]


extension = getExtension()

if extension == "jpg": #so the file is found
    extension = "jpeg"


face = cv2.imread('images/media.' + extension,-1)
glasses = cv2.imread('best.png',cv2.IMREAD_UNCHANGED)


gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #gets the parts of the image that contains a face


for(x,y,w,h) in faces:
    #print("face: " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
    # cv2.rectangle(face, (x,y), (x+w, y+h), (255,0,0), 2) #draw a rectangle
    roi_gray = gray[y:y+h, x:x+w] #gets the region
    roi_color = face[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    top = 100000000
    bottom = 0
    left = 100000000
    right = 0

    num_eyes = 0

    for(ex,ey,ew,eh) in eyes:
        # cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        print("eyes: " + str(ex) + " " + str(ey) + " " + str(ew) + " " + str(eh))
        if ex + x < left:
            left = ex + x
        if ex + ew + x > right:
            right = ex + ew + x
        if ey + y< top:
            top = ey + y
        if ey + eh + y > bottom:
            bottom = ey + eh + y
        num_eyes += 1


    if top == 100000000 or bottom == 0 or left == 100000000 or right == 0 or num_eyes < 2:
        continue

    #cv2.rectangle(face, (left,top), (right, bottom), (0,0,255), 2)

    #resize the right, left, top and bottom boundaries

    size_factor = 0.2
    x_offset = int((right - left) * 0.075)


    #add onto the right, top and bottom
    right += (right - left) * size_factor + x_offset
    left -= (right - left) * size_factor + x_offset
    top -= (bottom - top) * size_factor
    bottom += (bottom - top) * size_factor

    right = int(right)
    left = int(left)
    top = int(top)
    bottom = int(bottom)

    # cv2.rectangle(face, (left,top), (right, bottom), (0,0,255), 2)


    x_factor = ((right-left) / glasses.shape[1])
    y_factor = ((bottom-top) / glasses.shape[0])


    glasses = cv2.resize(glasses, (0,0), fx=x_factor, fy=y_factor)

    #now we place the image on top

    alpha_s = glasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        face[top:bottom, left+x_offset:right+x_offset, c] = (alpha_s * glasses[:, :, c] +
                                  alpha_l * face[top:bottom, left+x_offset:right+x_offset, c])



cv2.imwrite("images/media." + extension, face)
