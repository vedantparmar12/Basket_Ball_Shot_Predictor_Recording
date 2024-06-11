import cv2
import numpy as np
import math
cap = cv2.VideoCapture("Files/Videos/2.mp4")
posListX = []
posListY = []

XList = [item for item in range(0, 1300)]
def get_contours(mask, original_copy, minarea=700):
    center_points = []
    contours, hirearchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Length of the Contours", len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area> minarea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y,w, h = cv2.boundingRect(approx)
            cx, cy = x+(w//2), y+(h//2)
            #cv2.rectangle(original_copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.circle(original_copy, (cx, cy), 5, (0,0,255), cv2.FILLED)

            #cv2.drawContours(original_copy, cnt, -1, (255, 0,0), 3)
            center_points.append({"area":area, "center":[cx,cy]})
    center_points = sorted(center_points, key=lambda x:x["area"], reverse=True)
    return original_copy, center_points

while True:
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
        # Step 1: Covert the Image from BGR to HSV Color Space
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ball Detection
        lower_range = (7, 153, 29)
        upper_range = (162, 255, 255)

        mask = cv2.inRange(frameHSV, lower_range, upper_range)

        ball_detection = cv2.bitwise_and(frame, frame, mask=mask)

        frame_copy, center_points=get_contours(mask, frame_copy, minarea=700)
        if center_points:
            posListX.append(center_points[0]["center"][0])
            posListY.append(center_points[0]["center"][1])

        if posListX:
            A, B, C = np.polyfit(posListX, posListY, 2)
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos=(posX, posY)
                cv2.circle(frame_copy, pos, 8, (0,255,0), cv2.FILLED)
            for x in XList:
                y = int(A*x**2+B*x+C)
                cv2.circle(frame_copy, (x,y), 2, (0,0,0), cv2.FILLED)
            if len(posListX) <10:
                a = A
                b = B
                c = C - 590

                x = int((-b - math.sqrt(b ** 2 - (4*a*c))) / (2 * a))
                prediction = 330 < x < 430
            if prediction:
                IN = "BASKET"
                pos = (50, 150)
                offset = 10
                (w, h), _ = cv2.getTextSize(IN, cv2.FONT_HERSHEY_PLAIN, 3, thickness=3)
                x1, y1, x2, y2 = pos[0] - offset, pos[1] + offset, pos[0] + w + offset, pos[1] - h - offset
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), -1)
                cv2.putText(frame_copy, IN, (pos[0], pos[1]), cv2.FONT_HERSHEY_PLAIN, 3, [255, 255, 255], thickness=3)
            else:
                OUT = "NO BASKET"
                pos = (50, 150)
                offset = 10
                (w, h), _ = cv2.getTextSize(OUT, cv2.FONT_HERSHEY_PLAIN, 3, thickness=3)
                x1, y1, x2, y2 = pos[0] - offset, pos[1] + offset, pos[0] + w + offset, pos[1] - h - offset
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), -1)
                cv2.putText(frame_copy, OUT, (pos[0], pos[1]), cv2.FONT_HERSHEY_PLAIN, 3, [255, 255, 255], thickness=3)


        frame_copy = cv2.resize(frame_copy, (0,0), None, 0.6, 0.61)
        cv2.imshow("Video", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    else:
        break