import cv2

cap = cv2.VideoCapture("Files/Videos/1.mp4")
posList = []
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
            posList.append(center_points[0]["center"])
        for pos in posList:
            cv2.circle(frame_copy, pos, 8, (0,255,0), cv2.FILLED)
        frame_copy = cv2.resize(frame_copy, (0,0), None, 0.6, 0.61)
        cv2.imshow("Video", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    else:
        break