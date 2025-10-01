import cv2
import numpy as np

cap = cv2.VideoCapture("Traffic_input.mp4")
if not cap.isOpened():
    print("Error: could not open video")
    exit()

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, th = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # reset counter for each frame
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 800:  # filter noise
            count += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # overlay the counter on screen
    cv2.putText(frame, f"Cars: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Traffic Detection", frame)
    prev_gray = gray

    # Press ESC to quit
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
