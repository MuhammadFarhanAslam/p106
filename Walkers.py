import cv2


# Create our body classifier
# not working => body_classifier = "C:/Users/farha/OneDrive/Desktop/PY/P106/haarcascade_fullbody.xml"
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while (True):
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    faces = body_classifier.detectMultiScale(grey, 1.1, 5)
    
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow("detecting", frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
