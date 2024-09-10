import cv2 # OpenCV for image processing
import mediapipe # MediaPipe for face landmark detection
import pyautogui # PyAutoGUI for controlling the mouse and keyboard

# Initializing the FaceMesh model from MediaPipe with refined landmarks
face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Capture video from the default camera (index 0)
cam = cv2.VideoCapture(0)

# Getting the screen width and height for mouse movement calculations
screen_w, screen_h= pyautogui.size()

# An infinite loop to continuously capture video frames
while True:
    
    # Read a frame from the camera
    _, image = cam.read()
    
    # Flipping the image horizontally to create a mirror effect
    image = cv2.flip(image,1)

    # Getting the dimensions of the captured image
    window_h, window_w, _ = image.shape

    # Converting the image from BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processing the RGB image to find face landmarks
    processed_image = face_mesh_landmarks.process(rgb_image)

    # Extracting all detected face landmark points
    all_face_landmark_points = processed_image.multi_face_landmarks

    # Checking if any face landmarks were detected
    if all_face_landmark_points:
        one_face_landmark_points = all_face_landmark_points[0].landmark

        # Loop through specific landmark points (474 to 477 (eye landmarks)) for mouse control
        for id, landmark_point in enumerate(one_face_landmark_points[474:478]):
            # Calculating the x and y coordinates of the landmark point
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            
            # If the current landmark is the second one (id == 1), move the mouse
            if id==1:
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h / window_h * y)
                # Moving the mouse to the calculated position
                pyautogui.moveTo(mouse_x, mouse_y)

            # Draw a circle on the eye's landmark position for visualization
            cv2.circle(image, (x, y), 3, (0, 0, 255))
            
        # Defines the left eye landmarks and moves the mouse based on its movement using specific indices
        left_eye = [one_face_landmark_points[145],one_face_landmark_points[159]]

        # Loop through the left eye landmarks to draw them on the image
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            # print(x,y)
            cv2.circle(image, (x, y), 3, (0, 255, 255))
            
        # Checks if the y-coordinates of the left eye landmarks are close enough (when the user blinks)
        if(left_eye[0].y - left_eye[1].y<0.01):
            # If the user blinks a click is performed
            pyautogui.click()
            # Pause for 2 seconds after the click
            pyautogui.sleep(2)
            print("Mouse Clicked")
    cv2.imshow("Eye controlled mouse", image)

    # Waits for 100 milliseconds after a key press then runs the while loop
    key = cv2.waitKey(100)

    # ASCII value of the Escape (Esc) key
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()


