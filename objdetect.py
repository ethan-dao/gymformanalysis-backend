import numpy as np
import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables
successful_count = 0
start_reached = False
pullup_reached = False
top_reached = False
descent_reached = False
initial_shoulder_position = None
# Testing purposes
start_message_shown = False
pullup_message_shown = False
top_message_shown = False
descent_message_shown = False


# 1. Labeling our video frames
def is_successful_pullup(landmarks):

    global start_reached, pullup_reached, top_reached, descent_reached, successful_count, start_message_shown, pullup_message_shown, top_message_shown, descent_message_shown, initial_shoulder_position, phase_list

    # Extract relevant landmarks: wrist, elbow, shoulder positions
    wrist_y = landmarks['wrist'][1]
    elbow_angle = calculate_angle(landmarks['elbow'], landmarks['shoulder'], landmarks['wrist'])
    shoulder_angle = calculate_angle(landmarks['elbow'], landmarks['shoulder'], landmarks['hip'])
    shoulder_y = landmarks['shoulder'][1]

    # Define thresholds
    threshold_bent_angle = 70
    current_phase = 0

    # Phase 0: Pulling up
    if not pullup_reached:
        if not start_reached:
            initial_shoulder_position = shoulder_y
        start_reached = True
        if not start_message_shown:
            print('Start position confirmed')
            start_message_shown = True
        current_phase = 0


    # Phase 1: Check top position WORK ON THIS
    if shoulder_angle <= 160:
        pullup_reached = True
        if not pullup_message_shown:
            print('Pull up phase confirmed')
            pullup_message_shown = True 
        current_phase = 1
        

    # Phase 2: Check descent position
    if pullup_reached and not descent_reached and wrist_y < shoulder_y and elbow_angle < threshold_bent_angle:
        top_reached = True
        if not top_message_shown:
            print('Top position reached')
            top_message_shown = True
        current_phase = 2
    
    # Phase 3: Check start/end position
    if top_reached and elbow_angle < 5:
        descent_reached = True
        if not descent_message_shown:
            print('Descent position reached')
            descent_message_shown = True
        current_phase = 3
    
    # Check if pull-up is successful
    if start_reached and pullup_reached and top_reached and descent_reached:
        if elbow_angle > 120:
            successful_count += 1
            print(f'Pull-up number {successful_count} completed!')
            start_reached = pullup_reached = top_reached = descent_reached = False
            start_message_shown = pullup_message_shown = top_message_shown = descent_message_shown = False
    
    return current_phase

def calculate_angle(a, b, c): # Calculate angle between points a, b, c
    a = np.array(a)  # Convert three points to numpy arrays [(x,y,z)]
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# 2. Play video to label frames and get (frame, label) sequences to feed into our model
def process_video(video_path, sequence_length):

    cap = cv.VideoCapture(video_path)

    # Store sequence data and labels for RNN
    landmark_sequences = []
    label_sequences = []
    frame_sequence = []
    label_sequence = []

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        # Capture video frame by frame
        ret, frame = cap.read()
        # Ret will be true if frame read correctly
        if not ret:
            print("End of video. Exiting...")
            break

        # Operations on the video frame
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Find pose landmarks
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = {  # Extract landmark positions
                'wrist': (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1], 
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]),
                'elbow': (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]),
                'shoulder': (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]),
                'hip': (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0])
            }

            # Flatten landmarks for RNN input
            features = np.array(list(landmarks.values())).flatten()
            # Append features and label for this frame
            current_frame_phase = is_successful_pullup(landmarks)  # Returns label for each frame
            frame_sequence.append(features)
            label_sequence.append(current_frame_phase)

            # Check if we've reached the desired sequence length
            if len(frame_sequence) == sequence_length:
                landmark_sequences.append(np.array(frame_sequence))
                label_sequences.append(np.array(label_sequence))

                # Reset sequences for next chunk
                frame_sequence = []
                label_sequence = []
        
        # Display the resulting frame
        cv.imshow('MediaPipe Pose', frame)
        if cv.waitKey(5) & 0xFF == 27:
            break

    # Release the capture when everything is done
    cap.release()
    cv.destroyAllWindows()

    # Return the landmark sequences and label sequences for a video
    return np.array(landmark_sequences), np.array(label_sequences)

process_video(video_path='data/pullup_demo.mov', sequence_length=10)