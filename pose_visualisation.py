"""
This script will take a CSV file of pose keypoints and visualise the keypoints
"""

import pandas as pd
import cv2

keypoints_df = pd.read_csv('../Pose-Extractions/OAW06-bottom-back1.csv', index_col=0)
video_path = '../Videos/Cropped/OAW06/OAW06-bottom-back1.mp4' 

# draw keypoints on a video frame
def draw_keypoints(frame, keypoints):
    for i in range(0, len(keypoints), 3):
        try:
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            conf = keypoints[i+2]

            if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green keypoints
        except ValueError:
            continue

    return frame


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# iterate over each frame in the video
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # extract keypoints for the current frame from the df
    keypoints_row = keypoints_df.iloc[frame_idx].values
    keypoints = keypoints_row

    # draw keypoints on the frame
    frame_with_keypoints = draw_keypoints(frame, keypoints)
    
    # display the frame with keypoints
    cv2.imshow('Video with Keypoints', frame_with_keypoints)
    
    # exit the window if 'q' is pressed
    if cv2.waitKey(int(1000 / 35)) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()

print("Video ended")