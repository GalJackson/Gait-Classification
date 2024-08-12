import pandas as pd
import cv2
import numpy as np


def draw_keypoints(frame, keypoints):
    """
    This function takes a list of keypoints (x coord, y coord, and cofindence score) and draws them on the given frame
    """
    for i in range(0, len(keypoints), 3):
        try:
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            
            # draw green keypoints
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  
        except ValueError:
            continue

    return frame

# set the size of the background
frame_width = 1080
frame_height = 1920

keypoints_df = pd.read_csv('test.csv', index_col=0)

# iterate over each frame
for frame_i in range(len(keypoints_df)):
    # create a white background image
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # extract keypoints for the current frame
    keypoints = keypoints_df.iloc[frame_i].values

    # draw keypoints on the white background
    frame_with_keypoints = draw_keypoints(frame, keypoints)
    cv2.imshow('Keypoints', frame_with_keypoints)
    
    # display at 30 frames per second, exit if 'q' is pressed
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("Keypoints display complete")
