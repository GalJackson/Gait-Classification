import pandas as pd
import cv2
import numpy as np
import tkinter
from tkinter import filedialog


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


def read_data():
    """
    Prompts user to select file, then loads into a dataframe
    """

    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
    file_path = filedialog.askopenfile()

    df = pd.read_csv(file_path, index_col=0)

    return df


def simulate_data(df, frame_height, frame_width):
    """
    Takes a dataframe of keypoints and simulates their movement in a new window
    """

    # iterate over each frame
    for frame_i in range(len(df)):
        # if frame_i >= 85:
        #     print("pause")

        # create a white background image
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # extract keypoints for the current frame
        keypoints = df.iloc[frame_i].values

        # draw keypoints on the white background
        frame_with_keypoints = draw_keypoints(frame, keypoints)
        cv2.imshow('Keypoints', frame_with_keypoints)
        
        # display at 30 frames per second, exit if 'q' is pressed
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Keypoints display complete")


def main():
    # set the size of the background
    frame_width = 1080
    frame_height = 1920

    # load video to df
    keypoints_df = read_data()

    # simulate the keypoint movement
    simulate_data(keypoints_df, frame_height, frame_width)


if __name__ == "__main__":
    main()