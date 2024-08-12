import os
import cv2
import pandas as pd
import numpy as np
import tkinter
from tkinter import filedialog

def flip_video(directory_path, file_name): 
    # video file paths
    input_video_path = directory_path + '/' + file_name 
    output_video_path = input_video_path[:-4] + "-mirror.mp4" 

    video = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        raise IOError("Error: Could not open the input video.")

    # video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = video.get(cv2.CAP_PROP_FPS) 
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # create object for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # process video
    for i in range(frame_count):
        ret, frame = video.read()
        
        if not ret:
            raise IOError(f"Error: Frame {i} could not be read.")

        # flip video in horizontal direction
        mirrored_frame = cv2.flip(frame, 1)
        
        out.write(mirrored_frame)

    # release objects
    video.release()
    out.release()


def rotate_single_keypoint(x, y, rotation_matrix, center_x, center_y):
    # translate the point to the center the image, so it rotates about the center instead of the origin (0,0)
    translated_point = np.array([x - center_x, y - center_y])
    
    # apply the rotation matrix
    rotated_point = rotation_matrix @ translated_point
    
    # translate the point back from center
    new_x, new_y = rotated_point + np.array([center_x, center_y])
    
    return new_x, new_y


def rotate_keypoints(df, angle, img_width, img_height):
    # convert angle of rotation to radians
    theta = np.radians(angle)
    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # find center point of the video frame
    center_x = img_width / 2
    center_y = img_height / 2
    
    df_rotated = df.copy()
    
    for column in df.columns: # for each keypoint extracted from the pose 
        if '_x' in column:
            kp_name = column[:-2] 
            x_col = f'{kp_name}_x'
            y_col = f'{kp_name}_y'

            
            # apply the rotation matrix
            df_rotated[[x_col, y_col]] = df.apply(
                lambda row: rotate_single_keypoint(row[x_col], row[y_col], rotation_matrix, center_x, center_y), 
                axis=1, 
                result_type='expand'
            )
    
    return df_rotated


def resize_keypoints(df, n=1, m=1, img_width=1080, img_height=1920):
    """
    Resize keypoints via vertical and horizontal stretch or compression

    n: x-axis stretch factor
    m: y-axis stretch factor
    """

    # find center point of the video frame
    center_x = img_width / 2
    center_y = img_height / 2
    
    df_resized = df.copy()
    
    for column in df.columns: # for each keypoint extracted from the pose 
        if '_x' in column:
            keypoint = column[:-2]  
            x_col = f'{keypoint}_x'
            y_col = f'{keypoint}_y'
            
            # apply resizing: first translated relative to center, then scaled, and then translated back
            df_resized[x_col] = (df[x_col] - center_x) * n + center_x
            df_resized[y_col] = (df[y_col] - center_y) * m + center_y
    
    return df_resized


def mirror_keypoints(df, img_width):
    """
    flip keypoints horizontally (mirror across the y axis)
    """
    
    df_flipped = df.copy()
    
    for column in df.columns:
        if '_x' in column:
            # flip the x coordinate by subtracting it from the image width
            df_flipped[column] = img_width - df[column]
    
    return df_flipped


def augment_df(df):
    df_augmented = df.copy()

    # random rotation between -5 and 5 degrees
    angle = np.random.uniform(-10, 10)
    df_augmented = rotate_keypoints(df_augmented, angle, img_width=1920, img_height=1080)

    # randomly choose if to flip video horizontally
    if np.random.rand() > 0.5:
        df_augmented = mirror_keypoints(df_augmented, img_width=1920)

    # random rescaling on x and y axis between 0.8 and 1.2
    scale_x = np.random.uniform(0.8, 1.2)
    scale_y = np.random.uniform(0.8, 1.2)
    df_augmented = resize_keypoints(df_augmented, scale_x, scale_y, img_width=1920, img_height=1080)

    return df_augmented


def augment_gait_data(parent_dir, output_dir, augmentation_count):
    """
    This function performs various randomised augmentation operations to files in a given directory. Each augmentation will
    result in a new CSV file which will be saved under output_dir.

    parent_dir: path to the directory of existing CSV files
    output_dir: where the new augmented CSV files will be saved
    augmentation_count: number of augmentations per CSV file
    """

    # all CSV files in the parent directory
    csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv')]

    for csv in csv_files:
        df = pd.read_csv(os.path.join(parent_dir, csv))

        for i in range(augmentation_count):
            df_augmented = augment_df(df)
            augmented_filename = f"{csv.split('.')[0]}-augmented-{i}.csv"
            df_augmented.to_csv(os.path.join(output_dir, augmented_filename))


if __name__ == "__main__":
    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
    
    # directory_path = filedialog.askdirectory()
    # directory = os.fsencode(directory_path)
    
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if not filename.endswith("mirror.mp4"): # skip any videos already mirrored
    #         flip_video(directory_path, filename)
    #     else:
    #         continue

    filename = filedialog.askopenfilename()

    df = pd.read_csv(filename, index_col=0)

    df = rotate_keypoints(df, 5, 1080, 1920)

    df.to_csv('test.csv')