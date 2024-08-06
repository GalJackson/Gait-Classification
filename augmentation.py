import os
import cv2
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



if __name__ == "__main__":
    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
    
    directory_path = filedialog.askdirectory()
    directory = os.fsencode(directory_path)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith("mirror.mp4"): # skip any videos already mirrored
            flip_video(directory_path, filename)
        else:
            continue