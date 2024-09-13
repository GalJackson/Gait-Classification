import pandas as pd
from ultralytics import YOLO
import numpy as np

def extract_keypoints(results, direction):
    """ 
    Prepare results from keypoint extraction for CSV export
    Extracts keypoints for either the leftmost or rightmost person based on the user input
    """
    kps_series = []
    right_hip_index = 11  # index for RHip
    left_hip_index = 12   # index for LHip

    for frame in results:
        if frame.keypoints.has_visible:
            # get keypoints and confidence scores
            kps_all = frame.keypoints.xy  
            conf_all = frame.keypoints.conf  
            
            # decide if we should track the leftmost or rightmost person
            if direction == "right":
                # find the person with the most rightward 'RHip'
                extreme_x = -float('inf')
                selected_index = -1
                for i in range(len(kps_all)):
                    hip_x = kps_all[i][right_hip_index][0]
                    if hip_x > extreme_x:
                        extreme_x = hip_x
                        selected_index = i
            else:
                # find the person with the most leftward 'LHip'
                extreme_x = float('inf')
                selected_index = -1
                for i in range(len(kps_all)):
                    hip_x = kps_all[i][left_hip_index][0]
                    if hip_x < extreme_x:
                        extreme_x = hip_x
                        selected_index = i
            
            # get keypoints for the selected person
            kps = kps_all[selected_index]
            conf = conf_all[selected_index]
            
            row = []
            for i in range(len(kps)):
                row.append(list(kps[i]) + [conf[i]])

            # flatten row
            row = [float(i) for sublist in row for i in sublist]
        else:
            # NaN if no keypoints are found
            row = [float('nan')] * 51

        kps_series.append(row)
    
    return kps_series

def create_dataframe(keypoints_series):
    # headers for df columns
    header = []
    keypoint_names = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", 
                      "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
    for name in keypoint_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    
    # create keypoint df
    return pd.DataFrame(keypoints_series, columns=header)

def save_to_csv(df, output_path):
    # export df to CSV
    df.to_csv(output_path, index=False)
    print(f"Pose extraction complete. The CSV is at {output_path}")

def main():
    video_name = "OAW01-bottom-front2.mp4"
    video_path = f"../Videos/Cropped/{video_name[:5]}/" + video_name
    output_path = "yolo_exports/" + video_name[:-4] + ".csv"
    model_path = "yolov8n-pose.pt"
    
    # ask the user for which person to track - "left" or "right"
    while True:
        direction = input("Do you want to track the person who is most to the 'left' or 'right'? (Enter 'left' or 'right'): ").strip().lower()
        if direction in ['left', 'right']:
            break
        else:
            print("Invalid input. Please enter 'left' or 'right'.")

    # load yolo model
    model = YOLO(model_path) 
    
    # get results
    results = model(video_path, stream=True)
    
    # extract keypoints
    keypoints_series = extract_keypoints(results, direction)
    
    # create df from keypoints
    df = create_dataframe(keypoints_series)
    
    # save df to csv
    save_to_csv(df, output_path)

if __name__ == "__main__":
    main()
