import cv2
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm 

# set up model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda"

# initialize the predictor
predictor = DefaultPredictor(cfg)

# open video
video_path =  "../Videos/Cropped/OAW02/OAW02-bottom-back2.mp4"
cap = cv2.VideoCapture(video_path)

# get video properties
fps = 30
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

# CSV file info
csv_output_path = 'detectron_exports/OAW02-bottom-back2.csv'
keypoint_names = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", 
                  "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]

# cselect where the person to detect is standing - left/right
while True:
    direction = input("Do you want to track the person who is most to the 'left' or 'right'? (Enter 'left' or 'right'): ").strip().lower()
    if direction in ['left', 'right']:
        break
    else:
        print("Invalid input. Please enter 'left' or 'right'.")

# define the keypoint index based on direction
if direction == "right":
    keypoint_index = 12  # 'RHip' for rightmost
else:
    keypoint_index = 11  # 'LHip' for leftmost

# format CSV headers
csv_columns = ['time'] + [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y', 'confidence']]
csv_data = []

# initialize the progress bar with tqdm
frame_number = 0
with tqdm(total=frame_count, desc='Processing Frames', unit='frame') as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # calculate time for the current frame
        time_in_seconds = frame_number / fps

        # make hpe predictions on the frame
        outputs = predictor(frame)

        # extract keypoints
        if len(outputs["instances"].pred_keypoints) > 0:
            keypoints_all = outputs["instances"].pred_keypoints.cpu().numpy()  # All people detected

            # find the person with the most leftward or rightward keypoint
            if direction == "right":
                # find the person with the most rightward 'RHip' keypoint
                extreme_x = -float('inf')  # smallest possible value
                selected_index = -1
                for i, keypoints in enumerate(keypoints_all):
                    # get the x-coordinate of the 'RHip'
                    hip_x = keypoints[keypoint_index][0]
                    if hip_x > extreme_x:
                        extreme_x = hip_x
                        selected_index = i
            else:
                # find the person with the most leftward 'LHip' keypoint
                extreme_x = float('inf')  # largest possible value
                selected_index = -1
                for i, keypoints in enumerate(keypoints_all):
                    # get the x-coordinate of the 'LHip'
                    hip_x = keypoints[keypoint_index][0]
                    if hip_x < extreme_x:
                        extreme_x = hip_x
                        selected_index = i

            # use only the keypoints of the selected person
            keypoints = keypoints_all[selected_index]
        else:
            # if no keypoints detected use None
            keypoints = [[None, None, None] for _ in range(17)]

        # Prepare df row
        row = [time_in_seconds]
        for kp in keypoints:
            x, y, confidence = kp
            row.extend([x, y, confidence])

        csv_data.append(row)
        frame_number += 1

        # progress bar
        pbar.update(1)

# release resources 
cap.release()

# save in CSV
df = pd.DataFrame(csv_data, columns=csv_columns)
df.to_csv(csv_output_path, index=False)

print(f"Pose estimation completed and saved to {csv_output_path}")