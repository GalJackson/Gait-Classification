import pandas as pd
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt") 

# Predict with the model
results = model("../Videos/OAW01-bottom.mp4")

# Prepare results for export to CSV
kps_series = []

for frame in results:
    if frame.keypoints.has_visible:
        # x and y position of each keypoint (in pixels from top left)
        kps = frame.keypoints.xy[0]

        # keypoint confidence score
        conf = frame.keypoints.conf[0]

        row = []

        for i in range(0, len(kps)):
            row.append(list(kps[i]) + [conf[i]])

        # Flatten the row
        row = [float(i) for sublist in row for i in sublist]
    
    else:
        row = [float('nan')] * 51

    
    kps_series.append(row)

header = []
keypoint_names = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
for name in keypoint_names:
    header.extend([f"{name}X", f"{name}Y", f"{name}Conf"])

# Pandas DataFrame for the keypoints
df = pd.DataFrame(kps_series, columns=header)


# export DF to CSV


print("success")

