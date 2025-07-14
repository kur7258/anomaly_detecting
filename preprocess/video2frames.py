import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, final_size):
    # Open the video
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    # Iter to extract the frames
    while success:
        # Resize while preserving aspect ratio
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1:  # wider than tall
            new_w = final_size
            new_h = int(final_size / aspect_ratio)
        else:  # taller than wide
            new_h = final_size
            new_w = int(final_size * aspect_ratio)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Create a square canvas with padding
        square_frame = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        
        # Calculate padding to center the image
        y_offset = (final_size - new_h) // 2
        x_offset = (final_size - new_w) // 2
        
        # Place the resized frame in the center
        square_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        frame_path = f"{output_folder}/train4_{count}.jpg"
        cv2.imwrite(frame_path, square_frame)  # Save the frame
        success, frame = video.read()  # read the next frame
        count += 1

    print("Total frames: {}".format(count))

    video.release()


video_path = '/home/metanet/workspace/anomaly_detecting/temp_for_train/abnormal2.MOV'
output_folder = '/home/metanet/workspace/anomaly_detecting/tabacco_images/abnormal'
final_size = 1024
os.makedirs(output_folder, exist_ok=True)
extract_frames(video_path, output_folder, final_size)