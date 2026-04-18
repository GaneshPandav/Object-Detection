import cv2
from ultralytics import YOLO
import numpy as np
import time
import sys

def find_working_camera():
    """Try to find a working camera by testing multiple indices and backends."""
    backends = [
        ("Default", cv2.CAP_ANY),
        ("MSMF", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
    ]
    
    for cam_index in range(3):  # Try camera 0, 1, 2
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            
            # Give camera time to initialize
            time.sleep(1)
            
            # Read and discard a few frames
            for _ in range(10):
                cap.read()
            
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_val = np.mean(frame)
                std_val = np.std(frame)
                if mean_val > 5 and std_val > 10:
                    print(f"Found working camera: index={cam_index}, backend={backend_name}")
                    print(f"  Frame: {frame.shape}, mean={mean_val:.1f}, std={std_val:.1f}")
                    return cap, cam_index, backend_name
                else:
                    print(f"Camera {cam_index} ({backend_name}): frame appears blank or placeholder (mean={mean_val:.1f}, std={std_val:.1f})")
            
            cap.release()
    
    return None, -1, ""

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Find a working camera
print("Searching for a working camera...")
cap, cam_idx, backend_name = find_working_camera()

if cap is None:
    print("\n" + "="*60)
    print("ERROR: No working camera found!")
    print("="*60)
    print("\nPlease check the following:")
    print("  1. Is a webcam connected to your computer?")
    print("  2. Is the camera privacy shutter/cover open?")
    print("  3. Is another app using the camera? (Close Zoom, Teams, etc.)")
    print("  4. Windows Settings > Privacy > Camera > Allow desktop apps")
    print("  5. Check Device Manager for camera driver issues")
    print("\nTo use a video file instead, edit the script and change:")
    print("  cv2.VideoCapture(0) -> cv2.VideoCapture('path/to/video.mp4')")
    sys.exit(1)

print(f"\nUsing camera {cam_idx} with {backend_name} backend")
print("Press 'q' in the video window to quit.\n")

# Create the display window
cv2.namedWindow('YOLOv8 Object Counting', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model.predict(frame)

    # Count the number of detected objects
    num_objects = len(results[0].boxes)

    # Annotate the frame with the count
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f'Count: {num_objects}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Counting', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
