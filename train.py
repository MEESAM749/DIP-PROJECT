import cv2
import numpy as np
import sys
from ultralytics import YOLO
import concurrent.futures

# Load the YOLO model for dynamic obstacle detection
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded.")

def preprocess_image(frame):
    """
    Applies contrast enhancement and color space conversion to make 
    lane lines pop out regardless of lighting conditions.
    """
    # 1. Contrast Enhancement (CLAHE) - handles uneven lighting
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame

def detect_lanes_robust(frame):
    """
    Robust lane detection using color segmentation, morphological operations,
    Canny edge detection, and polynomial fitting for curved lanes.
    """
    height, width = frame.shape[:2]
    
    # 1. Convert to HLS color space for better white/yellow extraction
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # Define thresholds for White and Yellow lanes
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 2. Morphological Closing to fill gaps in lane lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Spatial Filtering & Edge Detection
    blurred = cv2.GaussianBlur(closed_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Region of Interest (ROI) Masking
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.40), int(height * 0.6)),
        (int(width * 0.60), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 5. Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, 
                            minLineLength=20, maxLineGap=100)
    
    # 6. Polynomial Fitting for Curved Roads
    left_x, left_y, right_x, right_y = [], [], [], []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-6) # Avoid division by zero
                if abs(slope) < 0.3: continue # Ignore near-horizontal lines
                
                if slope < 0: # Left lane
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                else:         # Right lane
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

    # Fit a 2nd degree polynomial (curve) if we have enough points, otherwise linear
    left_fit = np.polyfit(left_y, left_x, 2) if len(left_y) > 5 else None
    right_fit = np.polyfit(right_y, right_x, 2) if len(right_y) > 5 else None
    
    # Calculate target center at the bottom of the frame
    bottom_y = height
    target_left_x = int(np.polyval(left_fit, bottom_y)) if left_fit is not None else 0
    target_right_x = int(np.polyval(right_fit, bottom_y)) if right_fit is not None else width
    
    lane_center_x = (target_left_x + target_right_x) // 2
    
    return lines, lane_center_x, masked_edges

def detect_obstacles(rgb_img):
    """
    Detects dynamic obstacles using YOLOv8 and calculates relative size 
    to trigger braking mechanism.
    """
    results = model(rgb_img, verbose=False)[0]
    boxes = results.boxes
    bboxes = []
    stop_flag = False

    height, width, _ = rgb_img.shape

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Filter for vehicles and pedestrians (COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
        if cls in [0, 2, 3, 5, 7] and conf > 0.50:
            bboxes.append((x1, y1, x2, y2, cls))

            # Proximity heuristic based on bounding box area
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height

            # If the object takes up more than 15% of the frame and is in our direct path
            if (box_area / frame_area) > 0.15:
                if x2 > width * 0.25 and x1 < width * 0.75:
                    stop_flag = True

    return bboxes, stop_flag

def draw_hud(frame, lines, lane_center_x, bboxes, decision, color, masked_edges):
    """Draws heads-up display (HUD) and debug information."""
    height, width = frame.shape[:2]
    frame_center_x = width // 2

    # Draw detected raw hough lines for debugging
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw steering center markers
    cv2.circle(frame, (lane_center_x, height - 50), 10, (0, 255, 0), -1) # Desired path
    cv2.circle(frame, (frame_center_x, height - 50), 10, (0, 0, 255), -1) # Car center
    cv2.line(frame, (frame_center_x, height - 50), (lane_center_x, height - 50), (255, 255, 255), 2)

    # Draw Obstacles
    for x1, y1, x2, y2, cls in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "OBSTACLE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Action Output
    cv2.rectangle(frame, (10, 10), (500, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"ACTION: {decision}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return frame

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    print(f"Streaming from '{source}' at {fps:.1f} fps. Press 'q' to quit.")

    # PC Optimization: Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Pre-process for robust lighting adaptation
            enhanced_frame = preprocess_image(frame)

            # Submit parallel tasks for Hybrid Vision System
            future_lanes = executor.submit(detect_lanes_robust, enhanced_frame)
            future_obstacles = executor.submit(detect_obstacles, frame)

            # Gather results
            lines, lane_center_x, masked_edges = future_lanes.result()
            bboxes, stop_flag = future_obstacles.result()

            # Directional Control Logic
            height, width = frame.shape[:2]
            frame_center_x = width // 2
            steering_margin = int(width * 0.05) 

            decision = "FORWARD"
            color = (0, 255, 0)

            if stop_flag:
                decision = "STOP - OBSTACLE"
                color = (0, 0, 255)
            elif lane_center_x < (frame_center_x - steering_margin):
                decision = "TURN LEFT"
                color = (255, 255, 0)
            elif lane_center_x > (frame_center_x + steering_margin):
                decision = "TURN RIGHT"
                color = (0, 255, 255)

            # Display Output and Debugging
            display = draw_hud(frame.copy(), lines, lane_center_x, bboxes, decision, color, masked_edges)
            
            # Show a picture-in-picture debug view of the edge detection
            debug_view = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
            debug_view = cv2.resize(debug_view, (int(width * 0.3), int(height * 0.3)))
            display[10:10+debug_view.shape[0], width-debug_view.shape[1]-10:width-10] = debug_view
            cv2.putText(display, "Edge Mask Debug", (width-debug_view.shape[1]-10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Hybrid Vision System", display)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()