import cv2
import numpy as np
import sys
from ultralytics import YOLO
import concurrent.futures

yolo_path = "yolov8n.pt"

print("Loading YOLO model...")
model = YOLO(yolo_path)
print("Model loaded.")


def detect_lanes(gray_img):
    """
    Detects lanes using simple thresholding and centre-of-mass of the lane pixels.
    Returns (None, lane_center_x) – no Hough lines are computed.
    """
    height, width = gray_img.shape[:2]

    # 1. Threshold to get bright lane markings (adjust the value if needed)
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    # 2. Apply the same ROI mask as before
    mask = np.zeros_like(thresh)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(thresh, mask)

    # 3. Extract coordinates of all white pixels
    y_idx, x_idx = np.where(masked == 255)
    if len(x_idx) == 0:                     # no lane pixels found
        return None, width // 2

    frame_center = width // 2

    # 4. Separate into left and right halves
    left_mask = x_idx < frame_center
    right_mask = x_idx > frame_center

    # 5. Only consider the bottom part of the frame (e.g., bottom 30%)
    bottom_cutoff = int(height * 0.7)
    bottom_mask = y_idx > bottom_cutoff

    left_x_bottom = x_idx[left_mask & bottom_mask]
    right_x_bottom = x_idx[right_mask & bottom_mask]

    # Fall back to all points if the bottom region is empty for a side
    if len(left_x_bottom) == 0:
        left_x_bottom = x_idx[left_mask]
    if len(right_x_bottom) == 0:
        right_x_bottom = x_idx[right_mask]

    # 6. Compute average x for left and right
    left_avg = int(np.mean(left_x_bottom)) if len(left_x_bottom) > 0 else 0
    right_avg = int(np.mean(right_x_bottom)) if len(right_x_bottom) > 0 else width

    lane_center_x = (left_avg + right_avg) // 2
    return None, lane_center_x          # no line segments to draw


def detect_obstacles(rgb_img):
    results = model(rgb_img, verbose=False)[0]
    boxes = results.boxes

    bboxes = []
    stop_flag = False

    height, width, _ = rgb_img.shape

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in [0, 2, 3, 5, 7] and conf > 0.5:
            bboxes.append((x1, y1, x2, y2, cls))

            box_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height

            if (box_area / frame_area) > 0.15:
                if x2 > width * 0.25 and x1 < width * 0.75:
                    stop_flag = True

    return bboxes, stop_flag


def open_source(source):
    # If source is a digit string, treat as webcam index
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        sys.exit(1)

    return cap


def draw_frame(frame, lines, lane_center_x, bboxes, decision, color):
    height, width = frame.shape[:2]
    frame_center_x = width // 2

    # Only draw lane lines if provided (they are None with the new detection)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.circle(frame, (lane_center_x, height - 50), 10, (0, 255, 0), -1)
    cv2.circle(frame, (frame_center_x, height - 50), 10, (0, 0, 255), -1)

    for x1, y1, x2, y2, cls in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(frame, f"ACTION: {decision}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return frame


def main():
    # Usage:
    #   python pipeline_video.py              -> webcam 0
    #   python pipeline_video.py 1            -> webcam index 1
    #   python pipeline_video.py road.mp4     -> video file
    #   python pipeline_video.py rtsp://...   -> RTSP stream
    source = sys.argv[1] if len(sys.argv) > 1 else "0"

    cap = open_source(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    print(f"Streaming from '{source}' at {fps:.1f} fps. Press 'q' to quit.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or frame read failed.")
                break

            rgb_img = frame
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            future_lanes = executor.submit(detect_lanes, gray_img)
            future_obstacles = executor.submit(detect_obstacles, rgb_img)

            lines, lane_center_x = future_lanes.result()
            bboxes, stop_flag = future_obstacles.result()

            height, width = gray_img.shape[:2]
            frame_center_x = width // 2
            steering_margin = int(width * 0.05)

            decision = "FORWARD"
            color = (0, 255, 0)

            if stop_flag:
                decision = "STOP - OBSTACLE AHEAD"
                color = (0, 0, 255)
            elif lane_center_x < (frame_center_x - steering_margin):
                decision = "TURN LEFT"
                color = (255, 255, 0)
            elif lane_center_x > (frame_center_x + steering_margin):
                decision = "TURN RIGHT"
                color = (0, 255, 255)

            display = draw_frame(frame.copy(), lines, lane_center_x, bboxes, decision, color)

            cv2.imshow("Hybrid Self-Driving Pipeline", display)
            print(f"Decision: {decision}")

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()