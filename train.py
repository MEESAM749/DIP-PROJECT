import cv2
import numpy as np
import sys
from ultralytics import YOLO
import concurrent.futures
import torch

# --- Tunable constants ---
ROI_TOP        = 0.60   # Horizon as fraction of height
SLOPE_MIN      = 0.50   # Reject near-horizontal lines
SLOPE_MAX      = 2.50   # Reject near-vertical noise
HOUGH_THRESH   = 30
MIN_LINE_LEN   = 30
MAX_LINE_GAP   = 150
EMA_ALPHA      = 0.25   # Temporal smoothing for lane center
STEER_MARGIN   = 0.05   # Fraction of width for dead-zone
STOP_DEFAULTS  = {0: 0.08, 2: 0.15, 3: 0.15, 5: 0.18, 7: 0.20}  # class: area ratio

print("Loading YOLO model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO("yolov8n.pt")
model.to(device)
print("Model loaded.")

_smoothed_center = None  # Module-level EMA state


def preprocess_image(frame, roi_top=ROI_TOP):
    """CLAHE enhancement restricted to the road ROI."""
    height, width = frame.shape[:2]
    roi_y = int(height * roi_top)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only on road region
    l[roi_y:] = clahe.apply(l[roi_y:])

    limg = cv2.merge((l, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def _fit_lane_line(xs, ys):
    """Linear polyfit with outlier rejection via IQR on x-values."""
    if len(ys) < 6:
        return None
    xs, ys = np.array(xs), np.array(ys)
    q1, q3 = np.percentile(xs, 25), np.percentile(xs, 75)
    iqr = q3 - q1
    mask = (xs >= q1 - 1.5 * iqr) & (xs <= q3 + 1.5 * iqr)
    if mask.sum() < 6:
        return None
    return np.polyfit(ys[mask], xs[mask], 1)  # linear, not quadratic


def _eval_fit_center(fit, y_bottom, y_top):
    """Average x across the lower portion of the lane fit."""
    ys = np.linspace(y_top, y_bottom, 20)
    return int(np.mean(np.polyval(fit, ys)))


def detect_lanes(frame):
    global _smoothed_center
    height, width = frame.shape[:2]

    # Color masking in HLS
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask  = cv2.inRange(hls, np.array([0, 200, 0],   np.uint8),
                                   np.array([255, 255, 255], np.uint8))
    yellow_mask = cv2.inRange(hls, np.array([10, 0, 100],  np.uint8),
                                   np.array([40, 255, 255], np.uint8))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    blurred = combined_mask#cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # ROI mask
    roi_top = int(height * ROI_TOP)
    poly = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.40), roi_top),
        (int(width * 0.60), roi_top),
        (int(width * 0.90), height),
    ]], np.int32)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, poly, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, HOUGH_THRESH,
                            minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)

    left_x, left_y, right_x, right_y = [], [], [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < SLOPE_MIN or abs(slope) > SLOPE_MAX:
                continue
            if slope < 0:
                left_x.extend([x1, x2]);  left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2]); right_y.extend([y1, y2])

    left_fit  = _fit_lane_line(left_x,  left_y)
    right_fit = _fit_lane_line(right_x, right_y)

    lane_lost = left_fit is None and right_fit is None

    if not lane_lost:
        y_bottom = height
        y_eval   = int(height * 0.70)  # average center across lower 30%

        if left_fit is not None and right_fit is not None:
            raw_center = (_eval_fit_center(left_fit,  y_bottom, y_eval) +
                          _eval_fit_center(right_fit, y_bottom, y_eval)) // 2
        elif left_fit is not None:
            raw_center = _eval_fit_center(left_fit, y_bottom, y_eval) + width // 4
        else:
            raw_center = _eval_fit_center(right_fit, y_bottom, y_eval) - width // 4

        # Temporal EMA smoothing
        if _smoothed_center is None:
            _smoothed_center = raw_center
        else:
            _smoothed_center = int(EMA_ALPHA * raw_center +
                                   (1 - EMA_ALPHA) * _smoothed_center)

    return lines, _smoothed_center, masked_edges, lane_lost


def detect_obstacles(rgb_img):
    results = model(rgb_img, verbose=False)[0]
    height, width, _ = rgb_img.shape
    bboxes, stop_flag = [], False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if cls not in STOP_DEFAULTS or conf <= 0.50:
            continue
        bboxes.append((x1, y1, x2, y2, cls))

        box_area   = (x2 - x1) * (y2 - y1)
        frame_area = width * height
        area_ratio = box_area / frame_area
        proximity  = y2 / height  # 1.0 = bottom of frame = close

        threshold = STOP_DEFAULTS[cls]
        if area_ratio > threshold and proximity > 0.5:
            if x2 > width * 0.25 and x1 < width * 0.75:
                stop_flag = True

    return bboxes, stop_flag


def draw_hud(frame, lines, lane_center_x, bboxes, decision,
             color, masked_edges, lane_lost, steering_offset):
    height, width = frame.shape[:2]
    frame_center_x = width // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not lane_lost and lane_center_x is not None:
        cv2.circle(frame, (lane_center_x, height - 50), 10, (0, 255, 0), -1)
        cv2.line(frame, (frame_center_x, height - 50),
                         (lane_center_x, height - 50), (255, 255, 255), 2)

    cv2.circle(frame, (frame_center_x, height - 50), 10, (0, 0, 255), -1)

    for x1, y1, x2, y2, cls in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "OBSTACLE", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.rectangle(frame, (10, 10), (560, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"ACTION: {decision}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"Steer offset: {steering_offset:+.3f}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Debug edge mask inset
    debug = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    dw, dh = int(width * 0.3), int(height * 0.3)
    debug = cv2.resize(debug, (dw, dh))
    frame[10:10 + dh, width - dw - 10:width - 10] = debug
    cv2.putText(frame, "Edge Mask", (width - dw - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def main():
    global _smoothed_center
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    print(f"Streaming from '{source}' at {fps:.1f} fps. Press 'q' to quit.")

    last_decision = "FORWARD"
    last_color    = (0, 255, 0)
    frame_count   = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            enhanced = preprocess_image(frame)
            f_lanes  = executor.submit(detect_lanes, enhanced)
            f_obs    = executor.submit(detect_obstacles, frame)

            lines, lane_center_x, masked_edges, lane_lost = f_lanes.result()
            if frame_count % 3 == 0:
                f_obs = executor.submit(detect_obstacles, frame)
                bboxes, stop_flag = f_obs.result()
            frame_count += 1

            height, width  = frame.shape[:2]
            frame_center_x = width // 2
            margin         = int(width * STEER_MARGIN)

            # Proportional steering offset in [-1, 1]
            if lane_lost or lane_center_x is None:
                steering_offset = 0.0
                decision = "LANE LOST — HOLD"
                color    = (0, 165, 255)
            else:
                steering_offset = (lane_center_x - frame_center_x) / (width * 0.5)

                if stop_flag:
                    decision = "STOP — OBSTACLE"
                    color    = (0, 0, 255)
                elif lane_center_x < frame_center_x - margin:
                    decision = "TURN LEFT"
                    color    = (255, 255, 0)
                elif lane_center_x > frame_center_x + margin:
                    decision = "TURN RIGHT"
                    color    = (0, 255, 255)
                else:
                    decision = "FORWARD"
                    color    = (0, 255, 0)

                last_decision = decision
                last_color    = color

            display = draw_hud(frame.copy(), lines, lane_center_x, bboxes,
                               decision, color, masked_edges,
                               lane_lost, steering_offset)
            cv2.imshow("Hybrid Vision System", display)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
