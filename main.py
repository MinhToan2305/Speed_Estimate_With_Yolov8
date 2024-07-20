import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO
import math
from collections import defaultdict, deque

# Thêm đường dẫn đến OC_SORT vào sys.path
sys.path.append('C:/Users/Administrator/Downloads/Download Project/yolo/OC_SORT/trackers')

# Import OC-SORT
from ocsort_tracker.ocsort import OCSort

# Đường dẫn đến video và mô hình YOLO
video_path = 'C:/Users/Administrator/Downloads/Download Project/yolo/Video/vehicles.mp4'
model_path = 'C:/Users/Administrator/Downloads/Download Project/yolo/Models/yolov8n.pt'
output_path = 'C:/Users/Administrator/My Documents/processed_vehicles.mp4'

# Khởi tạo mô hình YOLO
model = YOLO(model_path)

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

# Đọc thuộc tính video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Khởi tạo VideoWriter để lưu video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Khởi tạo OC-SORT tracker với giá trị det_thresh phù hợp
tracker = OCSort(det_thresh=0.5, max_age=5)

# Danh sách các loại phương tiện cần thiết
necessary_classes = [2, 3, 7]

# Tọa độ của đa giác chứa vùng quan tâm (ROI)
polygon = np.array([[(1260, 800), (2280, 800), (5039, 2159), (-550, 2159)]], dtype=np.int32)

# Tạo mặt nạ dựa trên đa giác (ROI)
frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
mask = np.zeros(frame_shape, dtype=np.uint8)
cv2.fillPoly(mask, polygon, (255, 255, 255))

# Cài đặt thông số cho tính toán tốc độ
pixels_per_meter = 1
car_start_positions = {}
car_speeds = {}

# Điểm góc trong ảnh gốc
src_points = np.float32([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
# Điểm góc trong ảnh được biến đổi
dst_points = np.float32([[0, 0], [24, 0], [24, 249], [0, 249]])

# Tính toán ma trận biến đổi phối cảnh
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Đọc và xử lý từng frame của video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Áp dụng mặt nạ để chỉ giữ lại vùng quan tâm
    roi = cv2.bitwise_and(frame, mask)

    # Thực hiện phát hiện đối tượng trên vùng đã xác định
    results = model(roi)

    # Tạo danh sách các detections từ YOLOv8
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls in necessary_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                detections.append([x1, y1, x2, y2, confidence])

    # Cung cấp thông tin hình ảnh cho OC-SORT
    img_info = (frame.shape[1], frame.shape[0])
    img_size = (frame.shape[1], frame.shape[0])

    # Áp dụng OC-SORT để theo dõi các đối tượng
    tracked_objects = tracker.update(np.array(detections), img_info, img_size)

    # Vẽ bounding boxes lên frame và tính toán tốc độ
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Chuyển đổi điểm trung tâm từ ảnh gốc sang ảnh được biến đổi
        original_point = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(original_point, perspective_matrix)
        transformed_x, transformed_y = transformed_point[0][0]

        if track_id not in car_start_positions:
            car_start_positions[track_id] = (transformed_x, transformed_y, current_time)
        else:
            x_start, y_start, start_time = car_start_positions[track_id]
            distance_in_pixels = math.sqrt((transformed_x - x_start) ** 2 + (transformed_y - y_start) ** 2)
            distance_in_meters = distance_in_pixels / pixels_per_meter
            elapsed_time = current_time - start_time

            if elapsed_time > 0:
                speed = (distance_in_meters / elapsed_time) * 3.6
                car_speeds[track_id] = speed

                # Vẽ bounding box và hiển thị tốc độ
                label = f'#{track_id} - {int(speed)} km/h'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Vẽ đa giác phân vùng lên frame
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=4)

    # Điều chỉnh kích thước khung hình
    scale_percent = 30
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Thay đổi kích thước khung hình
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Ghi frame đã chú thích vào tệp video
    out.write(frame)

    # Hiển thị frame đã chú thích
    cv2.imshow('YOLOv8', resized_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
