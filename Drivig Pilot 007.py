import cv2
import numpy as np
from collections import deque

# Păstrăm o coadă pentru istoricul detectărilor (maxim 5 cadre)
num_frames = 5
left_lines = deque(maxlen=num_frames)
right_lines = deque(maxlen=num_frames)

# Funcția pentru definirea regiunii de interes (ROI) în imagine
def region_of_interest(image):
    height, width = image.shape[:2]  # Obținem dimensiunile imaginii
    base_large_y = int(height * 0.90)  # Y-ul pentru baza mare a trapezului
    base_small_y = int(height * 0.55)  # Y-ul pentru baza mică a trapezului
    base_large_left_x = int(width * 0.25)  # X-ul pentru colțul stâng al bazei mari
    base_large_right_x = width  # X-ul pentru colțul drept al bazei mari
    base_small_width = int(width * 0.1)  # Lățimea bazei mici
    base_small_left_x = int(width * 0.5 - base_small_width / 2)  # X-ul pentru colțul stâng al bazei mici
    base_small_right_x = int(width * 0.5 + base_small_width / 2)  # X-ul pentru colțul drept al bazei mici

    # Definim punctele trapezului
    trapezoid_points = np.array([
        (base_large_left_x, base_large_y),
        (base_large_right_x, base_large_y),
        (base_small_right_x, base_small_y),
        (base_small_left_x, base_small_y),
    ], np.int32)

    # Creăm o mască și aplicăm trapezul pe ea
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [trapezoid_points], 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image, trapezoid_points

# Funcția care calculează media pantei și interceptării liniilor detectate
def average_slope_intercept(lines):
    left_fit = []  # Lista pentru liniile de pe partea stângă
    right_fit = []  # Lista pentru liniile de pe partea dreaptă
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # Evităm diviziunea prin 0
            continue
        slope = (y2 - y1) / (x2 - x1)  # Calculăm panta liniei
        intercept = y1 - slope * x1  # Calculăm interceptul
        if abs(slope) < 0.5:  # Filtrăm linii cu pantă prea mică
            continue
        if slope < 0:  # Linia este pe partea stângă
            left_fit.append((slope, intercept))
        else:  # Linia este pe partea dreaptă
            right_fit.append((slope, intercept))
    # Calculăm media pantei și interceptului pentru fiecare grup de linii
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    return left_fit_average, right_fit_average

# Funcția pentru a crea punctele unei linii pe baza pantei și interceptului
def make_line_points(height, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = height  # Partea de jos a imaginii
    y2 = int(height * 0.6)  # Partea de sus a imaginii
    x1 = int((y1 - intercept) / slope)  # Calculăm punctul X pentru y1
    x2 = int((y2 - intercept) / slope)  # Calculăm punctul X pentru y2
    return [[x1, y1, x2, y2]]

# Funcția care stabilește liniile detectate pe baza istoricului
def stabilize_lines(lines, left_line, right_line):
    if left_line is not None:
        lines[0].append(left_line)  # Adăugăm linia stângă la istoricul său
    if right_line is not None:
        lines[1].append(right_line)  # Adăugăm linia dreaptă la istoricul său
    
    # Calculăm media liniilor din istoricul pentru a obține linii stabile
    left_lines_mean = np.mean(lines[0], axis=0) if lines[0] else None
    right_lines_mean = np.mean(lines[1], axis=0) if lines[1] else None
    
    return left_lines_mean, right_lines_mean

# Funcția principală pentru detectarea benzilor pe drum
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversia la gri
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicăm un filtru Gaussian
    edges = cv2.Canny(blur, 50, 150)  # Detectăm marginile cu Canny
    roi, trapezoid_points = region_of_interest(edges)  # Aplicați ROI
    edges_with_roi = cv2.polylines(np.copy(edges), [trapezoid_points], isClosed=True, color=255, thickness=5)  # Desenăm ROI pe margini
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, maxLineGap=150, minLineLength=40)  # Detectarea liniilor cu Hough Transform
    line_image = np.zeros_like(frame)  # Creăm o imagine goală pentru linii

    # Dacă există linii detectate, calculăm panta și interceptul pentru fiecare linie
    if lines is not None:
        left_fit_average, right_fit_average = average_slope_intercept(lines)
        left_line = make_line_points(frame.shape[0], left_fit_average) if left_fit_average is not None else None
        right_line = make_line_points(frame.shape[0], right_fit_average) if right_fit_average is not None else None
        stable_left_line, stable_right_line = stabilize_lines((left_lines, right_lines), left_line, right_line)  # Stabilizăm liniile

        # Dacă există o linie stabilă stângă, o desenăm pe imagine
        if stable_left_line is not None:
            for x1, y1, x2, y2 in stable_left_line:
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        # Dacă există o linie stabilă dreaptă, o desenăm pe imagine
        if stable_right_line is not None:
            for x1, y1, x2, y2 in stable_right_line:
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

    # Combinăm imaginea cu liniile detectate
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image, gray, blur, edges_with_roi, line_image

# Funcția pentru a combina imagini într-o singură imagine pentru vizualizare
def stack_images(scale, video_frame, steps, labels):
    video_h, video_w, _ = video_frame.shape
    step_images = []  # Lista pentru pașii de procesare

    # Redimensionăm fiecare imagine de procesare pentru a avea dimensiuni uniforme
    for step in steps:
        if len(step.shape) == 2:  # Dacă imaginea este în grayscale, o convertim la BGR
            step = cv2.cvtColor(step, cv2.COLOR_GRAY2BGR)
        resized_step = cv2.resize(step, (video_w // 3, video_h // 4))  # Redimensionăm imaginea
        step_images.append(resized_step)

    # Concatenăm imaginile pe orizontală
    stacked_steps = cv2.hconcat(step_images)

    # Redimensionăm imaginea principală pentru a se potrivi cu pașii de procesare
    resized_frame = cv2.resize(video_frame, (video_w, int(video_h * 0.75)))

    # Asigurăm că dimensiunile imaginilor se potrivesc pentru concatenare
    if resized_frame.shape[1] != stacked_steps.shape[1]:
        stacked_steps = cv2.resize(stacked_steps, (resized_frame.shape[1], stacked_steps.shape[0]))

    # Concatenăm imaginea principală cu pașii de procesare
    output = cv2.vconcat([resized_frame, stacked_steps])

    # Adăugăm etichete pentru fiecare imagine procesată
    for i, label in enumerate(labels):
        x = i * (video_w // 3) + 10
        y = int(video_h * 0.75) + 30
        cv2.putText(output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return output

# Deschidem fișierul video
cap = cv2.VideoCapture('driving.mov')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectăm benzile și obținem pașii procesării
    lane_frame, gray, blur, edges, line_image = detect_lanes(frame)
    output = stack_images(1, lane_frame, [gray, blur, edges, line_image], [
        "Grayscale Conversion", "Gaussian Filtering", "Edge Detection", "Line Detection"
    ])

    # Afișăm imaginea finală
    cv2.imshow("Lane Detection", output)

    # Oprirea buclei la apăsarea tastei 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Eliberăm sursa video
cv2.destroyAllWindows()  # Închidem toate feroneriile OpenCV
