import cv2
import os
import csv
import datetime
import numpy as np

DATASET = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 70       # kam = sakht
REQUIRED_FRAMES = 8  # itne frame lagatar sahi match aaye tab attendance lage

# ---------- Face detector ----------
cascade_name = "haarcascade_frontalface_default.xml"
cascade_path = None
try:
    cascade_path = cv2.data.haarcascades + cascade_name
except Exception:
    pass

if cascade_path is None or not os.path.exists(cascade_path):
    cascade_path = cascade_name
    if not os.path.exists(cascade_path):
        import urllib.request
        url = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/data/haarcascades/" + cascade_name)
        print("Detector file download ho rahi hai...")
        urllib.request.urlretrieve(url, cascade_path)

detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    print("Detector load nahi hua.")
    raise SystemExit

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels, names = [], [], {}

if not os.path.isdir(DATASET):
    print("known_faces folder nahi mila.")
    raise SystemExit


def add_image(path, label_id):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image read nahi hui:", path)
        return 0
    h, w = img.shape
    if max(h, w) > 800:
        s = 800 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    found = detector.detectMultiScale(img, 1.1, 6, minSize=(80, 80))
    if len(found) == 0:
        print("Chehra nahi mila:", path)
        return 0
    x, y, fw, fh = max(found, key=lambda r: r[2] * r[3])
    face = cv2.resize(img[y:y+fh, x:x+fw], (200, 200))
    face = cv2.equalizeHist(face)
    faces.append(face)
    labels.append(label_id)
    return 1


label_id = 0
for item in sorted(os.listdir(DATASET)):
    full = os.path.join(DATASET, item)
    if os.path.isfile(full):
        name = os.path.splitext(item)[0]
        n = add_image(full, label_id)
        names[label_id] = name
        print(f"{name}: {n} chehre mile")
        label_id += 1
    elif os.path.isdir(full):
        total = 0
        for f in sorted(os.listdir(full)):
            total += add_image(os.path.join(full, f), label_id)
        names[label_id] = item
        print(f"{item}: {total} chehre mile")
        label_id += 1

if len(faces) == 0:
    print("Koi chehra nahi mila.")
    raise SystemExit

recognizer.train(faces, np.array(labels))
print("Training mukammal. Attendance system shuru... (band karne ke liye q)")


# ---------- Attendance ----------
def today_str():
    return datetime.date.today().isoformat()


def load_marked_today():
    marked = set()
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1] == today_str():
                    marked.add(row[0])
    return marked


def mark_attendance(name):
    new_file = not os.path.exists(ATTENDANCE_FILE)
    now = datetime.datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["Name", "Date", "Time"])
        w.writerow([name, now.date().isoformat(), now.strftime("%H:%M:%S")])
    print(f"Attendance lag gayi: {name} at {now.strftime('%H:%M:%S')}")


marked_today = load_marked_today()
streak = {}  # har banday ke lagatar sahi match

# ---------- Camera ----------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera nahi khula.")
    raise SystemExit

while True:
    ok, frame = cam.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    seen_now = set()

    for (x, y, w, h) in detector.detectMultiScale(gray, 1.2, 6, minSize=(80, 80)):
        roi = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
        label, confidence = recognizer.predict(roi)

        if confidence < THRESHOLD:
            name = names[label]
            seen_now.add(name)
            streak[name] = streak.get(name, 0) + 1
            if name in marked_today:
                status, color = "Already Marked", (255, 200, 0)
            elif streak[name] >= REQUIRED_FRAMES:
                mark_attendance(name)
                marked_today.add(name)
                status, color = "Attendance Marked", (0, 255, 0)
            else:
                status, color = f"Checking {streak[name]}/{REQUIRED_FRAMES}", (0, 255, 255)
            text = f"{name} - {status}"
        else:
            text, color = f"Unknown ({int(confidence)})", (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # jo banda is frame mein nahi tha uska streak zero
    for n in list(streak):
        if n not in seen_now:
            streak[n] = 0

    cv2.putText(frame, f"Aaj ki attendance: {len(marked_today)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()