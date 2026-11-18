import cv2
import os
import time
import sqlite3
import datetime
import calendar
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

DATASET = "known_faces"
DB_FILE = "attendance.db"
THRESHOLD = 70
REQUIRED_FRAMES = 8
PHOTOS_TO_CAPTURE = 15

os.makedirs(DATASET, exist_ok=True)

# ---------------- Database ----------------
def db():
    return sqlite3.connect(DB_FILE)


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS students(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        created TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        UNIQUE(name, date))""")


def sync_students():
    """known_faces ke folders ko students table mein daalo."""
    with db() as c:
        for item in os.listdir(DATASET):
            if os.path.isdir(os.path.join(DATASET, item)):
                c.execute("INSERT OR IGNORE INTO students(name, created) VALUES(?,?)",
                          (item, datetime.date.today().isoformat()))


# ---------------- Face detector ----------------
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
        urllib.request.urlretrieve(url, cascade_path)

detector = cv2.CascadeClassifier(cascade_path)


# ---------------- Training ----------------
def train_model():
    faces, labels, names = [], [], {}
    label_id = 0
    for person in sorted(os.listdir(DATASET)):
        p = os.path.join(DATASET, person)
        if not os.path.isdir(p):
            continue
        count = 0
        for f in sorted(os.listdir(p)):
            img = cv2.imread(os.path.join(p, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            if max(h, w) > 800:
                s = 800 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            found = detector.detectMultiScale(img, 1.1, 6, minSize=(80, 80))
            if len(found) == 0:
                continue
            x, y, fw, fh = max(found, key=lambda r: r[2] * r[3])
            face = cv2.equalizeHist(cv2.resize(img[y:y+fh, x:x+fw], (200, 200)))
            faces.append(face)
            labels.append(label_id)
            count += 1
        if count > 0:
            names[label_id] = person
            label_id += 1
        else:
            labels_fix = None  # is banday ka koi chehra nahi mila
    if not faces:
        return None, {}
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels))
    return rec, names


# ---------------- Register ----------------
def register_student():
    name = simpledialog.askstring("Naya Banda", "Banday ka naam likho (English mein):")
    if not name or not name.strip():
        return
    name = name.strip()
    folder = os.path.join(DATASET, name)
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Camera nahi khula.")
        return

    existing = len(os.listdir(folder))
    saved, last = 0, 0
    while saved < PHOTOS_TO_CAPTURE:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = detector.detectMultiScale(gray, 1.2, 6, minSize=(100, 100))
        if len(found) > 0 and time.time() - last > 0.7:
            x, y, w, h = max(found, key=lambda r: r[2] * r[3])
            cv2.imwrite(os.path.join(folder, f"cam_{existing + saved + 1}.png"), frame)
            saved += 1
            last = time.time()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}: {saved}/{PHOTOS_TO_CAPTURE} - chehra ghumao",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

    if saved > 0:
        with db() as c:
            c.execute("INSERT OR IGNORE INTO students(name, created) VALUES(?,?)",
                      (name, datetime.date.today().isoformat()))
        messagebox.showinfo("Mukammal", f"{name} ki {saved} photos save ho gayin.")
    else:
        messagebox.showwarning("Photos nahi li", "Koi photo save nahi hui.")


# ---------------- Attendance ----------------
def start_attendance():
    rec, names = train_model()
    if rec is None:
        messagebox.showerror("Error", "Koi chehra train nahi hua. Pehle banday register karo.")
        return

    today = datetime.date.today().isoformat()
    with db() as c:
        marked = {r[0] for r in c.execute("SELECT name FROM attendance WHERE date=?", (today,))}

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Camera nahi khula.")
        return

    streak = {}
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen = set()

        for (x, y, w, h) in detector.detectMultiScale(gray, 1.2, 6, minSize=(80, 80)):
            roi = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
            label, conf = rec.predict(roi)
            if conf < THRESHOLD:
                name = names[label]
                seen.add(name)
                streak[name] = streak.get(name, 0) + 1
                if name in marked:
                    status, color = "Already Marked", (255, 200, 0)
                elif streak[name] >= REQUIRED_FRAMES:
                    now = datetime.datetime.now()
                    with db() as c:
                        c.execute("INSERT OR IGNORE INTO attendance(name,date,time) VALUES(?,?,?)",
                                  (name, now.date().isoformat(), now.strftime("%H:%M:%S")))
                    marked.add(name)
                    status, color = "Attendance Marked", (0, 255, 0)
                else:
                    status, color = f"Checking {streak[name]}/{REQUIRED_FRAMES}", (0, 255, 255)
                text = f"{name} - {status}"
            else:
                text, color = f"Unknown ({int(conf)})", (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for n in list(streak):
            if n not in seen:
                streak[n] = 0

        cv2.putText(frame, f"Aaj ki attendance: {len(marked)}  (band: q)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


# ---------------- Tables dekhna ----------------
def show_table(title, columns, rows):
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("520x380")
    tree = ttk.Treeview(win, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")
    for r in rows:
        tree.insert("", "end", values=r)
    sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")


def view_today():
    today = datetime.date.today().isoformat()
    with db() as c:
        rows = c.execute("SELECT name, date, time FROM attendance WHERE date=? ORDER BY time",
                         (today,)).fetchall()
    show_table(f"Aaj ki Attendance ({today})", ("Name", "Date", "Time"), rows)


def view_students():
    sync_students()
    with db() as c:
        rows = c.execute("SELECT id, name, created FROM students ORDER BY name").fetchall()
    show_table("Registered Students", ("ID", "Name", "Registered On"), rows)


def view_all():
    with db() as c:
        rows = c.execute("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC").fetchall()
    show_table("Poori Attendance History", ("Name", "Date", "Time"), rows)


# ---------------- Excel report ----------------
def excel_report():
    default = datetime.date.today().strftime("%Y-%m")
    m = simpledialog.askstring("Excel Report", "Mahina likho (YYYY-MM):", initialvalue=default)
    if not m:
        return
    try:
        year, month = map(int, m.split("-"))
        days_in_month = calendar.monthrange(year, month)[1]
    except Exception:
        messagebox.showerror("Error", "Mahina aise likho: 2026-11")
        return

    today = datetime.date.today()
    if (year, month) == (today.year, today.month):
        counted_days = today.day
    elif (year, month) > (today.year, today.month):
        messagebox.showerror("Error", "Ye mahina abhi aaya nahi.")
        return
    else:
        counted_days = days_in_month

    sync_students()
    with db() as c:
        students = [r[0] for r in c.execute("SELECT name FROM students ORDER BY name")]
        rows = c.execute("SELECT name, date FROM attendance WHERE date LIKE ?",
                         (f"{m}-%",)).fetchall()
    present = {(n, d) for n, d in rows}

    wb = Workbook()
    ws = wb.active
    ws.title = m
    green = PatternFill("solid", fgColor="C6EFCE")
    red = PatternFill("solid", fgColor="FFC7CE")
    head = PatternFill("solid", fgColor="1F4E78")

    headers = ["Name"] + list(range(1, days_in_month + 1)) + ["Present", "Absent", "Percentage"]
    ws.append(headers)
    for i in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=i)
        cell.fill = head
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center")

    for r_i, s in enumerate(students, start=2):
        ws.cell(row=r_i, column=1, value=s)
        pcount = 0
        for d in range(1, days_in_month + 1):
            cell = ws.cell(row=r_i, column=d + 1)
            cell.alignment = Alignment(horizontal="center")
            if d > counted_days:
                cell.value = "-"
                continue
            date_str = f"{m}-{d:02d}"
            if (s, date_str) in present:
                cell.value, cell.fill = "P", green
                pcount += 1
            else:
                cell.value, cell.fill = "A", red
        ws.cell(row=r_i, column=days_in_month + 2, value=pcount)
        ws.cell(row=r_i, column=days_in_month + 3, value=counted_days - pcount)
        pc = ws.cell(row=r_i, column=days_in_month + 4, value=pcount / counted_days)
        pc.number_format = "0%"

    ws.column_dimensions["A"].width = 20
    for i in range(2, days_in_month + 2):
        ws.column_dimensions[get_column_letter(i)].width = 4
    for i in range(days_in_month + 2, days_in_month + 5):
        ws.column_dimensions[get_column_letter(i)].width = 11
    ws.freeze_panes = "B2"

    fname = f"attendance_report_{m}.xlsx"
    try:
        wb.save(fname)
    except PermissionError:
        messagebox.showerror("Error", f"{fname} Excel mein khuli hui hai. Pehle band karo.")
        return
    messagebox.showinfo("Mukammal", f"Report ban gayi: {fname}")
    try:
        os.startfile(fname)
    except Exception:
        pass


# ---------------- Main window ----------------
init_db()
sync_students()

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("360x430")
root.resizable(False, False)

ttk.Label(root, text="Face Attendance System", font=("Segoe UI", 16, "bold")).pack(pady=15)

buttons = [
    ("Naya Banda Register Karo", register_student),
    ("Attendance Shuru Karo", start_attendance),
    ("Aaj ki Attendance Dekho", view_today),
    ("Registered Students", view_students),
    ("Poori History", view_all),
    ("Excel Report Banao", excel_report),
    ("Band Karo", root.destroy),
]
for text, cmd in buttons:
    ttk.Button(root, text=text, command=cmd).pack(fill="x", padx=40, pady=5, ipady=6)

root.mainloop()