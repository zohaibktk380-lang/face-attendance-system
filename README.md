# Face Attendance System

A face recognition attendance system built with Python, OpenCV, Tkinter and SQLite.

## Features
- Register new students using the webcam (auto-captures 15 photos)
- Real-time face recognition (OpenCV LBPH)
- Duplicate-entry prevention (one attendance per person per day)
- SQLite database for students and attendance
- Monthly Excel report with Present/Absent marking and percentage
- Tkinter GUI

## Tech Stack
Python, OpenCV, NumPy, Tkinter, SQLite, openpyxl

## Installation
```bash
git clone https://github.com/YOUR-USERNAME/face-attendance-system.git
cd face-attendance-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Usage
1. Click **Register New Student** and look at the camera
2. Click **Start Attendance** to mark attendance
3. Click **Excel Report** to generate the monthly report

## Note
The `known_faces/` folder (face photos) is not included for privacy reasons. It is created automatically when you register a student.