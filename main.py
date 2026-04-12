import sys, os, cv2, numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
import mediapipe as mp
import pyttsx3

# ---------------- Global Settings ----------------
MODEL_PATH = Path("best.pt")  # YOLO model path
MODEL = None
SAVED_DIR = Path("history")
SAVED_DIR.mkdir(exist_ok=True)

# Class names and operator mapping
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'divide', 'minus', 'multiplication']
OPERATOR_MAP = {'add': '+', 'minus': '-', 'multiplication': '*', 'divide': '/'}

# ---------------- Helper Functions ----------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def save_history(expr, result):
    file = SAVED_DIR / "history.txt"
    with open(file, "a") as f:
        f.write(f"{expr} = {result}\n")

def load_history():
    file = SAVED_DIR / "history.txt"
    if not file.exists(): return []
    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]

def delete_history(selected):
    file = SAVED_DIR / "history.txt"
    history = load_history()
    history = [line for line in history if line not in selected]
    with open(file, "w") as f:
        for line in history: f.write(line+"\n")

# ---------------- Widgets ----------------
class WelcomeWidget(QWidget):
    def __init__(self, goto_user, goto_admin):
        super().__init__()
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#0b3954"))
        self.setPalette(pal)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Smart Air Writing")
        title.setFont(QFont("Arial",36,QFont.Bold))
        title.setStyleSheet("color:#ffdd00;")
        title.setAlignment(Qt.AlignCenter)

        quote = QLabel("“Write in the air — let your hands do the talking.”")
        quote.setFont(QFont("Arial",16))
        quote.setStyleSheet("color:#e0f7fa;font-style:italic;")
        quote.setAlignment(Qt.AlignCenter)

        btn_user = QPushButton("Continue as User")
        btn_admin = QPushButton("Admin Login")
        for b in (btn_user, btn_admin):
            b.setFixedSize(280,60)
            b.setStyleSheet(
                "background:#ff6f61;color:white;font-weight:700;border-radius:10px;"
            )
        btn_user.clicked.connect(goto_user)
        btn_admin.clicked.connect(goto_admin)

        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(quote)
        layout.addSpacing(30)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(btn_user)
        row.addSpacing(20)
        row.addWidget(btn_admin)
        row.addStretch()
        layout.addLayout(row)
        self.setLayout(layout)

        QTimer.singleShot(700, lambda: speak("Welcome to Smart Air Writing. Choose User or Admin to continue."))

class UserWidget(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        self.btn_back = QPushButton("← Back")
        self.btn_back.setFixedHeight(35)
        self.btn_back.setStyleSheet("background:#ff6f61;color:white;font-weight:700;border-radius:5px;")
        top_layout.addWidget(self.btn_back)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        layout = QHBoxLayout()
        # ---------------- Left: Camera Preview ----------------
        left = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(1000,750)
        self.preview_label.setStyleSheet("background:black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.preview_label)

        # ---------------- Canvas Overlay ----------------
        self.canvas_image = np.zeros((750,1000,3),dtype=np.uint8)
        self.prev_pt = None
        self.canvas_label = QLabel()
        self.canvas_label.setFixedSize(1000,750)
        self.canvas_label.setStyleSheet("background:transparent;position:absolute;")
        left.addWidget(self.canvas_label)

        # ---------------- Buttons ----------------
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_process = QPushButton("Process Expression")
        self.btn_erase = QPushButton("Erase Canvas")
        self.btn_save = QPushButton("Save Result")
        for b in (self.btn_start,self.btn_stop,self.btn_process,self.btn_erase,self.btn_save):
            b.setFixedHeight(40)
            b.setStyleSheet(
                "background:#ff6f61;color:white;font-weight:700;border-radius:5px;"
            )
            btn_layout.addWidget(b)
        left.addLayout(btn_layout)
        layout.addLayout(left)
        main_layout.addLayout(layout)
        self.setLayout(main_layout)

        # ---------------- Mediapipe Hand Tracking ----------------
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._grab_frame)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

        # ---------------- Button Connections ----------------
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_erase.clicked.connect(self.erase_canvas)
        self.btn_process.clicked.connect(self.process_expression)
        self.btn_save.clicked.connect(self.save_result)

    def start_camera(self):
        if self.cap: return
        self.cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.timer.start(30)

    def stop_camera(self):
        if self.cap: 
            self.timer.stop()
            self.cap.release()
            self.cap=None
            self.preview_label.clear()
            self.canvas_label.clear()

    def erase_canvas(self):
        self.canvas_image.fill(0)
        self.canvas_label.clear()

    def save_result(self):
        temp_file = SAVED_DIR / "saved_result.png"
        cv2.imwrite(str(temp_file), self.canvas_image)
        QMessageBox.information(self, "Saved", f"Canvas saved to {temp_file}")

    def _grab_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame,1)
        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(display, handLms, self.mp_hands.HAND_CONNECTIONS)
                tip = handLms.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                dip = handLms.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
                fx, fy = int(tip.x*w_frame), int(tip.y*h_frame)
                cx, cy = int(tip.x*1000), int(tip.y*750)
                writing = tip.y < dip.y
                if writing:
                    if self.prev_pt: cv2.line(self.canvas_image,self.prev_pt,(cx,cy),(255,255,255),12)
                    self.prev_pt=(cx,cy)
                else:
                    self.prev_pt=None
                cv2.circle(display,(fx,fy),8,(0,255,128),-1)

        overlay=cv2.resize(self.canvas_image,(w_frame,h_frame))
        blended=cv2.addWeighted(display,0.66,overlay,0.34,0)
        rgb=cv2.cvtColor(blended,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        qimg=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        pix=QPixmap.fromImage(qimg).scaled(self.preview_label.width(),self.preview_label.height(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    # ---------------- Updated expression processing ----------------
    def process_expression(self):
        if MODEL is None:
            QMessageBox.information(self,"YOLO Model","YOLO model not loaded. Cannot process expression.")
            return

        temp_file=SAVED_DIR/"temp_expr.png"
        cv2.imwrite(str(temp_file),self.canvas_image)

        try:
            results = MODEL(str(temp_file), stream=True)
            detected_elements = []

            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes:
                    for b in boxes:
                        x1, y1, x2, y2 = b.xyxy[0]
                        x1, x2 = int(x1), int(x2)
                        cls_idx = int(b.cls[0])
                        class_name = CLASS_NAMES[cls_idx]
                        center_x = (x1 + x2) // 2
                        detected_elements.append((center_x, class_name))

            # Sort left-to-right
            detected_elements.sort(key=lambda x: x[0])

            # Merge very close symbols to prevent duplicates
            merged_elements = []
            margin = 10
            for x, cls in detected_elements:
                if merged_elements and abs(x - merged_elements[-1][0]) < margin:
                    continue
                merged_elements.append((x, cls))

            # Build expression
            expression = ""
            for _, cls_name in merged_elements:
                if cls_name in OPERATOR_MAP:
                    expression += OPERATOR_MAP[cls_name]
                else:
                    expression += cls_name

            # Evaluate
            try:
                result = eval(expression) if expression else "Error"
            except:
                result = "Error"

            save_history(expression, result)
            QMessageBox.information(self,"Expression Result",f"{expression} = {result}",QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

class AdminWidget(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        self.btn_back = QPushButton("← Back")
        self.btn_back.setFixedHeight(35)
        self.btn_back.setStyleSheet("background:#ff6f61;color:white;font-weight:700;border-radius:5px;")
        top_layout.addWidget(self.btn_back)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        self.table = QListWidget()
        self.load_history()
        main_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setFixedHeight(40)
        self.btn_delete.setStyleSheet("background:#ff6f61;color:white;font-weight:700;border-radius:5px;")
        btn_layout.addWidget(self.btn_delete)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        self.btn_delete.clicked.connect(self.delete_selected)

    def load_history(self):
        self.table.clear()
        for item in load_history():
            self.table.addItem(item)

    def delete_selected(self):
        selected = [i.text() for i in self.table.selectedItems()]
        delete_history(selected)
        self.load_history()

# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Air Writing")
        self.setGeometry(50,50,1200,900)

        global MODEL
        try:
            MODEL = YOLO(str(MODEL_PATH))
        except:
            MODEL = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.welcome_widget = WelcomeWidget(self.goto_user, self.goto_admin)
        self.user_widget = UserWidget()
        self.admin_widget = AdminWidget()

        self.stack.addWidget(self.welcome_widget)
        self.stack.addWidget(self.user_widget)
        self.stack.addWidget(self.admin_widget)

        self.stack.setCurrentWidget(self.welcome_widget)

        # Connect back buttons
        self.user_widget.btn_back.clicked.connect(self.back_to_welcome)
        self.admin_widget.btn_back.clicked.connect(self.back_to_welcome)

    def goto_user(self):
        password, ok = QInputDialog.getText(self,"User Login","Enter User Password:", QLineEdit.Password)
        if ok and password=="user123":
            self.stack.setCurrentWidget(self.user_widget)
        else:
            QMessageBox.warning(self,"Error","Incorrect Password!")

    def goto_admin(self):
        password, ok = QInputDialog.getText(self,"Admin Login","Enter Admin Password:", QLineEdit.Password)
        if ok and password=="admin123":
            self.stack.setCurrentWidget(self.admin_widget)
        else:
            QMessageBox.warning(self,"Error","Incorrect Password!")

    def back_to_welcome(self):
        self.stack.setCurrentWidget(self.welcome_widget)

# ---------------- Run Application ----------------
if __name__=="__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
