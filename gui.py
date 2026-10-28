import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

from opencv_processing import opencv_method


class DiceGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Dice Project")
        self.root.geometry("1000x700")

        self.image_path = None

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Button(
            control_frame,
            text="Kép betöltése",
            command=self.load_image
        ).pack(fill=tk.X)

        self.method = tk.StringVar(value="opencv")

        tk.Label(control_frame, text="Feldolgozás:").pack(pady=10)

        tk.Radiobutton(
            control_frame,
            text="OpenCV",
            variable=self.method,
            value="opencv"
        ).pack(anchor="w")

        tk.Radiobutton(
            control_frame,
            text="YOLO",
            variable=self.method,
            value="yolo"
        ).pack(anchor="w")

        tk.Button(
            control_frame,
            text="Futtatás",
            command=self.run_processing
        ).pack(fill=tk.X, pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack(expand=True)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
        )

        if self.image_path:
            self.show_image(self.image_path)

    def show_image(self, path):
        img = Image.open(path)

        img.thumbnail((800, 600))

        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)

    def run_processing(self):
        if not self.image_path:
            return

        if self.method.get() == "opencv":
            result = opencv_method(self.image_path, True)
        else:
            result = self.yolo_method(self.image_path)

        cv2.imwrite("temp_result.jpg", result)
        self.show_image("temp_result.jpg")

    def yolo_method(self, path):
        # TODO : Átírni, implementálni egy külön fájlba
        pass
