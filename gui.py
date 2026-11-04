import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import glob

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

        tk.Button(
            control_frame,
            text="Összes kép kiértékelése",
            command=self.evaluate_dataset,
            bg="#4CAF50",
            fg="white"
        ).pack(fill=tk.X, pady=10)

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

    # def run_processing(self):
    #     if not self.image_path:
    #         return
    #
    #     if self.method.get() == "opencv":
    #         result = opencv_method(self.image_path, True)
    #     else:
    #         result = self.yolo_method(self.image_path)
    #
    #     cv2.imwrite("temp_result.jpg", result)
    #     self.show_image("temp_result.jpg")

    def run_processing(self):
        if not self.image_path:
            return

        if self.method.get() == "opencv":
            # Szétbontjuk a visszakapott tuple-t (kép és összeg)
            result, calculated_total = opencv_method(self.image_path, True)
        else:
            result = self.yolo_method(self.image_path)

        cv2.imwrite("temp_result.jpg", result)
        self.show_image("temp_result.jpg")

    def yolo_method(self, path):
        # TODO : Átírni, implementálni egy külön fájlba
        pass

    def get_expected_total_from_yolo(self, label_path):
        if not os.path.exists(label_path):
            return None
        total = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    total += (class_id + 1)
        return total

    def evaluate_dataset(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        images_dir = os.path.join(base_dir, "DiceDetection", "data", "original", "images")
        labels_dir = os.path.join(base_dir, "DiceDetection", "data", "original", "labels")

        if not os.path.exists(images_dir):
            images_dir = os.path.join(base_dir, "data", "original", "images")
            labels_dir = os.path.join(base_dir, "data", "original", "labels")

        dir_exists = os.path.exists(images_dir)

        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        if dir_exists:
            for ext in extensions:
                image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

        if len(image_paths) == 0:
            error_msg = (
                f"HIBA: Nem található kép!\n\n"
                f"1. Keresett teljes útvonal:\n{images_dir}\n\n"
                f"2. Létezik ez a mappa a gépen? -> {'IGEN' if dir_exists else 'NEM'}\n\n"
                f"Ha NEM, akkor a mappa szerkezete nem ott van, ahol a Python fut!"
            )
            messagebox.showerror("Útvonal Hiba", error_msg)
            return

        self.root.config(cursor="watch")
        self.root.update()

        correct_predictions = 0
        incorrect_predictions = 0
        skipped_images = 0

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]

            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            expected_total = self.get_expected_total_from_yolo(label_path)

            if expected_total is None:
                skipped_images += 1
                continue

            try:
                result_data = opencv_method(img_path, debug=False)
                if isinstance(result_data, tuple):
                    _, calculated_total = result_data
                else:
                    calculated_total = 0

                if calculated_total == expected_total:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
            except Exception:
                incorrect_predictions += 1

        self.root.config(cursor="")
        total_images = len(image_paths)
        evaluated_total = total_images - skipped_images
        correct_ratio = (correct_predictions / evaluated_total) * 100 if evaluated_total > 0 else 0
        incorrect_ratio = (incorrect_predictions / evaluated_total) * 100 if evaluated_total > 0 else 0

        result_msg = (
            f"Összes kép a mappában: {len(image_paths)} db\n"
            f"Átugrott (nincs label): {skipped_images} db\n"
            f"Kiértékelt képek: {evaluated_total} db\n\n"
            f"Helyes felismerés: {correct_predictions} db ({correct_ratio:.2f}%)\n"
            f"Helytelen felismerés: {incorrect_predictions} db ({incorrect_ratio:.2f}%)"
        )
        messagebox.showinfo("Kiértékelés eredménye", result_msg)
