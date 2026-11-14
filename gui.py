import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import glob
from ultralytics import YOLO

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

        base_dir = os.path.dirname(os.path.abspath(__file__))
        runs_dir = os.path.join(base_dir, "runs")
        model_path = os.path.join(
            runs_dir,
            "dice_yolo26n_adamw_lr0.001_wd0.0005",
            "weights",
            "best.pt"
        )
        self.yolo_model = YOLO(model_path)

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
            result, calculated_total = opencv_method(self.image_path, False)
            cv2.imwrite("temp_result.jpg", result)
            self.show_image("temp_result.jpg")
        elif self.method.get() == "yolo":
            result, calculated_total = self.yolo_method(self.image_path)
            cv2.imwrite("temp_result.jpg", result)
            self.show_image("temp_result.jpg")

    def yolo_method(self, path):
        results = self.yolo_model.predict(
            source=path,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot(
            boxes=True,
            labels=True,
            conf=False
        )

        if result.boxes is not None and result.boxes.cls is not None:
            class_ids = result.boxes.cls.tolist()
            calculated_total = sum(int(class_id) + 1 for class_id in class_ids)
        else:
            calculated_total = 0

        cv2.putText(
            annotated_image,
            f"Total: {calculated_total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        return annotated_image, calculated_total

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

        correct_predictions_opencv = 0
        incorrect_predictions_opencv = 0
        correct_predictions_yolo = 0
        incorrect_predictions_yolo = 0
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
                result_data_opencv = opencv_method(img_path, debug=False)
                if isinstance(result_data_opencv, tuple):
                    _, calculated_total_opencv = result_data_opencv
                else:
                    calculated_total_opencv = 0

                if calculated_total_opencv == expected_total:
                    correct_predictions_opencv += 1
                else:
                    incorrect_predictions_opencv += 1
            except Exception:
                incorrect_predictions_opencv += 1

            try:
                result_data_yolo = self.yolo_method(img_path)
                if isinstance(result_data_yolo, tuple):
                    _, calculated_total_yolo = result_data_yolo
                else:
                    calculated_total_yolo = 0

                if calculated_total_yolo == expected_total:
                    correct_predictions_yolo += 1
                else:
                    incorrect_predictions_yolo += 1
            except Exception:
                incorrect_predictions_yolo += 1

        self.root.config(cursor="")
        total_images = len(image_paths)
        evaluated_total = total_images - skipped_images
        correct_ratio_opencv = (correct_predictions_opencv / evaluated_total) * 100 if evaluated_total > 0 else 0
        incorrect_ratio_opencv = (incorrect_predictions_opencv / evaluated_total) * 100 if evaluated_total > 0 else 0
        correct_ratio_yolo = (correct_predictions_yolo / evaluated_total) * 100 if evaluated_total > 0 else 0
        incorrect_ratio_yolo = (incorrect_predictions_yolo / evaluated_total) * 100 if evaluated_total > 0 else 0

        result_msg = (
            f"Összes kép a mappában: {len(image_paths)} db\n"
            f"Átugrott (nincs label): {skipped_images} db\n"
            f"Kiértékelt képek: {evaluated_total} db\n\n"
            f"── OpenCV eredmények ──\n"
            f"Helyes felismerés: {correct_predictions_opencv} db ({correct_ratio_opencv:.2f}%)\n"
            f"Helytelen felismerés: {incorrect_predictions_opencv} db ({incorrect_ratio_opencv:.2f}%)\n\n"
            f"── YOLO eredmények ──\n"
            f"Helyes felismerés: {correct_predictions_yolo} db ({correct_ratio_yolo:.2f}%)\n"
            f"Helytelen felismerés: {incorrect_predictions_yolo} db ({incorrect_ratio_yolo:.2f}%)"
        )
        messagebox.showinfo("Kiértékelés eredménye", result_msg)
