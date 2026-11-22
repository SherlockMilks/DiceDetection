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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
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

    def show_evaluation_result(self, result_msg):
        window = tk.Toplevel(self.root)
        window.title("Kiértékelés eredménye")

        text = tk.Text(
            window,
            wrap="word",
        )
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", result_msg)
        text.config(state="disabled")

        def save_result():
            file_path = filedialog.asksaveasfilename(
                title="Eredmények mentése",
                defaultextension=".txt",
                filetypes=[("Szöveges fájl", "*.txt")],
                initialdir=self.base_dir,
                initialfile="dice_evaluation_results.txt"
            )
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(result_msg)

                messagebox.showinfo(
                    "Mentés",
                    "Az eredmények sikeresen mentésre kerültek!",
                    parent=window
                )

            except Exception as e:
                messagebox.showerror(
                    "Mentési hiba",
                    f"Nem sikerült menteni a fájlt:\n{e}",
                    parent=window
                )

        button_frame = tk.Frame(window)
        button_frame.pack(pady=(0, 10))

        tk.Button(
            button_frame,
            text="Mentés TXT-be",
            command=save_result,
            width=12
        ).pack(side="left", padx=5)

        tk.Button(
            button_frame,
            text="Bezárás",
            command=window.destroy,
            width=12
        ).pack(side="left", padx=5)

    def run_processing(self):
        if not self.image_path:
            return

        if self.method.get() == "opencv":
            # Szétbontjuk a visszakapott tuple-t (kép és összeg)
            result, calculated_total, _ = opencv_method(self.image_path, False)
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

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls.item()),
                    "bbox": box.xywhn[0].tolist(),
                    "confidence": float(box.conf.item())
                })

        calculated_total = sum(
            detection["class_id"] + 1 for detection in detections
        )

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

        return annotated_image, detections

    def get_objects_from_yolo(self, label_path):
        objects = []

        with open(label_path, "r") as file:
            for line in file:
                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                objects.append({
                    "class_id": class_id,
                    "bbox": (x_center, y_center, width, height)
                })
        return objects

    def calculate_iou(self, box1, box2):
        def to_xyxy(box):
            x, y, w, h = box
            return (
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2
            )

        x1_min, y1_min, x1_max, y1_max = to_xyxy(box1)
        x2_min, y2_min, x2_max, y2_max = to_xyxy(box2)

        intersection_width = max(
            0, min(x1_max, x2_max) - max(x1_min, x2_min)
        )
        intersection_height = max(
            0, min(y1_max, y2_max) - max(y1_min, y2_min)
        )

        intersection = intersection_width * intersection_height

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_objects(self, ground_truth, detections, iou_threshold=0.5):
        matched_gt = set()
        matched_det = set()

        candidates = []

        for det_idx, detection in enumerate(detections):
            for gt_idx, gt in enumerate(ground_truth):
                if detection["class_id"] != gt["class_id"]:
                    continue

                iou = self.calculate_iou(
                    detection["bbox"],
                    gt["bbox"]
                )

                if iou >= iou_threshold:
                    candidates.append((iou, det_idx, gt_idx))

        candidates.sort(reverse=True)

        for iou, det_idx, gt_idx in candidates:
            if det_idx not in matched_det and gt_idx not in matched_gt:
                matched_det.add(det_idx)
                matched_gt.add(gt_idx)

        tp = len(matched_det)
        fp = len(detections) - tp
        fn = len(ground_truth) - tp

        return tp, fp, fn

    def evaluate_dataset(self):
        images_dir = os.path.join(self.base_dir, "DiceDetection", "data", "dice_yolo", "images", "test")
        labels_dir = os.path.join(self.base_dir, "DiceDetection", "data", "dice_yolo", "labels", "test")

        if not os.path.exists(images_dir):
            images_dir = os.path.join(self.base_dir, "data", "dice_yolo", "images", "test")
            labels_dir = os.path.join(self.base_dir, "data", "dice_yolo", "labels", "test")

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

        skipped_images = 0
        correct_images_opencv = 0
        tp_opencv = 0
        fp_opencv = 0
        fn_opencv = 0

        correct_images_yolo = 0
        tp_yolo = 0
        fp_yolo = 0
        fn_yolo = 0


        for img_path in image_paths:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]

            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            ground_truth = self.get_objects_from_yolo(label_path)

            if ground_truth is None:
                skipped_images += 1
                continue


            _, _, opencv_detections = opencv_method(
                img_path,
                debug=False
            )

            tp, fp, fn = self.evaluate_objects(
                ground_truth,
                opencv_detections,
                iou_threshold=0.5
            )

            tp_opencv += tp
            fp_opencv += fp
            fn_opencv += fn

            if tp == len(ground_truth) and fp == 0 and fn == 0:
                correct_images_opencv += 1


            _, yolo_detections = self.yolo_method(img_path)

            tp, fp, fn = self.evaluate_objects(
                ground_truth,
                yolo_detections,
                iou_threshold=0.5
            )

            tp_yolo += tp
            fp_yolo += fp
            fn_yolo += fn

            if tp == len(ground_truth) and fp == 0 and fn == 0:
                correct_images_yolo += 1

        precision_opencv = (
            tp_opencv / (tp_opencv + fp_opencv) * 100
            if tp_opencv + fp_opencv > 0 else 0
        )

        recall_opencv = (
            tp_opencv / (tp_opencv + fn_opencv) * 100
            if tp_opencv + fn_opencv > 0 else 0
        )


        precision_yolo = (
            tp_yolo / (tp_yolo + fp_yolo) * 100
            if tp_yolo + fp_yolo > 0 else 0
        )

        recall_yolo = (
            tp_yolo / (tp_yolo + fn_yolo) * 100
            if tp_yolo + fn_yolo > 0 else 0
        )

        self.root.config(cursor="")
        total_images = len(image_paths)
        evaluated_total = total_images - skipped_images

        correct_ratio_opencv = (correct_images_opencv / evaluated_total) * 100 if evaluated_total > 0 else 0
        incorrect_images_opencv = evaluated_total - correct_images_opencv
        incorrect_ratio_opencv = (incorrect_images_opencv / evaluated_total) * 100 if evaluated_total > 0 else 0

        correct_ratio_yolo = (correct_images_yolo / evaluated_total) * 100 if evaluated_total > 0 else 0
        incorrect_images_yolo = evaluated_total - correct_images_yolo
        incorrect_ratio_yolo = (incorrect_images_yolo / evaluated_total) * 100 if evaluated_total > 0 else 0

        result_msg = (
            f"Összes kép a mappában: {len(image_paths)} db\n"
            f"Átugrott: {skipped_images} db\n"
            f"Kiértékelt képek: {evaluated_total} db\n\n"
            
            f"── OPENCV eredmények ──\n"
            f"Precision: {precision_opencv:.2f}%\n"
            f"Recall: {recall_opencv:.2f}%\n"
            f"Helyes képek: {correct_images_opencv} db ({correct_ratio_opencv:.2f}%)\n"
            f"Helytelen képek: {incorrect_images_opencv} db ({incorrect_ratio_opencv:.2f}%)"
            
            f"\n\n── YOLO eredmények ──\n"
            f"Precision: {precision_yolo:.2f}%\n"
            f"Recall: {recall_yolo:.2f}%\n"
            f"Helyes képek: {correct_images_yolo} db ({correct_ratio_yolo:.2f}%)\n"
            f"Helytelen képek: {incorrect_images_yolo} db ({incorrect_ratio_yolo:.2f}%)"
        )
        self.show_evaluation_result(result_msg)
