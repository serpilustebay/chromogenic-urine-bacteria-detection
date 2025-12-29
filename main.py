import os
import cv2
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO


# ======================================================================
# Helper Function – Majority Vote Classification
# ======================================================================
def classify_bacteria(path, model, thr=0.0):
    """
    Classify the image based on majority vote of detected colonies.
    Returns:
        -1  → No detection
        1   → E. coli
        2   → K. pneumoniae
    """
    if model is None:
        return -1

    try:
        results = model.predict(path)
        if not results:
            return -1
        result = results[0]
        if result is None or result.boxes is None:
            return -1

        classes = []
        for b in result.boxes:
            conf = float(b.conf[0])
            if conf < thr:
                continue
            cls = int(b.cls[0]) + 1
            classes.append(cls)

        if len(classes) == 0:
            return -1
        return max(set(classes), key=classes.count)
    except Exception:
        return -1


# ======================================================================
# Main GUI Application
# ======================================================================
class App:

    def __init__(self, root, model):
        self.root = root
        self.model = model

        win_w, win_h = 1300, 800
        self.root.geometry(f"{win_w}x{win_h}+0+0")
        self.root.resizable(False, False)
        self.root.title("Bacteria Colony Viewer")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Modern.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            background="#2B6CB0",
            foreground="white",
            borderwidth=0,
        )
        self.style.map(
            "Modern.TButton",
            background=[("active", "#1E4E8C")],
            foreground=[("active", "white")]
        )

        self.zoom = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 8.0

        self.annotated_bgr = None
        self.annotated_rgb = None

        self.canvas_max_w = 950
        self.canvas_max_h = 650

        self._build_layout()

    # ==================================================================
    def _build_layout(self):
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(ctrl, text="Load Model (.pt)", style="Modern.TButton",
                   command=self.load_model).pack(side=tk.LEFT, padx=10)

        tk.Label(ctrl, text="Threshold:").pack(side=tk.LEFT)
        self.thr_var = tk.StringVar(value="0.4")  # DoubleVar yerine string: güvenli parse edeceğiz
        tk.Entry(ctrl, textvariable=self.thr_var, width=6).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text="Choose Image", style="Modern.TButton",
                   command=self.choose_image).pack(side=tk.LEFT, padx=10)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        cframe = tk.Frame(main, bd=1, relief=tk.SUNKEN)
        cframe.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas = tk.Canvas(
            cframe, bg="black",
            width=self.canvas_max_w,
            height=self.canvas_max_h
        )
        self.canvas.pack()

        # Pan / Zoom
        self.canvas.bind("<ButtonPress-1>", self.pan_start)
        self.canvas.bind("<B1-Motion>", self.pan_move)

        # Windows/macOS çoğunlukla <MouseWheel>, Linux: Button-4/5
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)     # Windows/macOS
        self.canvas.bind("<Button-4>", self.mouse_zoom_linux) # Linux scroll up
        self.canvas.bind("<Button-5>", self.mouse_zoom_linux) # Linux scroll down

        sframe = tk.Frame(main, bd=2, relief=tk.GROOVE)
        sframe.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Label(sframe, text="Colony Statistics", font=("Arial", 12, "bold")).pack()

        self.lbl_imageclass = tk.Label(
            sframe, text="Image Prediction: -",
            fg="blue", font=("Arial", 12, "bold")
        )
        self.lbl_imageclass.pack(anchor="w", pady=(0, 10))

        self.lbl_total = tk.Label(sframe, text="Total colonies: 0")
        self.lbl_total.pack(anchor="w")

        self.lbl_id1 = tk.Label(sframe, text="E. coli (ID1): 0")
        self.lbl_id1.pack(anchor="w")

        self.lbl_id2 = tk.Label(sframe, text="K. pneumoniae (ID2): 0")
        self.lbl_id2.pack(anchor="w")

        tk.Label(sframe, text="Detected Colonies", font=("Arial", 12, "bold")).pack(pady=5)

        tree_frame = tk.Frame(sframe)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("id", "prob", "x1", "y1", "x2", "y2"),
            show="headings",
            height=15
        )
        for col in ["id", "prob", "x1", "y1", "x2", "y2"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=60, anchor="center")

        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical",
                                 command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btm = tk.Frame(self.root)
        btm.pack(side=tk.BOTTOM, pady=5)

        ttk.Button(btm, text="Zoom In", style="Modern.TButton",
                   command=self.zoom_in).pack(side=tk.LEFT, padx=5)

        ttk.Button(btm, text="Zoom Out", style="Modern.TButton",
                   command=self.zoom_out).pack(side=tk.LEFT, padx=5)

        ttk.Button(btm, text="Save", style="Modern.TButton",
                   command=self.save_img).pack(side=tk.LEFT, padx=5)

        ttk.Button(btm, text="Quit", style="Modern.TButton",
                   command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

    # ==================================================================
    def _get_threshold(self) -> float:
        """Threshold değerini güvenli biçimde parse eder, hatalıysa 0.4'e döner."""
        raw = (self.thr_var.get() or "").strip().replace(",", ".")
        try:
            thr = float(raw)
        except ValueError:
            thr = 0.4
            self.thr_var.set("0.4")
        # sınırla
        if thr < 0.0:
            thr = 0.0
            self.thr_var.set("0.0")
        if thr > 1.0:
            thr = 1.0
            self.thr_var.set("1.0")
        return thr

    # ==================================================================
    def load_model(self):
        pt_path = filedialog.askopenfilename(
            title="Select YOLO Model (.pt)",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
        )
        if not pt_path:
            messagebox.showwarning("Model Not Loaded", "No model file selected.")
            return

        try:
            self.model = YOLO(pt_path)
            messagebox.showinfo("Model Loaded", f"Model loaded successfully:\n{pt_path}")
        except Exception as e:
            self.model = None
            messagebox.showerror("Error Loading Model", f"Could not load model:\n{e}")

    # ==================================================================
    def update_canvas(self):
        if self.annotated_rgb is None:
            return

        try:
            h, w = self.annotated_rgb.shape[:2]
            nw, nh = int(w * self.zoom), int(h * self.zoom)
            nw = max(1, nw)
            nh = max(1, nh)

            resized = cv2.resize(self.annotated_rgb, (nw, nh))
            tk_img = ImageTk.PhotoImage(Image.fromarray(resized))

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
            self.canvas.image = tk_img
            self.canvas.config(scrollregion=(0, 0, nw, nh))
        except Exception as e:
            messagebox.showerror("Canvas Error", f"Could not render image:\n{e}")

    # ==================================================================
    def pan_start(self, e):
        self.canvas.scan_mark(e.x, e.y)

    def pan_move(self, e):
        self.canvas.scan_dragto(e.x, e.y, gain=1)

    def mouse_zoom(self, e):
        try:
            if e.delta > 0:
                self.zoom = min(self.max_zoom, self.zoom * 1.1)
            else:
                self.zoom = max(self.min_zoom, self.zoom / 1.1)
            self.update_canvas()
        except Exception:
            pass

    def mouse_zoom_linux(self, e):
        # Linux: Button-4 scroll up, Button-5 scroll down
        try:
            if e.num == 4:
                self.zoom = min(self.max_zoom, self.zoom * 1.1)
            elif e.num == 5:
                self.zoom = max(self.min_zoom, self.zoom / 1.1)
            self.update_canvas()
        except Exception:
            pass

    def zoom_in(self):
        self.zoom = min(self.max_zoom, self.zoom * 1.1)
        self.update_canvas()

    def zoom_out(self):
        self.zoom = max(self.min_zoom, self.zoom / 1.1)
        self.update_canvas()

    # ==================================================================
    def choose_image(self):
        if self.model is None:
            messagebox.showwarning("No Model Loaded", "Please load a .pt model first.")
            return

        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.run_prediction(path)

    # ==================================================================
    def run_prediction(self, path):
        if self.model is None:
            messagebox.showwarning("Model Required", "Load a model before predicting.")
            return

        thr = self._get_threshold()

        try:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Image Error", "Could not read image file (cv2.imread returned None).")
                return

            results = self.model.predict(path)
            if not results:
                messagebox.showwarning("Prediction", "No result returned by the model.")
                self._clear_visuals()
                return

            result = results[0]
            if result is None or result.boxes is None:
                messagebox.showwarning("Prediction", "No detections found.")
                self._clear_visuals()
                return

            annotated = img.copy()
            rows = []
            colors = {1: (0, 255, 0), 2: (255, 0, 255)}

            for b in result.boxes:
                try:
                    conf = float(b.conf[0])
                    if conf < thr:
                        continue

                    cls = int(b.cls[0]) + 1
                    x1, y1, x2, y2 = map(int, b.xyxy[0])

                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2),
                        colors.get(cls, (255, 0, 0)), 2
                    )

                    text = f"ID{cls} ({conf:.2f})"
                    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    ty = y1 - 10 if y1 - 10 > th else y1 + th + 10

                    cv2.rectangle(
                        annotated,
                        (x1, ty - th - base),
                        (x1 + tw + 6, ty + base),
                        colors.get(cls, (255, 0, 0)),
                        -1
                    )
                    cv2.putText(
                        annotated, text, (x1 + 3, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                    )

                    rows.append([cls, round(conf, 2), x1, y1, x2, y2])
                except Exception:
                    # tek bir bbox bozuksa tüm tahmini çöpe atma
                    continue

            self.annotated_bgr = annotated
            self.annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            self.update_tree(rows)

            # Fit image to canvas
            h, w = self.annotated_rgb.shape[:2]
            if w > 0 and h > 0:
                self.zoom = min(self.canvas_max_w / w, self.canvas_max_h / h, 1.0)
            else:
                self.zoom = 1.0
            self.update_canvas()

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed:\n{e}")
            self._clear_visuals()

    def _clear_visuals(self):
        self.annotated_bgr = None
        self.annotated_rgb = None
        self.canvas.delete("all")
        self.update_tree([])

    # ==================================================================
    def update_tree(self, rows):
        for item in self.tree.get_children():
            self.tree.delete(item)

        # boşsa bile DataFrame güvenli
        df = pd.DataFrame(rows, columns=["id", "prob", "x1", "y1", "x2", "y2"]) if rows else pd.DataFrame(
            columns=["id", "prob", "x1", "y1", "x2", "y2"]
        )

        for r in rows:
            self.tree.insert("", "end", values=r)

        self.lbl_total.config(text=f"Total colonies: {len(df)}")
        self.lbl_id1.config(text=f"E. coli (ID1): {(df['id'] == 1).sum() if len(df) else 0}")
        self.lbl_id2.config(text=f"K. pneumoniae (ID2): {(df['id'] == 2).sum() if len(df) else 0}")

        if len(df) == 0:
            self.lbl_imageclass.config(text="Image Prediction: No detection", fg="red")
        else:
            pred = df["id"].value_counts().idxmax()
            label = "E. coli (ID1)" if pred == 1 else "K. pneumoniae (ID2)"
            self.lbl_imageclass.config(text=f"Image Prediction: {label}", fg="#0A4BD3")

    # ==================================================================
    def save_img(self):
        if self.annotated_bgr is None:
            messagebox.showwarning("Nothing to Save", "Run a prediction first.")
            return

        fname = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if not fname:
            return

        try:
            ok = cv2.imwrite(fname, self.annotated_bgr)
            if not ok:
                messagebox.showerror("Save Error", "cv2.imwrite failed (returned False).")
                return
            messagebox.showinfo("Image Saved", f"Image saved successfully:\n{fname}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save image:\n{e}")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, model=None)
    root.mainloop()
