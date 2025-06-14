from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time
import threading
import cv2
import main
from pathlib import Path

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition System")
        self.window.geometry("1024x768")
        self.window.configure(bg="#2c3e50")
        
        # Variables
        self.img_input_path = StringVar()
        self.dataset_folder_path = StringVar()
        
        # Image references
        self.current_input_image_tk = None
        self.current_output_image_tk = None
        
        self.setup_ui()
        
    def setup_ui(self):
        header_frame = Frame(self.window, bg="#34495e")
        header_frame.pack(fill=X, pady=10)
        
        Label(header_frame, text="Face Recognition System", 
              font=("Helvetica", 24, "bold"), fg="#ecf0f1", bg="#34495e").pack(pady=20)
        
        main_frame = Frame(self.window, bg="#2c3e50")
        main_frame.pack(expand=True, fill=BOTH, padx=20, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_columnconfigure(2, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)
        
        left_frame = Frame(main_frame, bg="#34495e", bd=2, relief=GROOVE, padx=10, pady=10)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        Label(left_frame, text="DATASET FOLDER", font=("Helvetica", 12, "bold"), 
             fg="#ecf0f1", bg="#34495e").pack(pady=(10, 5))
        Button(left_frame, text="Select Dataset", command=self.open_folder, 
              bg="#3498db", fg="white", font=("Helvetica", 10)).pack(pady=5, fill=X)
        Label(left_frame, textvariable=self.dataset_folder_path, fg="#bdc3c7", 
             bg="#34495e", wraplength=200, font=("Helvetica", 9)).pack(pady=(0, 20))
        
        Label(left_frame, text="TEST IMAGE", font=("Helvetica", 12, "bold"), 
             fg="#ecf0f1", bg="#34495e").pack(pady=(10, 5))
        Button(left_frame, text="Select Image", command=self.open_image, 
              bg="#3498db", fg="white", font=("Helvetica", 10)).pack(pady=5, fill=X)
        Label(left_frame, textvariable=self.img_input_path, fg="#bdc3c7", 
             bg="#34495e", wraplength=200, font=("Helvetica", 9)).pack(pady=(0, 20))
        
        Label(left_frame, text="PROCESSING TIME", font=("Helvetica", 12, "bold"), 
             fg="#ecf0f1", bg="#34495e").pack(pady=(20, 5))
        self.execution_time_label = Label(left_frame, text="0.000 seconds", 
                                        font=("Helvetica", 14), fg="#2ecc71", bg="#34495e")
        self.execution_time_label.pack(pady=(0, 20))
        
        center_frame = Frame(main_frame, bg="#34495e", bd=2, relief=GROOVE)
        center_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        Label(center_frame, text="INPUT IMAGE", font=("Helvetica", 14, "bold"), 
             fg="#ecf0f1", bg="#34495e").pack(pady=10)
        
        self.img_input_canvas = Canvas(center_frame, bg="#2c3e50", width=300, height=300)
        self.img_input_canvas.pack(expand=True)
        
        self.start_button = Button(center_frame, text="START RECOGNITION", 
                                 command=self.start_recognition,
                                 bg="#2ecc71", fg="white", 
                                 font=("Helvetica", 12, "bold"),
                                 state=NORMAL)
        self.start_button.pack(pady=20, ipadx=10, ipady=5)
        
        right_frame = Frame(main_frame, bg="#34495e", bd=2, relief=GROOVE)
        right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        Label(right_frame, text="CLOSEST MATCH", font=("Helvetica", 14, "bold"), 
             fg="#ecf0f1", bg="#34495e").pack(pady=10)
        
        self.img_output_canvas = Canvas(right_frame, bg="#2c3e50", width=300, height=300)
        self.img_output_canvas.pack(expand=True)
        
        self.status_frame = Frame(self.window, bg="#34495e", height=30)
        self.status_frame.pack(fill=X, pady=(0, 10))
        self.status_label = Label(self.status_frame, text="Ready", fg="#ecf0f1", 
                                bg="#34495e", font=("Helvetica", 10))
        self.status_label.pack(side=LEFT, padx=10)
    
    def open_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            self.img_input_path.set(filename)
            try:
                img = Image.open(filename)
                img = img.resize((300, 300), Image.LANCZOS)
                
                self.current_input_image_tk = ImageTk.PhotoImage(img)
                self.img_input_canvas.delete("all")
                self.img_input_canvas.create_image(150, 150, image=self.current_input_image_tk, anchor=CENTER)
                
                self.status_label.config(text=f"Loaded: {Path(filename).name}", fg="#2ecc71")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.img_input_path.set("")
                self.status_label.config(text="Error loading image", fg="#e74c3c")
    
    def open_folder(self):
        foldername = filedialog.askdirectory()
        if foldername:
            self.dataset_folder_path.set(foldername)
            self.status_label.config(text=f"Dataset: {Path(foldername).name}", fg="#2ecc71")
    
    def run_recognition_process(self, image_path, folder_path):
        start_time = time.time()
        try:
            # FIX: Kirim path string, bukan gambar array!
            output_image_path = main.mainprog(image_path, folder_path)
            duration = round(time.time() - start_time, 3)
            self.window.after(0, self.update_gui_after_recognition, output_image_path, duration, None)
        except Exception as e:
            duration = round(time.time() - start_time, 3)
            self.window.after(0, self.update_gui_after_recognition, None, duration, str(e))
    
    def update_gui_after_recognition(self, output_path, duration, error_message):
        self.start_button.config(state=NORMAL)
        self.execution_time_label.config(text=f"{duration:.3f} seconds")
        
        if error_message:
            messagebox.showerror("Recognition Error", error_message)
            self.status_label.config(text=f"Error: {error_message}", fg="#e74c3c")
            return
        
        try:
            if not Path(output_path).is_file():
                raise FileNotFoundError(f"Output file not found: {output_path}")
            
            img_result = Image.open(output_path)
            img_result = img_result.resize((300, 300), Image.LANCZOS)
            
            self.current_output_image_tk = ImageTk.PhotoImage(img_result)
            self.img_output_canvas.delete("all")
            self.img_output_canvas.create_image(150, 150, image=self.current_output_image_tk, anchor=CENTER)
            
            self.status_label.config(text=f"Match found in: {Path(output_path).parent.name}", fg="#2ecc71")
        except Exception as e:
            messagebox.showerror("Display Error", f"Cannot show result: {str(e)}")
            self.status_label.config(text=f"Display error: {str(e)}", fg="#e74c3c")
    
    def start_recognition(self):
        input_img = self.img_input_path.get()
        dataset = self.dataset_folder_path.get()
        
        if not input_img or not Path(input_img).is_file():
            messagebox.showwarning("Input Error", "Please select a valid input image")
            return
        if not dataset or not Path(dataset).is_dir():
            messagebox.showwarning("Dataset Error", "Please select a valid dataset folder")
            return
        
        self.start_button.config(state=DISABLED)
        self.status_label.config(text="Processing... Please wait", fg="#f39c12")
        self.execution_time_label.config(text="Processing...")
        self.img_output_canvas.delete("all")
        
        thread = threading.Thread(
            target=self.run_recognition_process,
            args=(input_img, dataset),
            daemon=True
        )
        thread.start()

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
