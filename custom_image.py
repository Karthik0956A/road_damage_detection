import os
import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

def main():
    # 1. Hide the main blank tkinter window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Bring the file dialog to the front

    # 2. Open the file selection dialog
    print("Opening file explorer... Please select an image.")
    file_path = filedialog.askopenfilename(
        title="Select an Image for YOLO Prediction",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    # 3. Check if the user selected a file or cancelled
    if not file_path:
        print("Selection cancelled. Exiting.")
        return

    print(f"Selected image: {os.path.basename(file_path)}")
    print("Running prediction...")

    image = cv2.imread(file_path)
    if image is None:
        print("Could not read the selected image. Exiting.")
        return

    # Match the letterboxed 640x640 input used by YOLO during training.
    image = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False)(image=image)

    # 4. Load your model and predict (without saving to disk)
    model = YOLO("./best.pt")  # Load the trained model
    results = model(image, imgsz=640, save=False)

    # 5. Extract the image with bounding boxes drawn on it
    annotated_img = results[0].plot()

    # 6. Display the image in a window until closed
    cv2.imshow("YOLO Prediction Result", annotated_img)
    print("Displaying results. Press any key or click 'X' to close.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
