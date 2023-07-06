import cv2
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
input_shape = (360, 102, 3)
batch_size = 32
epochs = 20
train_data_dir = 'vitiligo/train'
test_data_dir = 'vitiligo/test'
train_data_generator = ImageDataGenerator(rescale=1./255)
train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)
test_data_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)
# Define the CNN model architecture
model = tf.keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the output from the convolutional layers
    layers.Flatten(),
    
    # Dense (fully connected) layers
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (vitiligo or not)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
# Save the trained model
model.save('vitiligo_model.h5')
# Load the pre-trained machine learning model
model = tf.keras.models.load_model('vitiligo_model.h5') 
# Function to process the camera feed frame and identify vitiligo
def detect_vitiligo(frame):
    # Preprocess the frame (resize, normalize, etc.)
    processed_frame = preprocess_frame(frame)
    
    # Apply the trained model for vitiligo detection
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    
    # Process the prediction to identify vitiligo-affected regions
    vitiligo_regions = process_prediction(prediction)
    
    # Draw bounding boxes or masks on the frame to highlight vitiligo regions
    frame_with_vitiligo = draw_regions(frame, vitiligo_regions)
    
    return frame_with_vitiligo
def preprocess_frame(frame):
    # Check if the frame is empty
    if frame is None:
        return np.empty((360, 102, 3), dtype=np.float32)
    
    try:
        # Resize the frame to the desired input size
        resized_frame = cv2.resize(frame, (102, 360))
    
        # Convert the frame to float32 data type
        processed_frame = resized_frame.astype(np.float32)
    
        # Normalize the pixel values to the range of [0, 1]
        processed_frame /= 255.0
    
        return processed_frame
    except Exception as e:
        print("Error during frame preprocessing:", str(e))
        return np.empty((360, 102, 3), dtype=np.float32)
def process_prediction(prediction, confidence_threshold=0.5):
    vitiligo_regions = []
    
    for i, pred in enumerate(prediction[0]):
        if pred >= confidence_threshold:
            # Retrieve the bounding box coordinates based on your implementation
            x, y, width, height = get_bounding_box_coordinates(pred)
            
            # Store the vitiligo-affected region information
            region = {"label": "vitiligo", "confidence": pred, "box": (x, y, width, height)}
            vitiligo_regions.append(region)
    
    return vitiligo_regions
def get_bounding_box_coordinates(pred):
    # Placeholder implementation, replace with your own logic
    x = 0  # Placeholder value for the x-coordinate
    y = 0  # Placeholder value for the y-coordinate
    width = 0  # Placeholder value for the width of the bounding box
    height = 0  # Placeholder value for the height of the bounding box
    
    return x, y, width, height
# Initialize the camera feed
camera = cv2.VideoCapture(0)  # Use appropriate camera index or video file path if not using webcam
# Create GUI window
window = tk.Tk()
window.title("Vitiligo Detection")

# Create label to display image
image_label = tk.Label(window)
image_label.pack()

# Create label to display result
result_label = tk.Label(window, font=("Helvetica", 16))
result_label.pack()
def update_frame():
    # Capture frame from the camera
    ret, frame = camera.read()
    
    # Detect vitiligo
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    vitiligo_presence = process_prediction(prediction)
    
    # Convert the frame to PIL Image format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize((400, 300))
    
    # Display the image in the GUI
    image = ImageTk.PhotoImage(image=pil_image)
    image_label.config(image=image)
    image_label.image = image
    
    # Display the result in the GUI
    result = "Vitiligo" if vitiligo_presence else "No Vitiligo"
    result_label.config(text=result)
    
    # Schedule the next frame update
    window.after(100, update_frame)
# Start updating frames
update_frame()

# Run the GUI main loop
window.mainloop()

# Release the camera
camera.release()
cv2.destroyAllWindows()
