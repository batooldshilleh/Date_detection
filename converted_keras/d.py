import cv2
import numpy as np
from keras.preprocessing import image
import depthai
import tensorflow as tf

# Load the pre-trained model from the correct path
model = tf.keras.models.load_model('keras_model.h5')

# Initialize the DepthAI pipeline
pipeline = depthai.Pipeline()

# Create a node for the camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

# Start the pipeline
device = depthai.Device(pipeline)
class_labels = ['Other', 'Date']
# Define your class labels and corresponding dimensions (width, length)
class_info = {
    'Other': (0, 0),  # Define appropriate dimensions for 'Other' class
    'Date': (2, 5),   # Define dimensions for 'Date' class (example values)
    # Add more classes and dimensions as needed
}

while True:
    # Get the camera frames
    in_video = device.getOutputQueue(name="video", maxSize=1, blocking=True)
    frame = in_video.get().getCvFrame()

    # Pre-process the frame for classification
    img = cv2.resize(frame, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    
    # Use the model to classify the frame
    prediction = model.predict(img_tensor)
    
    # Get the class index with the highest probability
    class_index = np.argmax(prediction[0])
    
    # Get the corresponding class label
    class_label = class_labels[class_index]
    
    # Display the frame with the class label
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 3)
    
    # Calculate and display object size
    if class_label == 'Date':
        object_width, object_length = class_info[class_label]
        focal_length = 1000  # Focal length in pixels
        
        object_pixel_width = frame.shape[1] * object_width / img.shape[1]
        object_pixel_length = frame.shape[0] * object_length / img.shape[0]
        
        size = object_width * object_length
        distance = (size * focal_length) / (object_pixel_width + object_pixel_length)  # Calculate distance to object
        
        size_text = f"Size: {size:.2f} cm^2, Distance: {distance:.2f} cm"
        cv2.putText(frame, size_text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 3)
    
    cv2.imshow('Camera', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
device.close()
cv2.destroyAllWindows()


