from flask import Flask, Response
import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the TFLite model and allocate tensors (using the Edge TPU)
interpreter = tflite.Interpreter(model_path="mobilenet_v2_quant_edgetpu.tflite",
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the image and prepare the input data
def preprocess_image(image):
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return img_array[np.newaxis, ...]

# Connect to the RTSP feed
rtsp_url = 'rtsp://Web:B3achdog@192.168.0.55:8000/++stream?cameraNum=1&codec=h264'
cap = cv2.VideoCapture(rtsp_url)

# Function to capture frames and process them
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the captured frame
        input_data = preprocess_image(frame)

        # Perform the inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve the results
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process the results (e.g., find the highest probability class)
        predicted_label = np.argmax(output_data[0])
        print("Predicted Label:", predicted_label)

        # Add text to frame (optional)
        cv2.putText(frame, f'Label: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Return the video feed as a multipart response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
