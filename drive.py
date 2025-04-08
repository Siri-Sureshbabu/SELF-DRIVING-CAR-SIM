import argparse
import base64
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from datetime import datetime
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
import tensorflow as tf

eventlet.monkey_patch()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model
print("📂 Loading model...")
model = tf.keras.models.load_model("model.keras")
print("✅ Model loaded successfully!")

# Initialize server
sio = socketio.Server()
app = Flask(__name__)

# Speed parameters
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        print(f"📡 Received telemetry: {data}")  # Debugging line

        try:
            speed = float(data["speed"])
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image = np.asarray(image)
            image = img_preprocess(image)
            image = np.expand_dims(image, axis=0)

            # Predict steering angle
            steering_angle = float(model.predict(image, batch_size=1))
            print(f"🛞 Predicted Steering Angle: {steering_angle:.4f}")

            # Set fixed throttle for testing
            throttle = 0.3
            print(f"🚗 Sending Control: Steering = {steering_angle:.4f}, Throttle = {throttle:.4f}")

            send_control(steering_angle, throttle)

        except Exception as e:
            print(f"🚨 Error processing telemetry: {e}")

    else:
        print("🚨 No telemetry data received!")
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print(f"🔗 Connected to Simulator! Session ID: {sid}")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': str(steering_angle), 'throttle': str(throttle)},
        skip_sid=True
    )

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    print("🚀 Starting server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
