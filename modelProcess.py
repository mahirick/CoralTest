import tensorflow as tf
import numpy as np  # Import NumPy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Create a TensorFlow model (e.g., MobileNetV2).
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def representative_data_gen():
    for _ in range(100):
        # Generate random data as input and cast it to FLOAT32
        data = np.random.rand(1, 224, 224, 3).astype(np.float32) * 255
        yield [preprocess_input(data)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

tflite_quant_model = converter.convert()

# Save the quantized model to a file.
with open('mobilenet_v2_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Quantized TensorFlow Lite model is successfully created and saved!")

# Compile the model for the Edge TPU
# You need to use the Edge TPU Compiler which is a separate tool.
# Run this command in your terminal:
# edgetpu_compiler mobilenet_v2_quant.tflite
