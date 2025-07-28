import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def preprocess_img(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, debug=False):
    # Create sub-model: input → conv layer + prediction
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_output = predictions[:, 0]  # For binary classification

    # Compute gradients of class output w.r.t conv layer output
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global Average Pooling

    # Debug info (optional)
    if debug:
        print("Predicted value:", predictions.numpy()[0][0])
        print("Max gradient:", tf.reduce_max(grads).numpy())
        print("Sample pooled grads:", pooled_grads.numpy()[:5])

    # Weight the convolution outputs with the gradients
    conv_outputs = conv_outputs[0]  # shape: (H, W, Channels)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()

    return heatmap.numpy()

def apply_heatmap(img_path, heatmap, alpha=0.5, threshold=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize and threshold the heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_masked = np.where(heatmap > threshold, heatmap, 0)

    # Convert to 8-bit and apply colormap
    heatmap_masked = np.uint8(255 * heatmap_masked)
    heatmap_color = cv2.applyColorMap(heatmap_masked, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
