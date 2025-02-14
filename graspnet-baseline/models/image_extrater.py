from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the VGG16 model pre-trained on ImageNet, excluding the top classification layer
model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path, input_dimension):
    # Load the image with the specified input dimension
    img = image.load_img(img_path, target_size=input_dimension)
    # Convert the image to a numpy array
    img_data = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_data = np.expand_dims(img_data, axis=0)
    # Preprocess the image data
    img_data = preprocess_input(img_data)
    # Extract features
    features = model.predict(img_data)
    return features

# Example usage
img_path = 'path/to/your/image.jpg'
input_dimension = (224, 224)  # Example input dimension for VGG16
features = extract_features(img_path, input_dimension)
print(features)