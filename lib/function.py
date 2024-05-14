from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def detect_non_vegetative_regions(image):
  """
  Detects non-green areas in an image and provides a conceptual approach for visualization.

  Args:
      image (ndarray): The input image.

  This function cannot directly modify the image due to constraints. It performs
  preprocessing, color detection
  """

  # Step 2: Preprocessing
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Step 3: Color Detection (Adjusted thresholds for green in HSV)
  lower_green = np.array([25, 40, 40])
  upper_green = np.array([100, 255, 255])
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

  # Step 4: Invert the mask to get the non-green areas
  non_green_mask = cv2.bitwise_not(green_mask)

  # Step 5: Overlay non-green areas on original image (for visualization)
  masked_image = cv2.bitwise_and(image, image, mask=non_green_mask)

  # Step 6: Invert the masked image to replace black with white (for visualization)
  masked_image_inverted = cv2.bitwise_not(masked_image)
  
  return {
      "image": image, 
      "masked_image": masked_image,
      "green_mask": green_mask,
      "Result": masked_image_inverted, 
  }

def detect_disease_zones(image):
    # Convert the image to HSV color space
    origin = image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a color range for brown spots
    lower_brown = np.array([10, 100, 50])  # Adjust these values based on the color of brown spots in your images
    upper_brown = np.array([30, 255, 255])  # Adjust these values based on the color of brown spots in your images

    # Create a mask for brown spots
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Apply connected components labeling to identify individual disease zones
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    # Identify disease zones based on size and location
    disease_zones = []
    for i, stat in enumerate(stats):
        if stat[4] > 100:  # Adjust this value based on the minimum size of disease zones you want to detect
            disease_zones.append((stat[0], stat[1], stat[2], stat[3]))

    # Draw rectangles around the disease zones on the original image
    for zone in disease_zones:
        x, y, w, h = zone
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangles

    return {
        "Origin": origin,
        "hsv": hsv,
        "mask": mask,
        "Result": image,
    }

def extract_hog_features_resized(image, target_size=(128, 128)):
    # Redimensionner l'image à la taille cible
    resized_image = cv2.resize(image, target_size)
    
    # Convertir l'image redimensionnée en niveaux de gris si elle a plus de deux dimensions
    if len(resized_image.shape) > 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Extraire les caractéristiques HOG de l'image redimensionnée
    features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False,
                transform_sqrt=True, block_norm='L2-Hys')
    return features

svm_model = joblib.load('models/svm_model.pkl')
pca = joblib.load('models/pca_model.pkl')
scaler = joblib.load('models/scaler_model.pkl')
def prediction(image):
    target_size = (64, 64)
    image_resized = cv2.resize(image, target_size)

    hog_features = extract_hog_features_resized(np.array(image_resized))
    hog_features_scaled = scaler.transform(pca.transform([hog_features]))
    prediction = svm_model.predict(hog_features_scaled)
    return "La feuille n'est pas malade." if prediction == 0 else "La feuille est malade."


