import streamlit as st
import numpy as np
import cv2
import os
import pickle
import lightgbm as lgb
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.65  # Adjusted for LightGBM
MODEL_PATH = "model/model.txt"
SCALER_PATH = "model/scaler.pkl"
CLASS_LABELS = ["Cat", "Dog"]
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Custom exception for model loading

class ModelLoadingError(Exception):
    pass

# Cache for storing prediction history
class PredictionCache:
    def __init__(self, cache_file=CACHE_DIR/"prediction_history.json"):
        self.cache_file = cache_file
        self.predictions = self._load_cache()
    
    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_prediction(self, filename, prediction, confidence):
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'prediction': prediction,
            'confidence': float(confidence)
        }
        self.predictions.append(prediction_data)
        self._save_cache()
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.predictions[-100:], f)  # Keep last 100 predictions

    def get_statistics(self):
        if not self.predictions:
            return None
        total = len(self.predictions)
        cats = sum(1 for p in self.predictions if p['prediction'] == 'Cat')
        dogs = sum(1 for p in self.predictions if p['prediction'] == 'Dog')
        avg_confidence = np.mean([p['confidence'] for p in self.predictions])
        return {
            'total': total,
            'cats': cats,
            'dogs': dogs,
            'avg_confidence': avg_confidence
        }

# Updated ML Model wrapper class for LightGBM
class PetClassifier:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model, self.scaler = self._load_model(model_path, scaler_path)
        
    def _load_model(self, model_path, scaler_path):
        try:
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                raise ModelLoadingError("Model or scaler file not found")
            
            model = lgb.Booster(model_file=model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            return model, scaler
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelLoadingError(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image):
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image.convert('RGB'))
            # Resize image
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # Convert to HSV for additional features
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            # Combine features
            img_features = np.concatenate([
                img_array.flatten() / 255.0,
                img_hsv.flatten() / 255.0
            ])
            # Apply scaler
            img_features = self.scaler.transform(img_features.reshape(1, -1))
            return img_features
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, image):
        try:
            processed_image = self.preprocess_image(image)
            probability = self.model.predict(processed_image)[0]
            predicted_class = CLASS_LABELS[int(probability > 0.5)]
            confidence = float(probability if probability > 0.5 else 1 - probability)
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
        

# Streamlit UI components
class UI:
    def __init__(self):
        self.cache = PredictionCache()
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(
            page_title="ML Pet Classifier",
            page_icon="🐾",
            layout="wide"
        )
        st.title("🐾 Machine Learning Pet Classifier")
        st.markdown("""
        ### Welcome to the ML Pet Classifier!
        This application uses traditional machine learning to classify cats and dogs.
        """)
    
    def display_sidebar_stats(self):
        st.sidebar.title("📊 Statistics")
        stats = self.cache.get_statistics()
        if stats:
            st.sidebar.metric("Total Predictions", stats['total'])
            st.sidebar.metric("Cats Identified", stats['cats'])
            st.sidebar.metric("Dogs Identified", stats['dogs'])
            st.sidebar.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")
    
    def display_prediction(self, prediction, confidence):
        col1, col2 = st.columns(2)
        with col1:
            emoji = "🐱" if prediction == "Cat" else "🐶"
            st.success(f"Prediction: {emoji} {prediction}")
        with col2:
            self.display_confidence_gauge(confidence)
    
    def display_confidence_gauge(self, confidence):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': CONFIDENCE_THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)


def main():
    try:
        # Initialize UI
        ui = UI()
        ui.display_sidebar_stats()
        
        # Add model info to sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Information")
        st.sidebar.markdown("""
        - **Model Type:** LightGBM
        - **Features:** RGB + HSV
        - **Image Size:** 64x64
        """)
        
        # Initialize model
        classifier = PetClassifier()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a cat or dog"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("🤔 Analyzing image..."):
                prediction, confidence = classifier.predict(image)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    ui.display_prediction(prediction, confidence)
                    ui.cache.add_prediction(
                        uploaded_file.name,
                        prediction,
                        confidence
                    )
                else:
                    st.warning("⚠️ Low confidence prediction. Please try with a clearer image.")
            
            # Display additional information
            with st.expander("ℹ️ More Information"):
                st.markdown(f"""
                * **File name:** {uploaded_file.name}
                * **Confidence threshold:** {CONFIDENCE_THRESHOLD * 100}%
                * **Image size:** {image.size}
                * **Processing size:** {IMG_SIZE}x{IMG_SIZE}
                * **Model type:** LightGBM
                * **Feature extraction:** RGB + HSV color spaces
                """)
    
    except ModelLoadingError as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model and scaler files are available and valid.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()