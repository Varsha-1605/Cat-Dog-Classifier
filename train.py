import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 64
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# LightGBM parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_boost_round': 100,
    'early_stopping_rounds': 10
}

def load_data(data_dir):
    images = []
    labels = []
    
    # Load cats
    logger.info("Loading cat images...")
    cat_dir = os.path.join(data_dir, 'cats')
    for img_file in os.listdir(cat_dir)[:1000]:
        img_path = os.path.join(cat_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Extract additional features
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_features = np.concatenate([
                img.flatten() / 255.0,  # RGB features
                img_hsv.flatten() / 255.0,  # HSV features
            ])
            images.append(img_features)
            labels.append(0)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
    
    # Load dogs
    logger.info("Loading dog images...")
    dog_dir = os.path.join(data_dir, 'dogs')
    for img_file in os.listdir(dog_dir)[:1000]:
        img_path = os.path.join(dog_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Extract additional features
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_features = np.concatenate([
                img.flatten() / 255.0,  # RGB features
                img_hsv.flatten() / 255.0,  # HSV features
            ])
            images.append(img_features)
            labels.append(1)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def train_model():
    logger.info("Starting model training process...")
    
    # Load and prepare data
    X, y = load_data('dataset')
    logger.info(f"Loaded {len(X)} images with shape {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    # Train model
    logger.info("Training LightGBM model...")
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[lgb.log_evaluation(period=20)]
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
    
    # Feature importance analysis
    logger.info("\nTop 10 Most Important Features:")
    importance = model.feature_importance(importance_type='gain')
    for i in range(min(10, len(importance))):
        logger.info(f"Feature {i}: {importance[i]}")
    
    # Save model and scaler
    logger.info("Saving model and scaler...")
    model.save_model(str(MODEL_DIR/'model.txt'))
    with open(MODEL_DIR/'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("Training complete!")
    return model, scaler

if __name__ == "__main__":
    train_model()