# Cat vs Dog Image Classification and Real-Time Prediction Web Application

## Introduction
This project focuses on building an image classification web application capable of distinguishing between images of cats and dogs. The application leverages a Deep Learning model and is deployed via an interactive Streamlit dashboard, providing users with a beginner-friendly interface for real-time predictions. 

## Project Objectives
* Develop an image classification system to categorize images into predefined classes (Cat and Dog).
* Preprocess image data and prepare it for deep learning pipelines.
* Deploy the final model as a real-time web application using Streamlit.
* Provide an interactive UI that displays input images, predicted labels, and confidence scores.

## Project Guidelines / Requirements Implementation Status
The project was evaluated against the strict assignment guidelines. Below is an accurate reflection of what is present in the provided codebase versus the original requirements:

* **Implemented Successfully:**
  * Dataset loading via Kaggle API and initial exploration (folder checking, counting).
  * Displaying sample images from the dataset using Matplotlib.
  * Image resizing to 224×224 during the prediction phase.
  * Integration of a Fine-Tuned MobileNetV2 model.
  * Real-time prediction with confidence score calculation.
  * Unknown image detection (threshold < 70% confidence).
  * Complete, multi-page Streamlit web application (Home, Model Info, Prediction).
* **Not Implemented in Provided Code:**
  * Code for detecting and removing corrupted images is missing.
  * Code for Image Augmentation (Rotation, Horizontal Flip, Zoom, etc.) and explicit Data Splitting (Train/Val/Test) is not present.
  * **Model 1 (Custom CNN)** and **Model 2 (Base MobileNetV2)** were not built or trained in the provided code.
  * Model evaluation graphs (Training/Validation Accuracy and Loss), Confusion Matrix, Classification Report, and the 3-model comparison table are missing.

## Dataset Description
The project utilizes the Kaggle Cat VS Dog Dataset. It contains two primary classes organized into a specific folder structure:
* `PetImages/Cat/`
* `PetImages/Dog/`

## Data Collection
Data collection was automated in the Jupyter Notebook using the Kaggle API (`!kaggle datasets download -d karakaggle/kaggle-cat-vs-dog-dataset`). The dataset was successfully downloaded, unzipped, and stored in the local working directory.

## Exploratory Data Analysis (EDA)
The provided notebook code successfully performs initial EDA:
* **Folder Structure Verification:** Confirmed the existence of the core dataset paths.
* **Class Distribution:** Counted the total files, revealing a balanced dataset with 12,491 Cat images and 12,470 Dog images.
* **Visual Exploration:** Displayed a random sample of 6 Cat images using Matplotlib to visually inspect the data quality. 
*(Note: While the markdown in the notebook mentions detecting and removing corrupted images, the actual Python code to perform this action is not included).*

## Data Preprocessing
Based on the Streamlit application code, the following preprocessing step is explicitly applied to incoming images:
* **Resizing:** Images are converted to RGB and resized to 224×224 pixels to match the input shape expected by MobileNetV2.
* **Array Conversion and Batching:** Images are converted to `float32` NumPy arrays, and a batch dimension is added. 
* **Normalization:** The application code explicitly notes that division by 255 is omitted because the normalization layer is already baked into the saved `.keras` model.
*(Note: Data splitting and data augmentation steps required by the guidelines were not implemented in the provided code).*

## Model Building & Models Used
The assignment guidelines required the development of three distinct models (Custom CNN, Base MobileNetV2, and Fine-Tuned MobileNetV2). 

The provided code focuses exclusively on the final optimized model: **Model 3 - Fine-Tuned MobileNetV2**. The code for building the Custom CNN and the Base MobileNetV2 model is not present. The application directly loads a pre-saved transfer learning model (`fine_tuned_mobilenet_v2.keras`).

## Model Training
While the training script is missing from the provided code, the Streamlit application documents the hyperparameters used to train the final Fine-Tuned MobileNetV2 model:
* **Optimizer:** RMSprop (Note: The guideline requested Adam for the CNN, but RMSprop was used for this fine-tuned model).
* **Learning Rate:** 0.00001
* **Dropout:** 0.2
* **Trainable Layers:** The last 30 layers of the MobileNetV2 base were unfrozen for fine-tuning.

## Model Evaluation & Results
The provided code does not dynamically generate evaluation metrics, confusion matrices, or training graphs. However, the final test performance metrics are documented within the Streamlit app's "Model Information" page:
* **Accuracy:** 98.90%
* **Precision:** 98.62%
* **Recall:** 99.20%
* **F1-Score:** 98.91%

## Final Model Selection
The Fine-Tuned MobileNetV2 was selected for deployment due to its high reported accuracy and F1-Score, demonstrating the effectiveness of transfer learning and fine-tuning on a pre-trained ImageNet architecture.

## Prediction / Testing
The real-time prediction pipeline is fully implemented. When an image is uploaded:
1. The model predicts a float value between 0 and 1.
2. A threshold of 0.5 determines the class (>= 0.5 is Dog, < 0.5 is Cat).
3. A confidence score is calculated as a percentage.
4. **Unknown Image Detection:** If the model's confidence is below 70%, the app triggers a warning stating "Not a Cat or Dog", preventing false classifications on irrelevant images.

## Streamlit Web Application
A beginner-friendly, clean, and organized Streamlit dashboard was developed with three navigational pages:
1. **Home:** Introduces the project, its features, and provides a brief description.
2. **Model Information:** Displays the hyperparameters, architecture details, and test performance metrics of the chosen model.
3. **Prediction:** Features a file uploader for `.jpg`, `.jpeg`, and `.png` images. It displays the uploaded image alongside the final prediction, class label (with emojis), and the calculated confidence score.

## Project Structure
```text
Project_Folder/
├── app.py                              # Streamlit web application code
├── PROJECT_REPORT.md                   # This project documentation
├── model/
│   └── fine_tuned_mobilenet_v2.keras   # Saved deep learning model
└── PetImages/                          # Downloaded Kaggle dataset
    ├── Cat/
    └── Dog/
