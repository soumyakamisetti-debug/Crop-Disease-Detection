🌱 AI-Based Crop Disease Detection System

This project is an AI-powered crop disease detection system developed as part of the Microsoft Elevate Virtual Internship.
The system uses Deep Learning (CNN) to detect diseases in potato plant leaves and helps farmers take early preventive action.

🎯 Problem Statement

Crop diseases significantly reduce agricultural productivity and farmers’ income.
Manual disease detection is time-consuming, costly, and error-prone.
This project aims to automatically detect potato leaf diseases using AI, ensuring faster and more accurate diagnosis.

💡 Proposed Solution

An AI-based system that:

Takes an image of a potato leaf as input

Uses a Convolutional Neural Network (CNN) to classify the leaf

Predicts whether the leaf is:

Healthy

Early Blight

Late Blight

Displays the prediction through a user-friendly Streamlit web interface

📊 Dataset

The dataset used is the PlantVillage Dataset, containing labeled potato leaf images:

Potato___Healthy

Potato___Early_Blight

Potato___Late_Blight

🔗 Dataset Source:
https://www.kaggle.com/datasets/emmarex/plantdisease

⚠️ Note:
The dataset and trained model files are excluded from this repository due to GitHub file size limitations.

🧠 Algorithm Used

Convolutional Neural Network (CNN)

Image preprocessing and normalization

Trained using labeled leaf images

Optimized for multi-class image classification

⚙️ System Requirements

Python 3.10

TensorFlow

Keras

OpenCV

Streamlit

NumPy, Pandas, Matplotlib

All required libraries are listed in requirements.txt.

🚀 How to Run the Project

1️⃣ Clone the Repository
git clone https://github.com/soumyakamisetti-debug/Crop-Disease-Detection.git
cd Crop-Disease-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train_model.py


This will train the CNN model and save the trained model locally.

4️⃣ Run the Web Application
streamlit run app.py


Upload a potato leaf image to get the disease prediction.

📈 Results

The model successfully classifies potato leaf images into healthy and diseased categories.

Provides fast and accurate predictions.

Demonstrates the effectiveness of deep learning in precision agriculture.

🔮 Future Scope

Extend detection to more crops and diseases

Improve accuracy using advanced deep learning architectures

Deploy the model on mobile or edge devices

Integrate real-time farm monitoring systems

📚 References

Mishra, U. et al., Deep learning-based disease detection in potato and mango leaves, Scientific Reports.

Reddy, J. K., Plant Disease Detection Using Deep Learning, IJRASET.

Multilevel Deep Learning Model for Potato Leaf Disease Recognition, MDPI Electronics.

Systematic Review of Deep Learning for Plant Diseases, Artificial Intelligence Review.

## 👨‍💻 Developed By

**Soumya Kamisetti**  
Microsoft Elevate Virtual Internship  
AI-based Social Impact Project
