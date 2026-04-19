# Iris Flower Species Predictor
# Data Science Certification Project


# Project Structure
```
iris_project/
│
├── iris_project.py    ← Main Python script (EDA + Model Training)
├── app.py             ← Streamlit Web Application
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
│
├── iris_model.pkl     ← Saved trained model  (generated after running iris_project.py)
├── iris_scaler.pkl    ← Saved StandardScaler (generated after running iris_project.py)
│
├── eda_distributions.png   ← EDA plot (generated)
├── eda_correlation.png     ← EDA plot (generated)
├── eda_pairplot.png        ← EDA plot (generated)
├── eda_boxplots.png        ← EDA plot (generated)
└── confusion_matrix.png    ← Model evaluation plot (generated)
```

---
# How to Run the Project



### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python iris_project.py
```
This will:
- Load and explore the dataset
- Clean and preprocess the data
- Perform EDA and save visualization plots
- Train KNN and Logistic Regression models
- Tune hyperparameters with GridSearchCV
- Save the best model (`iris_model.pkl`) and scaler (`iris_scaler.pkl`)

### Step 3: Launch the Web App
```bash
python-m streamlit run app.py
```
Then open your browser and go to: **http://localhost:8501**

---

## 📊 Project Covered

|  | Description |
|------|-------------|
| 1 | Data Collection and Exploration |
| 2 | Data Cleaning and Transformation |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Feature Selection |
| 5 | Model Development (KNN + Logistic Regression) |
| 6 | Model Evaluation and Hyperparameter Tuning |
| 7 | Streamlit Web App Deployment on localhost |
| 8 | Documentation (comments in code + README) |
| 9 | Version Control with Git & GitHub |

---

#Machine Learning Details

- **Dataset**: Iris Dataset (150 samples, 4 features, 3 classes)
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Preprocessing**: StandardScaler (feature normalization)
- **Tuning**: GridSearchCV over k = [3, 5, 7, 9, 11]
- **Evaluation**: Accuracy, Classification Report, Confusion Matrix, 5-Fold Cross-Validation
- **Test Accuracy**: ~96–100%

---

## 📦 Technologies Used

- Python 3.8+
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 👤 Author

**[Umesh Shrestha ]**  
Data Science Certification Project  
[https://github.com/umesh625/iris-flower-predictor.git]
