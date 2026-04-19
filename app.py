# =============================================================================
# 🌸 Iris Flower Prediction - Streamlit Web Application
# =============================================================================
# Run this app with:  streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# ─── Load Model and Scaler ────────────────────────────────────────────────────
@st.cache_resource   # Cache so they load only once
def load_model():
    model  = joblib.load('iris_model.pkl')
    scaler = joblib.load('iris_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─── App Title & Description ──────────────────────────────────────────────────
st.title("🌸 Iris Flower Species Predictor")
st.markdown("""
Welcome! This app uses a **K-Nearest Neighbors** machine learning model to predict 
the species of an Iris flower based on its measurements.

 **How to use:**
1. Adjust the sliders on the left to enter flower measurements
2. Click **Predict Species** to see the result
3. Explore the **Dataset Info** tab to learn more!
""")

# ─── Sidebar: User Input ──────────────────────────────────────────────────────
st.sidebar.header("🔢 Enter Flower Measurements (in cm)")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width  (cm)", 2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.sidebar.slider("Petal Width  (cm)", 0.1, 2.5, 1.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("🌿 Model: K-Nearest Neighbors (KNN)\n\n📊 Dataset: Iris (150 samples, 3 classes)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Predict", " Dataset Info", "ℹ About"])

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Your Input Measurements")

    # Show input as a neat table
    input_data = {
        "Measurement"  : ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)"   : [sepal_length, sepal_width, petal_length, petal_width]
    }
    st.table(pd.DataFrame(input_data))

    # Predict button
    if st.button("🔮 Predict Species", type="primary", use_container_width=True):
        if not model_loaded:
            st.error(" Model files not found! Please run `iris_project.py` first to train and save the model.")
        else:
            # Prepare input and scale it
            user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            user_scaled = scaler.transform(user_input)

            # Make prediction
            prediction = model.predict(user_scaled)[0]
            probabilities = model.predict_proba(user_scaled)[0]

            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            emoji_map   = {0: "🌺", 1: "🌷", 2: "🌹"}
            predicted_species = species_map[prediction]

            # Display result
            st.success(f"### {emoji_map[prediction]} Predicted Species: **{predicted_species}**")

            # Show confidence bar chart
            st.subheader("Prediction Confidence")
            prob_df = pd.DataFrame({
                'Species'     : ["Setosa", "Versicolor", "Virginica"],
                'Confidence %': [round(p * 100, 2) for p in probabilities]
            })

            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ['#FF9999' if s != predicted_species else '#4CAF50'
                      for s in prob_df['Species']]
            bars = ax.barh(prob_df['Species'], prob_df['Confidence %'], color=colors)
            ax.set_xlabel('Confidence (%)')
            ax.set_xlim(0, 100)
            for bar, val in zip(bars, prob_df['Confidence %']):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontweight='bold')
            ax.set_title('Model Confidence per Species', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

            # Description of predicted species
            descriptions = {
                "Setosa"    : "🌺 **Iris Setosa** – Small flowers with short petals. Easily distinguishable from other species. Found in Arctic and subarctic regions.",
                "Versicolor": "🌷 **Iris Versicolor** – Medium-sized flowers. Also known as the Blue Flag Iris. Common in eastern North America.",
                "Virginica" : "🌹 **Iris Virginica** – Largest of the three species. Known as the Virginia Iris. Found in eastern North America."
            }
            st.info(descriptions[predicted_species])

# ── TAB 2: Dataset Info ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Iris Dataset Overview")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]

    st.write(f"**Total Samples:** {len(df)}  |  **Features:** 4  |  **Classes:** 3")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Feature Distributions by Species")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flatten(), iris.feature_names):
        for species in df['species'].unique():
            subset = df[df['species'] == species][col]
            ax.hist(subset, alpha=0.6, label=species, bins=12)
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('cm')
        ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    corr = df[iris.feature_names].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

# ── TAB 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader(" About This Project")
    st.markdown("""
    ###  Project Summary
    This is a **Data Science Certification Project** that demonstrates:
    - ✅ Data loading and exploration with **Pandas**
    - ✅ Data cleaning and preprocessing
    - ✅ Exploratory Data Analysis with **Matplotlib** & **Seaborn**
    - ✅ Machine learning with **Scikit-learn** (KNN & Logistic Regression)
    - ✅ Model evaluation (accuracy, confusion matrix, cross-validation)
    - ✅ Hyperparameter tuning with **GridSearchCV**
    - ✅ Model deployment with **Streamlit**

    ###  Model Used
    - **Algorithm**: K-Nearest Neighbors (KNN)
    - **Best K value**: Found via GridSearchCV
    - **Test Accuracy**: ~96-100%
    - **Preprocessing**: StandardScaler (feature normalization)

    ###  Dataset
    - **Name**: Iris Dataset
    - **Source**: UCI Machine Learning Repository (built into sklearn)
    - **Samples**: 150 (50 per class)
    - **Features**: Sepal length, Sepal width, Petal length, Petal width
    - **Target**: Species (Setosa, Versicolor, Virginica)

    ###  Technologies Used
    | Tool | Purpose |
    |------|---------|
    | Python | Programming language |
    | Pandas | Data manipulation |
    | NumPy | Numerical operations |
    | Matplotlib / Seaborn | Visualization |
    | Scikit-learn | Machine learning |
    | Joblib | Model saving |
    | Streamlit | Web application |
    | Git / GitHub | Version control |
    """)
