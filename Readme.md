# 💼 Employee Salary Prediction App

This is a Streamlit web application that predicts whether an individual's income is **greater than $50K or not**, based on demographic features from the UCI Adult dataset. The model is trained using a **Random Forest Classifier** with a preprocessing pipeline, and deployed using **Streamlit**.

---

## 📦 Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application code |
| `model_lightgbm_v2.pkl` | Trained scikit-learn pipeline (preprocessing + classifier) |
| `model_features.pkl` | Feature names used by the model (for input alignment) |
| `salary_model_adult.ipynb` | Jupyter notebook used for training and saving the model |
| `adult 3.csv` | Cleaned UCI Adult dataset used for training |

---

## 🚀 How to Run the App Locally

### ✅ Step 1: Install Requirements

Make sure Python 3.8+ is installed.

You can install dependencies using `pip`:

```bash
pip install streamlit scikit-learn pandas joblib

Or with Conda:

conda install -c conda-forge streamlit scikit-learn pandas joblib


✅ Step 2: Run the App
1. Open Anaconda Prompt or any terminal.

2. Navigate to the folder containing your app.py:
cd "C:\Path\To\Your\Project\Folder"

3. Run the app:
streamlit run app.py

4. Open http://localhost:8501 in your browser if it doesn’t open automatically.

📊 Features Used in Prediction
age

work_class

education

marital-status

occupation

relationship

race

gender

native-country

hours-per-week

These features are automatically preprocessed using ColumnTransformer + OneHotEncoder.


🧠 Model Info
Algorithm: RandomForestClassifier (100 estimators)

Preprocessing: ColumnTransformer with one-hot encoding

Accuracy: ~85% on test set


🛠 Folder Structure
📁 your-project-folder
├── app.py
├── model_lightgbm_v2.pkl
├── model_features.pkl
├── salary_model_adult.ipynb
├── adult 3.csv
└── README.md

🌐 Deployment
You can deploy this app to the web using:

Streamlit Community Cloud

Hugging Face Spaces

Heroku or Docker (optional)

🙋‍♂️ Author
Bharath
IBM SkillBuild - EDUNET Internship Project
MSc Internship (2025)

📜 License
This project is for educational and demonstration purposes.