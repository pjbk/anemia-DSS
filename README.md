# 🩸 Real-Time Anemia Risk Predictor

An Ensemble learning-powered web application designed to diagnose anemia along with the likelihood from clinical hematological and demographic parameters. A real-time interface developted with Streamlit that can predict anemia and enabling data-driven clinical decisions at point-of-care. This tool empowers medical practitioners and health professionals with a decision support system (DSS) that not only predicts dengue status but also explains the reasoning using SHAP (SHapley Additive exPlanations) visualizations.

## 🌐 Live Web App Preview
**Live Predictive System:** [https://anemia-dss.streamlit.app/](https://anemia-dss.streamlit.app/)

![App Header](https://github.com/pjbk/anemia-DSS/blob/main/anemia-predictor-interface.jpg)

## Key Features

- **Accurate Disease Diagnosis**: Diagnoses Anemia with high precision (Ensemble model's predictio accuracy is 99.67%).
- **Model Explainability**: Utilizes SHAP-XAI to facilitate better understanding AI models. Top 5 Influential Features highlighted for clinical insights.
- **Responsive UI Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities to showcase the model's certainty.

## Dataset

The model is trained using clinical records, collected from Aalok Healthcare Ltd., Dhaka, Bangladesh.   
**Ref 1.** Mojumdar, M.U., et al.: Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification. Mendeley Data, V1(2024). https://doi.org/10.17632/y7v7ff3wpj.1  
**Ref 2.** Mojumdar, M.U., et al.: AnaDetect: An extensive dataset for advancing anemia detection, di-agnostic methods, and predictive analytics in healthcare. Data in Brief 58, 111195 (2025). https://doi.org/10.1016/j.dib.2024.111195  

The dataset can be accessed from the following Mendeley Data link:
[](https://data.mendeley.com/datasets/y7v7ff3wpj/1)

## Model Architecture
```python
# model architecture  
base_learners = [
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='log2', min_samples_split=3, random_state=42)),
    ('lr', make_pipeline(StandardScaler(), LogisticRegression(C= 10, solver= ‘liblinear’, random_state=42))),
    ('svm', make_pipeline(StandardScaler(), SVC(C=10, kernel= 'linear', probability=True, random_state=42))),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(algorithm= 'auto', n_neighbors= 4)))
]

meta_learner = make_pipeline(StandardScaler(), SVC(C=10, kernel= 'linear', probability=True, random_state=42))

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=True,
    n_jobs=-1
)
```  
## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/dengue-diagnosis-app.git
   cd dengue-diagnosis-app
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**

   ```bash
   streamlit run app.py
   ```
