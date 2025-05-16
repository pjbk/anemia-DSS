# 🩸 Anemia Risk Predictor

An Ensemble learning-powered web application designed to diagnose anemia from clinical hematological and demographic parameters. A real-time interface that can predict anemia and provides SHAP-based explanations, enabling data-driven clinical decisions at point-of-care.

**Live Predictive System:** [https://anemia-dss.streamlit.app/](https://anemia-dss.streamlit.app/)

![App Header](https://github.com/pjbk/anemia-DSS/blob/main/anemia-predictor-interface.jpg)

## Key Features

- **Accurate Disease Detection**: Diagnoses Anemia with high precision (Ensemble model's accuracy 99.67%).
- **Model Explainability**: Utilizes SHAP-XAI to ensure model transparency, build trust, and facilitate better understanding and refinement of AI models, particularly in clinical applications.
- **Responsive UI Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities to showcase the model's certainty.

## Dataset

The model is trained using clinical records, collected from Aalok Healthcare Ltd., Dhaka, Bangladesh. 
**Ref 1.** Mojumdar, M.U., et al.: Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification. Mendeley Data, V1(2024). https://doi.org/10.17632/y7v7ff3wpj.1
**Ref 2.** Mojumdar, M.U., et al.: AnaDetect: An extensive dataset for advancing anemia detection, di-agnostic methods, and predictive analytics in healthcare. Data in Brief 58, 111195 (2025). https://doi.org/10.1016/j.dib.2024.111195

The dataset can be accessed from the following Mendeley Data link:
[Pediatric Anemia Dataset](https://data.mendeley.com/datasets/y7v7ff3wpj/1)
