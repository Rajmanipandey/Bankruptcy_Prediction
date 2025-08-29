# Bankruptcy Prediction

![Gradio-Based User Interface](https://github.com/user-attachments/assets/4bd3b766-7041-499b-87c8-fefcaa3dfd20)

This repository provides a machine learning-based bankruptcy prediction tool for Small and Medium Enterprises (SMEs). The project aims to predict bankruptcy status by analyzing financial data and providing interpretable results through a user-friendly interface developed in Gradio. The ensemble model outperforms traditional methods, achieving high accuracy and reliability in financial distress prediction. This README outlines the project goals, methodology, usage, and key findings.

## Overview

- **Objective**: Develop an interpretable, robust model for predicting bankruptcy using financial ratios, with an interface that offers a report and visual explanation for users.
- **Key Contributions**:
  - Ensemble model with Logistic Regression, Random Forest, and XGBoost for high prediction accuracy.
  - SHapley Additive exPlanations (SHAP) to provide insights into key financial ratios influencing predictions.
  - Automated narrative report generation using Natural Language Generation (NLG) for accessible insights.
  
![Bankruptcy Prediction Report](https://github.com/user-attachments/assets/6f0a4eb8-c020-482d-94b6-c0b1ea5908cb)

## Project Workflow

1. **Data Preparation**:
   - Data sourced from a private dataset of 427 companies with 66 financial features.
   - Preprocessing included handling missing values, outlier detection, normalization, and SMOTE for class balance.

2. **Model Development**:
   - Developed an ensemble model with Logistic Regression, Random Forest, and XGBoost, combined using a Voting Classifier.
   - Performance evaluated through accuracy, ROC AUC, and F1-scores.
   - The final model achieved an accuracy of 96.63% and an ROC AUC of 99.6%.

3. **User Interface and Interpretability**:
   - Built with Gradio for easy user interaction, enabling input of financial metrics and real-time prediction display.
   - Integrated SHAP for interpretability, highlighting top factors affecting bankruptcy predictions.

4. **Automated Report Generation**:
   - NLG used to generate comprehensive, readable reports outlining prediction results, influential features, and recommendations.

![Workflow Diagram](https://github.com/user-attachments/assets/159e6ccb-d02b-4dd5-a853-2e32072f27c8)

## Getting Started

### Prerequisites

- Install required Python libraries.
- Compatible with Python 3.12

### Running the Project

1. **Model Training**:
   - Run `Bankruptcy_Prediction.ipynb` to preprocess data and train models compatible with your system.
   
2. **Launching the Interface**:
   - Execute `GradioApp.py` to start the Gradio app locally. Click the generated link to access the interface in a browser.

## Results

- The Voting Classifier model achieved high accuracy and consistency across all prediction classes.
- SHAP analysis provides transparency, enhancing model trustworthiness by allowing users to see which factors most influence the model’s decisions.
- Generated reports summarize predictions and suggest actionable financial strategies for users.

## Conclusion

The bankruptcy prediction tool developed here offers a robust solution for assessing financial distress risks in SMEs. By combining high predictive performance with interpretability and an easy-to-use interface, this tool provides a practical resource for financial analysts, business owners, and stakeholders.
