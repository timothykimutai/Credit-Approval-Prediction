# Machine Learning Project: Predictive Analysis

## Overview
This project demonstrates the implementation of a machine learning pipeline, starting from data preprocessing to building predictive models, with integrated CI/CD using GitHub Actions and visualizations using Power BI. The project uses a dataset to predict outcomes based on historical data, ensuring a comprehensive approach to machine learning.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Workflow](#workflow)
  - [1. Data Inspection and Preprocessing](#1-data-inspection-and-preprocessing)
  - [2. Machine Learning Pipeline](#2-machine-learning-pipeline)
  - [3. CI/CD with GitHub Actions](#3-cicd-with-github-actions)
  - [4. Visualization with Power BI](#4-visualization-with-power-bi)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Project Description
This project predicts the likelihood of specific outcomes using supervised machine learning algorithms. The pipeline includes data cleaning, feature engineering, and model evaluation. GitHub Actions automate testing and deployment, while Power BI provides actionable insights through interactive dashboards.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Database**: CSV file storage for dataset
- **CI/CD**: GitHub Actions
- **Visualization Tool**: Power BI

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/timothykimutai/Credit-Approval-Prediction.git
   cd project-name
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`clean_dataset.csv`) in the root directory.

4. Run the preprocessing script:
   ```bash
   python Model.py
   ```

5. Train the machine learning model:
   ```bash
   python train_model.py
   ```

6. Export the predictions for Power BI:
   ```bash
   python export_prediction.py
   ```

## Workflow

### 1. Data Inspection and Preprocessing
- **Steps**:
  1. Load the dataset.
  2. Handle missing values and outliers.
  3. Perform feature engineering (e.g., one-hot encoding, normalization).
  4. Split the data into training and testing sets.

- **Code**:
  Refer to `venv/Model.py` for detailed preprocessing steps.

### 2. Machine Learning Pipeline
- **Steps**:
  1. Train a Random Forest classifier.
  2. Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
  3. Save the trained model as a pickle file for deployment.

- **Code**:
  Refer to `venv/Model.py` for model building and evaluation.

### 3. CI/CD with GitHub Actions
- **Steps**:
  1. Set up a `.github/workflows/ml_pipeline.yml` file for automation.
  2. Configure tasks to run tests and linting during every commit.
  3. Automatically deploy the updated model or scripts.

- **Code**:
  Check the `ml_pipeline.yml` file for CI/CD configurations.

### 4. Visualization with Power BI
- **Steps**:
  1. Export model predictions to a CSV file.
  2. Load the CSV into Power BI.
  3. Create interactive dashboards for insights such as feature importance, accuracy distribution, and trends.

- **Example Dashboards**:
  - **Bar Chart**: Accuracy distribution.
  - **Scatterplot**: Relationship between features and predictions.
  - **Line Chart**: Prediction trends over time.

## Results
- **Model Accuracy**: Achieved 85% accuracy on the test dataset.
- **Insights**:
  - Feature importance showed `Income` and `Debt` as key predictors.
  - Prediction accuracy varied significantly across customer demographics.

## Future Enhancements
- Experiment with other algorithms like Gradient Boosting or Neural Networks.
- Incorporate additional features into the dataset.
- Deploy the model using a web interface or API for real-time predictions.
- Enhance visualizations in Power BI for deeper insights.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.