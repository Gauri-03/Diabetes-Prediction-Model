
# Diabetes Prediction Model

This project involves building a diabetes prediction model using Support Vector Machines (SVM). The model is trained on a labeled dataset to classify whether an individual is diabetic or non-diabetic based on specific health parameters.

## Technologies Used

- **Python**: The core programming language used to develop and train the model.
- **NumPy**: For numerical operations and handling array structures.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: Used for model training, evaluation, and data preprocessing.
  - **SVM (Support Vector Machine)**: The algorithm used for classification.
  - **StandardScaler**: For normalizing the feature data.
  - **Train-Test Split**: To split the dataset into training and testing sets.
  - **Accuracy Score**: For evaluating the performance of the model.

## Dataset

The model is trained on the **PIMA Diabetes Dataset**, which contains health-related data and labels indicating whether a patient has diabetes. The dataset includes features such as:

- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

Each record in the dataset is labeled as either:

- **0**: Non-Diabetic
- **1**: Diabetic

## Model Training

### Steps Involved:

1. **Data Collection**: The PIMA Diabetes Dataset is loaded and analyzed.
2. **Data Preprocessing**: 
   - Standardization of the features using `StandardScaler` to bring all features to the same scale.
   - Splitting the data into training and testing sets using `train_test_split`.
3. **Model Building**: 
   - An SVM model is trained on the training data.
4. **Model Evaluation**:
   - The trained model is evaluated on the test data using accuracy scores to assess its performance.

### Results

The model's performance is measured using the accuracy score, indicating how well it predicts diabetes in the test data.

## Instructions to Run the Project

1. Ensure you have Python installed on your machine.
2. Install the necessary dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Load the notebook (`.ipynb` file) and run all the cells to train the model and evaluate its performance.

## Conclusion

This project demonstrates a basic application of SVM in a healthcare setting to predict diabetes. The model serves as a foundation for further exploration and improvements, potentially incorporating more sophisticated techniques or larger datasets for better performance.
