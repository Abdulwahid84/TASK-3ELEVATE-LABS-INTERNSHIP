# TASK-3ELEVATE-LABS-INTERNSHIP
# Task 3: Linear Regression - Simple & Multiple

## 📌 Objective

This project demonstrates the implementation and understanding of **Simple and Multiple Linear Regression** using Python's popular data science libraries: **Scikit-learn**, **Pandas**, and **Matplotlib**.

---

## 🛠 Tools & Libraries Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- NumPy (optional, for numerical operations)

---

## 📁 Dataset

The dataset used for this project should contain numerical features suitable for regression analysis. You can use any relevant dataset like:
- Housing prices
- Advertising sales data
- Car mileage and engine stats, etc.

Make sure the dataset is properly cleaned and formatted (e.g., no missing values or categorical data unless encoded).

---

## 📌 Steps Followed

1. **Import & Preprocess the Dataset**
   - Load the dataset using Pandas.
   - Clean and preprocess data (handle missing values, normalize if needed).

2. **Split Data**
   - Divide the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.

3. **Fit Linear Regression Model**
   - Use `LinearRegression` from `sklearn.linear_model`.
   - Train the model on the training dataset.

4. **Model Evaluation**
   - Use the following metrics to evaluate model performance on the test set:
     - **MAE** (Mean Absolute Error)
     - **MSE** (Mean Squared Error)
     - **R² Score**

5. **Plotting**
   - Plot the regression line for simple linear regression.
   - Visualize residuals and prediction vs actual values.
   - Interpret the regression coefficients to understand the impact of features.

---

## 📊 Output & Results

- Regression line plotted over data points.
- Coefficients printed with explanations.
- Evaluation metrics summarized for model performance.
- (Optional) Comparison between simple and multiple regression models.

---

## 🧠 What You'll Learn

- How to prepare data for regression models.
- The differences between simple and multiple linear regression.
- How to evaluate and visualize model performance.
- How to interpret regression coefficients in real-world context.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linear-regression-task.git
   cd linear-regression-task
