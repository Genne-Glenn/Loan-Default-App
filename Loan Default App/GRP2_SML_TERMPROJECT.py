# Removes Warnings and import libraries needed for the project
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import streamlit as st
from imblearn.over_sampling import SMOTE

# Global variables were to aid in easy usage throughout the work
# various variables were created for null values, shape of data, categorical columns in the data and data types present
# A final fully encoded dataset was also created to convert all strings to numerical data for easy analysis
df1 = pd.read_csv("/Users/genneglenn/PycharmProjects/PythonProject/SML Term project/GRP 2 SML TERMPROJECT/Loan_default.csv")
df = df1.drop(columns=['LoanID'])  # since the ID column doesnt have any reasonable data for analysis, it was dropped
df_shape = df1.shape  # this shows the total number of rows and columns
df_nullvalues = df1.isnull().sum()  # checking for total number of missing values
df_datatypes = df1.dtypes
df_describe = df.describe()
categorical_cols = [col for col in df.select_dtypes(include='object')]
df_encoded1 = df.replace({'Yes': 1, 'No': 0})
df_encoded = pd.get_dummies(df_encoded1, drop_first=True, dtype=int)

# Splitting the independent and dependent variables to be used in the modelling and prediction
X = df_encoded.drop('Default', axis=1)
y = df_encoded['Default']

# Standardization of the dataset using the standard scaler
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[X.select_dtypes(include='number').columns] = scaler.fit_transform(X.select_dtypes(include='number'))

# Splitting data for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# variables were created to show the shape of the split data
X_train_shape = X_train.shape
X_test_shape = X_test.shape
y_train_shape = y_train.shape
y_test_shape = y_test.shape

# SOME GLOBAL VARIABLES CREATED FOR MODELS
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

rf_model1 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf_model1.fit(X_resampled, y_resampled)
y_pred_rf1 = rf_model1.predict(X_test)
rf_report1 = classification_report(y_test, y_pred_rf1, output_dict=True)
rf_conf_matrix1 = confusion_matrix(y_test, y_pred_rf1)

# Evaluation metrics for Logistic Regression
logreg_model1 = LogisticRegression(max_iter=1000, random_state=42)
logreg_model1.fit(X_resampled, y_resampled)
y_pred_lr1 = logreg_model1.predict(X_test)
lr_report1 = classification_report(y_test, y_pred_lr1, output_dict=True)
lr_conf_matrix1 = confusion_matrix(y_test, y_pred_lr1)

# creating a sidebar for easy navigation
st.sidebar.header("PROJECT GUIDELINE")

# creating sidebar options and using conditional statements(if,elifs)
display = st.sidebar.selectbox("Choose what you want to view:",
                               ("NAMES OF PROJECT GROUP MEMBERS",
                                "BACKGROUND & OVERVIEW",
                                "DATA IMPORT & OVERVIEW",
                                "DATA PREPROCESSING",
                                "MODEL TRAINING",
                                "MODEL EVALUATION",
                                "PREDICTION FUNCTION",
                                "INTERPRETATION AND CONCLUSION",)
                               )

if display == "NAMES OF PROJECT GROUP MEMBERS":
    st.subheader("GENNE BOADI-BOATENG- 22253167")
    st.subheader("ELIZABETH NYARKO-22253329")
    st.subheader("SYLVESTER AMOAH-22253362")
    st.subheader("MILLICENT ASEYE- 22253272")
    st.subheader("VICTORIA KWARKOR QUARTEY- 22258138")

elif display == "BACKGROUND & OVERVIEW":
    st.write("""  ## LOAN DEFAULT PREDICTION PROJECT REPORT
---------------
## Context
---------------

In today's financial landscape, lending institutions rely heavily on data to assess the risk of loan applicants. Defaulting on loans can significantly impact a bank's profitability and stability. As a result, credit risk modeling has become a key area where machine learning adds real-world value. By analyzing historical loan records and applicant characteristics, predictive models can help identify high-risk individuals before loans are approved.

## Introduction
This project applies supervised machine learning techniques to a real-world problem: predicting the likelihood of a loan applicant defaulting. Using the Loan Default Prediction Dataset from Kaggle, we build a classification model that classifies applicants into two categories:

0 – Will not default

1 – Will default

The pipeline includes data cleaning, preprocessing, training models (Random Forest and Logistic Regression), evaluating their performance, and deploying a prediction function for new applicants..

-----------------
## Objective
-----------------

The main objectives of this project are to:

Explore and understand the structure and insights from the Loan Default dataset.

Preprocess and transform the data into a format suitable for machine learning.

Train and compare classification models (Random Forest & Logistic Regression).

Evaluate models using precision, recall, F1-score, and confusion matrix.

Deploy a prediction interface for real-time risk classification.

Interpret model results and feature importance to offer real-world insights..

-------------------------
## Data Dictionary
-------------------------

* Column      -      Name	Description

* LoanID:	        Unique identifier for each loan record
* Age:	           Age of the applicant
* Income:	       Annual income of the applicant (in local currency e.g., GHS)
* LoanAmount:	   Total loan amount requested
* CreditScore:	   Applicant's credit score (300–850 scale)
* MonthsEmployed:  Total months the applicant has been employed
* NumCreditLines:  Number of active credit lines (loans, credit cards, etc.)
* InterestRate:	   Annual interest rate (%) on the loan
* LoanTerm:	       Duration of loan repayment (in months, e.g., 12/24/36/60)
* DTIRatio:	       Debt-to-Income ratio (e.g., 0.5 means 50% of income goes to debt)
* Education:	   Highest education level (e.g., High School, Bachelor's, Master's)
* EmploymentType:  Employment status (e.g., Full-time, Part-time, Unemployed)
* MaritalStatus:   Marital status (e.g., Single, Married, Divorced)
* HasMortgage:	   Whether the applicant has an existing mortgage (Yes/No)
* HasDependents:   Whether the applicant has dependents (children or others) (Yes/No)
* LoanPurpose:	   Reason for the loan (e.g., Auto, Education, Business, Personal, Other)
* HasCoSigner:     Whether the applicant has a co-signer or guarantor (Yes/No)
* Default	Target variable: 1 if the applicant defaulted, 0 otherwise

-------------------------
## Data Source 
Source: Kaggle – Loan Default Prediction Dataset    """)

elif display == "DATA IMPORT & OVERVIEW":
    st.subheader("ORIGINAL DATA")
    st.write(df1
             )

    # Display the shape of the data
    st.write("shape of data:", df_shape)

    # Display the total count of null values in the data
    st.markdown("SUM OF NULL VALUES PER COLUMN")
    st.write(df_nullvalues)

    # Display of the datatypes of each column
    st.markdown("DATA TYPES PRESENT")
    st.write(df_datatypes)

    # Display of the summary statistics of the data
    st.markdown("SUMMARY STATISTICS")
    st.write(df.describe())

    # Create a new subheader
    st.subheader("DISTRIBUTION PLOTS")

    # Identify a list of all numerical columns
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()

    ncols = 3

    # Calculate the number of rows needed to fit all plots
    nrows = (len(numerical_features) + ncols - 1) // ncols

    # Create the figure and subplots with the specified layout
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    # Flatten the axes array for easier looping
    axes = axes.flatten()

    # Loop through each numerical feature and plot it
    for i, col in enumerate(numerical_features):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(col)
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel('')

    # Hide any empty subplots if the total number doesn't perfectly fill the grid
    for j in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    st.subheader("BOX PLOTS OF VARIABLES SHOWING OUTLIERS")
    # specifying the number of columns
    ncols = 3

    # Calculate the number of rows needed to fit all plots
    nrows = (len(numerical_features) + ncols - 1) // ncols

    # Create the figure and subplots with the specified layout
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    # Flatten the axes array for easier looping
    axes = axes.flatten()

    # Loop through each numerical feature and plot it
    for i, col in enumerate(numerical_features):
        sns.boxplot(y=df[col], color='skyblue', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel('')

    # Hide any empty subplots if the total number doesn't perfectly fill the grid
    for j in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    st.subheader("ANALYSIS OF CATEGORICAL COLUMNS")
    # Define the number of columns and calculate the number of rows
    ncols = 3
    nrows = (len(categorical_cols) + ncols - 1) // ncols

    # Create the figure and subplots with the specified layout
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6 * nrows))

    # Flatten the axes array for easier looping
    axes = axes.flatten()

    # Define a list of colors to use for the pie chart slices
    colors = ['lightblue', 'lightgreen', 'orange', 'salmon', 'lightcoral']

    for i, col in enumerate(categorical_cols):
        # This is the corrected line:
        # Use df[col] to get the Series for the current column
        df[col].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            colors=colors,
            ax=axes[i]
        )
        axes[i].set_title(f'Proportion of {col}')
        axes[i].set_ylabel('')

    # Hide any unused subplots if the total number doesn't perfectly fill the grid
    for j in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    st.write("Due to an imbalanced dataset in the default column, we will need to balance it")
    st.write("we will later use the SMOTE method to balance it just before the splitting")




elif display == "DATA PREPROCESSING":
    st.subheader("VARIOUS PROCESSING STAGES")
    st.write("The processing begins with a correlation map to understand the relationship between variables")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the heatmap on the created axes
    sns.heatmap(df1.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)

    # Add a title to the plot
    ax.set_title('Correlation Heatmap')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Replace 'Yes' with 1 and 'No' with 0 in categorical columns
    st.write("Next was to replace all Yes and NO for the categorical columns that had these two values with 1s and 0s",
             df_encoded1)

    # Apply one-hot encoding to other categorical columns using get_dummies
    st.subheader("FULLY ENCODED DATA")
    st.write("Encoding remaining categorical data", df_encoded)

    # Display the data after scaling it with StandardScaler
    st.write("Standardization was done using the StandardScaler from sklearn and this was the result", X_scaled)

    # Show the subheader of the data and display the dataset
    st.subheader("RAW DATA")
    st.write(df1)

    # Showing the processed data
    st.subheader("PROCESSED/STANDARDIZED DATA-without target variable")
    st.write(X_scaled)


elif display == "MODEL TRAINING":
    st.subheader("VARIOUS SPLITS AND SHAPES OF SPLITS")  # Showing all the dataset to be used for training and testing
    st.subheader("TRAINING DATASET FOR INDEPENDENT VARIABLES")
    st.write("X_TRAIN", X_train, "NUMBER OF ROWS AND COLUMNS", X_train_shape)

    st.subheader("TRAINING DATASET FOR PREDICTOR VARIABLE")
    st.write("y_TRAIN", y_train, "NUMBER OF ROWS AND COLUMNS", y_train_shape)

    st.subheader("TESTING DATASET FOR INDEPENDENT VARIABLES")
    st.write("X_TEST", X_test, "NUMBER OF ROWS AND COLUMNS", X_test_shape)

    st.subheader("TESTING DATASET FOR PREDICTOR VARIABLE")
    st.write("y_TEST", y_test, "NUMBER OF ROWS AND COLUMNS", y_test_shape)

    # Create and train a Random Forest model, then display the trained model details
    st.subheader("RANDOM FOREST MODEL")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    st.write(rf_model.fit(X_train, y_train))

    # Create and train a Logistic Regression Model, then display the trained model details
    st.subheader("LOGISTIC REGRESSION MODEL")
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    st.write(logreg_model.fit(X_train, y_train))

    # Use the trained Random Forest Model to predict on test data and display the predicted values
    st.subheader("RANDOM FOREST PREDICTION ON TEST DATA")
    st.write(rf_model.predict(X_test))

    # Generate and display predictions on the test data using the trained Logistics Regression Model
    st.subheader("LOGISTIC REGRESSION PREDICTION ON TEST DATA")
    st.write(logreg_model.predict(X_test))



elif display == "MODEL EVALUATION":
    st.subheader("MODEL REPORTS AND CONFUSION MATRIX")

    # Evaluation metrics for Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)

    # Evaluation metrics for Logistic Regression
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train, y_train)
    y_pred_lr = logreg_model.predict(X_test)
    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    lr_conf_matrix = confusion_matrix(y_test, y_pred_lr)

    # Create two columns for side-by-side display
    rf_report_df = pd.DataFrame(rf_report).transpose()
    lr_report_df = pd.DataFrame(lr_report).transpose()

    # Display the classification report for the Random Forest Model in a table format
    st.markdown("### Random Forest Report")
    st.dataframe(rf_report_df)

    # Display the classification report for the Logistics Regression Model in a table format
    st.markdown("### Logistic Regression Report")
    st.dataframe(lr_report_df)

    # Set up a figure to display the Confusion Matrix for the Random Forest model
    st.subheader("Random Forest Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create the confusion matrix plot on the specified axes
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=['No Default', 'Default'], ax=ax)
    ax.set_title("Random Forest Confusion Matrix")
    st.pyplot(fig)

    st.subheader("LOGISTIC REGRESSION CONFUSION MATRIX")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the confusion matrix on the created axes
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_lr,
        display_labels=['No Default', 'Default'],
        ax=ax
    )

    # Set the title and other plot attributes on the axes
    ax.set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(fig)

    st.markdown("REPORTS AND CONFUSION MATRIX AFTER BALANCING USING SMOTE METHOD")
    # X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

    # rf_model1 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    # rf_model1.fit(X_resampled, y_resampled)
    # y_pred_rf1 = rf_model1.predict(X_test)
    # rf_report1 = classification_report(y_test, y_pred_rf1, output_dict=True)
    # rf_conf_matrix1 = confusion_matrix(y_test, y_pred_rf1)

    # Evaluation metrics for Logistic Regression
    # logreg_model1 = LogisticRegression(max_iter=1000, random_state=42)
    # logreg_model1.fit(X_resampled, y_resampled)
    # y_pred_lr1 = logreg_model1.predict(X_test)
    # lr_report1 = classification_report(y_test, y_pred_lr1, output_dict=True)
    # lr_conf_matrix1 = confusion_matrix(y_test, y_pred_lr1)

    # Create two columns for side-by-side display
    rf_report_df1 = pd.DataFrame(rf_report1).transpose()
    lr_report_df1 = pd.DataFrame(lr_report1).transpose()

    # Display the classification report for the Random Forest Model in a table format after balancing
    st.markdown("### Random Forest Report")
    st.dataframe(rf_report_df1)

    # Display the classification report for the Logistics Regression Model in a table format after balancing
    st.markdown("### Logistic Regression Report")
    st.dataframe(lr_report_df1)

    # Set up a figure to display the Confusion Matrix for the Random Forest model
    st.subheader("Random Forest Confusion Matrix After SMOTE METHOD")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create the confusion matrix plot on the specified axes
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf1, display_labels=['No Default', 'Default'], ax=ax)
    ax.set_title("Random Forest Confusion Matrix After SMOTE METHOD")
    st.pyplot(fig)

    st.subheader("LOGISTIC REGRESSION CONFUSION MATRIX AFTER SMOTE METHOD")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the confusion matrix on the created axes
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_lr1,
        display_labels=['No Default', 'Default'],
        ax=ax
    )
    ax.set_title("Logistic Regression Confusion Matrix After SMOTE METHOD")
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(fig)

    st.subheader("Top 7 Most Important Features (Random Forest)")

    # Get feature importances and top features
    importances = pd.Series(rf_model1.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(7)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the top features on the created axes
    top_features.plot(kind='barh', ax=ax)

    # Set plot attributes on the axes
    ax.set_title("Top 7 Most Important Features (Random Forest)")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()

    # Display the plot in Streamlit
    st.pyplot(fig)


elif display == "PREDICTION FUNCTION":

    # The prediction function
    def predict_loan_outcome(user_input):
        input_df = pd.DataFrame([0] * len(X.columns), index=X.columns).T
        for key, value in user_input.items():
            if key in input_df.columns:
                input_df.at[0, key] = value
            else:
                encoded_col = f"{key}_{value}"
                if encoded_col in input_df.columns:
                    input_df.at[0, encoded_col] = 1

        # Scale only the numerical columns
        numerical_cols = X.select_dtypes(include='number').columns
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = rf_model1.predict(input_df)[0]
        return "✅ APPROVED" if prediction == 0 else "⚠️ DEFAULT RISK"


    # Streamlit app layout
    st.title("Loan Outcome Prediction App 📊")
    st.write("Enter the details below to predict the loan outcome.")

    # Collect user input for prediction using sliders and number input fields
    age = st.slider("Age", 18, 100)
    NumCreditLines = st.number_input("Number of Credit Lines", 0, 5)
    loanTerm = st.number_input("Loan Term", 1, 60)
    HasDependents = st.number_input("Has Dependents", 0, 1)
    months_employed = st.number_input("Months Employed", 0, 200)
    interest_rate = st.number_input("Interest Rate (%)", 2, 25)
    HasCoSigner = st.number_input("Has a Co-Signer", 0, 1)

    # Create a dictionary from the user's input
    user_input = {
        'Age': age,
        'Nuber of Credit lINES': NumCreditLines,
        'Loan Term': loanTerm,
        'Has Dependents': HasDependents,
        'MonthsEmployed': months_employed,
        'InterestRate': interest_rate,
        'HasCoSigner': HasCoSigner
    }

    st.write("kindly click this button to know the outcome")
    # The prediction button
    if st.button("Predict Loan Outcome"):
        prediction = predict_loan_outcome(user_input)
        st.subheader("Prediction:")
        if "APPROVED" in prediction:
            st.success(prediction)
        else:
            st.warning(prediction)

    # Show a subheading for displaying a sample decision tree
    st.subheader("Sample Decision Tree from Random Forest")

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(30, 20))

    # Plot the decision tree on the created axes
    plot_tree(rf_model1.estimators_[0],
              feature_names=X.columns,
              class_names=["Approved", "Default"],
              filled=True,
              max_depth=3,
              fontsize=15,
              ax=ax)

    # Set the title and other plot attributes on the axes
    plt.tight_layout()
    st.pyplot(fig)



elif display == "INTERPRETATION AND CONCLUSION":
    st.write("""    
    # Loan Default Prediction Project - Conclusion and Insights

## Summary of Work 

This project involved the end-to-end development of a machine learning model to predict loan default risk using a real-world dataset. Key stages included:

- **Data Import and Exploration:** Loaded the Loan Default dataset, inspected its structure, identified null values (none significant were reported), and understood data types and basic statistics.
- **Data Visualization:** Explored the distribution of numeric features via histograms and boxplots to detect outliers, and analyzed categorical columns through pie charts to understand class proportions.
- **Data Preprocessing:** Removed the irrelevant `LoanID` feature, converted categorical variables including Yes/No features into numerical formats using label encoding and one-hot encoding. Numerical features were standardized using StandardScaler to optimize model training.
- **Model Development:** Built and trained two classification models: a Random Forest Classifier with 50 trees and a maximum depth of 10, and a Logistic Regression model with up to 1000 iterations.
- **Model Evaluation:** Assessed both models on test data using classification reports (precision, recall, F1-score) and confusion matrices. Visualized model performance differences to inform selection.
- **Feature Importance:** Identified the top 7 most significant predictors for loan default using the feature importance from the Random Forest model.
- **Prediction Interface:** Developed a Streamlit web app interface allowing user-friendly input of applicant data to obtain real-time loan default risk predictions. A sample decision tree visualization from the Random Forest provides transparency into model decisions.

## Key Findings

- The **Random Forest model** generally offered superior performance on classification metrics compared to Logistic Regression, suggesting better handling of complex feature interactions in this dataset.
- Top features impacting default prediction includes:
  - Credit Score
  - Debt-to-Income Ratio (DTIRatio)
  - Loan Amount
  - Interest Rate
  - Months Employed
  - Income
  - Number of Credit Lines
- The model effectively differentiates between applicants likely to default and those who are not, allowing financial institutions to assess risk proactively.

## Practical Implications

Financial institutions can leverage this predictive model to:

- **Improve loan approval decisions:** By identifying high-risk applicants early, reducing non-performing loans.
- **Optimize interest rates and terms:** Tailoring offers based on predicted default probability.
- **Support regulatory compliance and risk management:** Providing data-driven justification for lending policies.
- **Enhance customer targeting:** Identifying low-risk customers for better loan products and higher profits.

## Limitations and Recommendations

- **Data Quality:** The model's accuracy depends heavily on data accuracy and completeness. Real-world applications should ensure continual data validation.
- **Model Complexity vs Interpretability:** While Random Forest offers strong predictive power, logistic regression provides clearer interpretability. A hybrid approach or explainer tools (like SHAP values) could further enhance trust.
- **Feature Coverage:** Additional relevant features such as credit history length, asset details, or macroeconomic indicators might improve predictions.
- **Deployment and Monitoring:** Continuous monitoring of model performance over time is essential to detect performance drift due to changing borrower behaviors or economic conditions.

## Future Work

- Incorporate additional data sources to enrich applicant profiles.
- Explore advanced models (e.g., gradient boosting, neural networks) for improved accuracy.
- Implement explainability techniques for better stakeholder understanding.
- Build an end-to-end automated pipeline integrating data ingestion, retraining, and deployment.

## Conclusion

This project demonstrates the practical utility of machine learning in managing credit risk. By combining rigorous data preprocessing, careful model training, evaluation, and an interactive prediction tool, the work equips lending institutions with an effective method for mitigating loan default risk, improving decision-making, and ultimately enhancing financial stability.    
""")