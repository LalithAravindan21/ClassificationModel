import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    st.title("Machine Learning Model Deployment with Streamlit")

    st.sidebar.subheader("Upload Data")

    # Allow user to upload train data
    train_file = st.sidebar.file_uploader("Upload Train Data", type=["xlsx"])

    # Allow user to upload test data
    test_file = st.sidebar.file_uploader("Upload Test Data", type=["xlsx"])

    if train_file is not None and test_file is not None:
        train_data = pd.read_excel(train_file)
        test_data = pd.read_excel(test_file)

        # Display the first few rows of train and test data
        st.subheader("Train data:")
        st.write(train_data.head())
        st.subheader("Test data:")
        st.write(test_data.head())

        # Separate features (X_train) and target variable (y_train) in train dataset
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']

        # Split the train dataset into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }

        # Train and make predictions for each model
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on the validation set
            y_pred = model.predict(X_val)
            
            # Print accuracy on the validation set
            st.write(f'{name} Accuracy: {accuracy_score(y_val, y_pred)}')

            # Make predictions on the test dataset using the trained model
            test_predictions = model.predict(test_data.drop(columns=['predicted_target'], errors='ignore'))

            # Create a DataFrame to store the predicted target values
            predicted_test_data = pd.DataFrame(test_predictions, columns=['predicted_target'], index=test_data.index)

            # Display the test dataset with predicted target values
            st.subheader(f"{name} Predicted Output:")
            st.write(predicted_test_data.head())

if __name__ == "__main__":
    main()
