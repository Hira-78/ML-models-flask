from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from general_operations import prep_for_training, save_model


def apply_logistic_regression(file_path):
    X_train_scaled, y_train, X_test_scaled, y_test, _, classes = prep_for_training(file_path)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print("\nAccuracy of Logistic regression:", accuracy)
    save_model(model, filename="logistic_regression_model.pkl")
    return model
