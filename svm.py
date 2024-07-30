from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from general_operations import prep_for_training, save_model


def apply_svm(file_path):
    X_train_scaled, y_train, X_test_scaled, y_test, _, classes = prep_for_training(file_path)

    # Create and train the SVM model
    model = SVC(kernel='rbf', C=1.0, random_state=42, degree=3)
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print("\nAccuracy of SYM:", accuracy)

    save_model(model, filename="svm_model.pkl")
    return model
