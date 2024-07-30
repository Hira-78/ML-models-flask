from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from general_operations import prep_for_training, save_model


def apply_random_forest(data_file):
    X_train_scaled, y_train, X_test_scaled, y_test, _, classes = prep_for_training(data_file)
    model = RandomForestClassifier(max_depth=8, min_samples_split=10, random_state=5)
    model.fit(X_train_scaled, y_train)

    # for feature info
    # model.feature_importances_

    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy of Random Forest: ", accuracy)
    save_model(model, filename="random_forest_model.pkl")
    return model
