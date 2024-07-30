from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from general_operations import prep_for_training, save_model


def apply_decision_tree(data_file):
    X_train_scaled, y_train, X_test_scaled, y_test, _, _ = prep_for_training(data_file)
    # Initialize and fit the Decision Tree model
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy of Decision Tree: {accuracy}")

    save_model(model, filename="decision_tree_model.pkl")
    return model
