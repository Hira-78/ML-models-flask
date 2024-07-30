from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from general_operations import prep_for_training, save_model


def apply_knn(data_file):
    X_train_scaled, y_train, X_test_scaled, y_test, _, classes = prep_for_training(data_file)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)
    print("Accuracy of K-NN : ", accuracy_score(y_test, prediction))
    save_model(model, filename="knn_model.pkl")
    return model
