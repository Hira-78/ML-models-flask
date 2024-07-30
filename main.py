from flask import Flask, render_template, request, session, redirect, url_for
from logistic_regression import prep_for_training, apply_logistic_regression
from svm import apply_svm
from decision_tree import apply_decision_tree
from random_forest import apply_random_forest
from k_nearest_neighbor import apply_knn
from general_operations import calculate_confusion_matrices, calculate_sensitivity_for_matrices, calculate_specificity_for_matrices
import os

app = Flask(__name__)
app.secret_key = "123"


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/load_data", methods=["GET", "POST"])
def load_data_page():
    if request.method == "GET":
        return render_template("load_data.html")
    elif request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join(app.static_folder, file.filename)
        file.save(file_path)
        session['file_path'] = file_path
        session['file_path'] = file_path
        return redirect(url_for("prep_for_train"))
    else:
        return render_template("load_data.html")


@app.route("/prep_for_train", methods=["GET", "POST"])
def prep_for_train():
    if request.method == "GET":
        return render_template("prep_for_train.html")
    elif request.method == "POST":
        file_path = session.get('file_path')
        X_train_scaled, y_train, X_test_scaled, y_test, plot, classes = prep_for_training(file_path)
        return render_template("prep_for_train.html", graph=plot, unique_classes=classes)


@app.route("/models", methods=["GET", "POST"])
def models_page():
    if request.method == "GET":
        return render_template("models.html")
    elif request.method == "POST":
        file_path = session.get('file_path')
        if request.form.get("logistic_regression"):
            apply_logistic_regression(file_path)
        if request.form.get("svm"):
            apply_svm(file_path)
        if request.form.get("decision_tree"):
            apply_decision_tree(file_path)
        if request.form.get("random_forest"):
            apply_random_forest(file_path)
        if request.form.get("knn"):
            apply_knn(file_path)
    return render_template("models.html")


@app.route("/metrics", methods=["GET", "POST"])
def show_metrics_page():
    if request.method == "GET":
        return render_template("metrics.html")
    if request.method == "POST":
        filenames = ["logistic_regression_model.pkl", "svm_model.pkl", "decision_tree_model.pkl", "random_forest_model.pkl", "knn_model.pkl"]
        file_path = session.get("file_path")
        action = request.form.get("action")
        print(action)
        X_train_scaled, y_train, X_test_scaled, y_test, plot, classes = prep_for_training(file_path)
        matrices, confusion_matrix_data = calculate_confusion_matrices(X_test_scaled, y_test, filenames)
        sensitivity = calculate_sensitivity_for_matrices(matrices)
        specificity = calculate_specificity_for_matrices(matrices)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        if action:
            return render_template("metrics.html", confusion_matrix_data=confusion_matrix_data, sensitivity_data=sensitivity, specificity_data=specificity)
        return render_template("metrics.html", confusion_matrix_data=confusion_matrix_data,
                               sensitivity_data=sensitivity, specificity_data=specificity)

    return render_template("metrics.html")
if __name__ == "__main__":
    app.run(debug=True)
