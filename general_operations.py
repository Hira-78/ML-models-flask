import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import io, base64


def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model(filename):
    with open(filename, "rb") as file:
        loaded_model = pickle.load(file)
        return loaded_model


def load_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.dropna()
    return dataframe


def prep_for_training(file_path):
    dataframe = load_data(file_path)
    dataframe = dataframe.dropna()

    # print("null values", dataframe.isnull().sum())
    # rows, columns = dataframe.shape

    # label column
    y = dataframe.iloc[:, -1]

    # copy for one-hot encoding to preserve original structure
    dataframe_encoded = dataframe.copy()

    # One-hot encoding categorical columns in the dataframe copy
    one_hot_encoding = OneHotEncoder(handle_unknown="ignore")
    categorical_columns = dataframe_encoded.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        feature_array = one_hot_encoding.fit_transform(dataframe_encoded[[column]]).toarray()
        feature_labels = one_hot_encoding.get_feature_names_out([column])
        one_encoded_dataframe = pd.DataFrame(feature_array, columns=feature_labels)
        dataframe_encoded = pd.concat([dataframe_encoded, one_encoded_dataframe], axis=1)
        dataframe_encoded = dataframe_encoded.drop(column, axis=1)

    # features from the encoded DataFrame
    X = dataframe_encoded.iloc[:, :-1]
    # unique classes
    unique_classes = y.unique()
    print(f"Unique classes: {unique_classes}")

    # showing data plot
    colors = ["#440154", "#35978f", "#fde725", "#d01c8b"]  # Example dark colors
    sns.set_style("whitegrid")
    sns.countplot(x=y, hue=y, palette=colors)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Classes in the Dataset")
    # plt.show()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.clf()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test, img_data, unique_classes


def calculate_confusion_matrices(x_test, y_test, filenames):
    confusion_matrices = []
    confusion_matrices_images = []

    for filename in filenames:
        model = load_model(filename)
        predictions = model.predict(x_test)
        c_matrix = confusion_matrix(y_test, predictions)
        confusion_matrices.append(c_matrix)
        img = plot_confusion_matrix(c_matrix, title=f"Confusion Matrix for {filename}")
        confusion_matrices_images.append(img)

    return confusion_matrices, confusion_matrices_images


def plot_confusion_matrix(c_matrix, title):
    plt.clf()
    sns.heatmap(c_matrix, annot=True, fmt="d", cmap="inferno")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    return img_data


def calculate_specificity(cm):
    true_negative = np.sum(np.diag(np.sum(cm, axis=1)) - np.diag(cm))
    false_positive = np.sum(cm, axis=0) - np.diag(cm)

    denominator = (true_negative + false_positive)
    if np.any(denominator == 0):
        specificity = 0
    else:
        specificity = true_negative / denominator
    return np.mean(specificity)


def calculate_sensitivity(cm):
    true_positive = np.diag(cm)
    false_negative = np.sum(cm, axis=1) - true_positive

    denominator = (true_positive + false_negative)
    if np.any(denominator == 0):
        sensitivity = 0
    else:
        sensitivity = true_positive / denominator
    return np.mean(sensitivity)


def calculate_sensitivity_for_matrices(confusion_matrices):
    result_sensitivity = []
    for cm in confusion_matrices:
        sensitivity = calculate_sensitivity(cm)
        result_sensitivity.append(sensitivity)
    return result_sensitivity


def calculate_specificity_for_matrices(confusion_matrices):
    result_specificity = []
    for cm in confusion_matrices:
        specificity = calculate_specificity(cm)
        result_specificity.append(specificity)
    return result_specificity
