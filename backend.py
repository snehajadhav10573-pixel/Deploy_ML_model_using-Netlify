from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# ================= CLASSIFICATION SETUP =================
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=200)
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# ================= HOME =================
@app.route("/")
def home():
    return "Backend Running Successfully ✅"


# ================= REGRESSION: LR + RF =================
@app.route("/LR_RF", methods=["POST"])
def LR_RF():
    req_data = request.json

    data = req_data.get("data", [])
    x_col = req_data.get("xColumn")
    y_col = req_data.get("yColumn")
    model_type = req_data.get("model", 1)

    if not data:
        return jsonify({"error": "No data received"})

    if x_col not in data[0] or y_col not in data[0]:
        return jsonify({"error": "Invalid column names"})

    x_vals = [float(row[x_col]) for row in data]
    y_vals = [float(row[y_col]) for row in data]

    combined = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    x_vals = [p[0] for p in combined]
    y_vals = [p[1] for p in combined]

    n = len(x_vals)

    # LINEAR REGRESSION
    if model_type == 1:
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x_vals[i] * y_vals[i] for i in range(n))
        sum_x2 = sum(x * x for x in x_vals)

        denominator = (n * sum_x2 - sum_x * sum_x)
        if denominator == 0:
            return jsonify({"error": "Division by zero"})

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        predictions = [slope * x + intercept for x in x_vals]

        min_x = min(x_vals)
        max_x = max(x_vals)
        line_x = [min_x, max_x]
        line_y = [slope * min_x + intercept, slope * max_x + intercept]

        return jsonify({
            "model": 1,
            "x": x_vals,
            "y": y_vals,
            "slope": slope,
            "intercept": intercept,
            "predictions": predictions,
            "line_x": line_x,
            "line_y": line_y
        })

    # RANDOM FOREST REGRESSOR
    elif model_type == 2:
        X_reg = np.array(x_vals).reshape(-1, 1)
        y_reg = np.array(y_vals)

        trees = req_data.get("trees", 20)
        depth = req_data.get("depth", 4)

        reg_model = RandomForestRegressor(
            n_estimators=int(trees),
            max_depth=int(depth),
            random_state=42
        )
        reg_model.fit(X_reg, y_reg)
        predictions = reg_model.predict(X_reg).tolist()

        mse = mean_squared_error(y_reg, predictions)
        r2 = r2_score(y_reg, predictions)

        return jsonify({
            "model": 2,
            "x": x_vals,
            "y": y_vals,
            "predictions": predictions,
            "mse": round(float(mse), 4),
            "r2": round(float(r2), 4),
            "accuracy": round(float(r2 * 100), 2),
            "slope": None,
            "intercept": None,
            "line_x": [],
            "line_y": []
        })

    else:
        return jsonify({"error": "Invalid model type"})


# ================= OVERALL RF ACCURACY =================
@app.route("/overall-rf-accuracy", methods=["POST"])
def overall_rf_accuracy():
    req_data = request.json
    data = req_data["data"]

    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    accuracies = []

    for x_col in columns:
        for y_col in columns:
            if x_col == y_col:
                continue

            x_vals = [float(row[x_col]) for row in data]
            y_vals = [float(row[y_col]) for row in data]

            X_reg = np.array(x_vals).reshape(-1, 1)
            y_reg = np.array(y_vals)

            model = RandomForestRegressor(
                n_estimators=20, max_depth=4, random_state=42
            )
            model.fit(X_reg, y_reg)
            predictions = model.predict(X_reg)

            r2 = r2_score(y_reg, predictions)
            accuracies.append(r2 * 100)

    overall_accuracy = sum(accuracies) / len(accuracies)
    return jsonify({"overall_accuracy": round(overall_accuracy, 2)})


# ================= CLASSIFICATION: PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]
    except:
        return jsonify({"error": "Invalid input"}), 400

    input_array = np.array(features).reshape(1, -1)

    start = time.time()
    lr_pred = lr_model.predict(input_array)[0]
    lr_time = time.time() - start

    start = time.time()
    knn_pred = knn_model.predict(input_array)[0]
    knn_time = time.time() - start

    start = time.time()
    rf_pred = rf_model.predict(input_array)[0]
    rf_time = time.time() - start

    return jsonify({
        "predictions": {
            "Logistic Regression": target_names[lr_pred],
            "KNN": target_names[knn_pred],
            "Random Forest": target_names[rf_pred]
        },
        "timing": {
            "Logistic Regression": round(lr_time, 6),
            "KNN": round(knn_time, 6),
            "Random Forest": round(rf_time, 6)
        }
    })


# ================= CLASSIFICATION: METRICS =================
@app.route("/metrics", methods=["GET"])
def metrics():
    models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Random Forest": rf_model
    }

    result = {}
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        fpr = {}
        tpr = {}
        roc_auc = {}

        try:
            y_score = model.predict_proba(X_test)
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        except:
            pass

        result[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "roc": {
                "fpr": {str(k): v.tolist() for k, v in fpr.items()},
                "tpr": {str(k): v.tolist() for k, v in tpr.items()},
                "auc": roc_auc
            }
        }

    return jsonify(result)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, port=5000)