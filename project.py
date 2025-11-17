import math
import numpy as ny
import pandas as ps
import matplotlib.pyplot as pt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    auc,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

RANDOM = 42
ny.random.seed(RANDOM)
tf.random.set_seed(RANDOM)



# Metric helper method
def calc_metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()

    def safe_division(a, b):
        return a / b if b != 0 else 0.0

    # Positives (P) and negatives (N)
    p = tp + fn
    n = tn + fp

    # Basic rates
    tprate = safe_division(tp, p)   
    tnrate = safe_division(tn, n)   
    fprate = safe_division(fp, n)
    fnrate = safe_division(fn, p)
    precision = safe_division(tp, tp + fp)
    f1 = safe_division(2 * tp, 2 * tp + fp + fn)

    # Accuracy, error, balanced accuracy
    accuracy = safe_division(tp + tn, p + n)
    error_rate = 1.0 - accuracy
    bacc = 0.5 * (tprate + tnrate)

    # True Skill Statistic
    tss = tprate - fprate

    # Heidke Skill Score (HSS)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_division(2 * (tp * tn - fp * fn), denom)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "P": p,
        "N": n,
        "TPR": tprate,
        "TNR": tnrate,
        "FPR": fprate,
        "FNR": fnrate,
        "Precision": precision,
        "F1": f1,
        "Accuracy": accuracy,
        "Error_rate": error_rate,
        "BACC": bacc,
        "TSS": tss,
        "HSS": hss,
    }


# Conv1D model helper method
def build_conv1d(n_features):
    model = Sequential(
        [
            Conv1D(64, 3, activation="relu", input_shape=(n_features, 1)),
            Dropout(0.2),
            Conv1D(32, 3, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
# 1. Load dataset
    path = "wine_quality.csv"
    data = ps.read_csv(path)

    print("Raw dataset shape:", data.shape)
    print(data.head(3))

    if "type" in data.columns:
        print("\nWine type counts:")
        print(data["type"].value_counts())

    df = data.copy()

# 2. Prepare features (X) and target (y)

    # target is the last column: wine type (red/white)
    target_col = df.columns[-1]

    # Label-encode the target column inside df so it becomes numeric
    if df[target_col].dtype == "object":
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col])

    # Split features and target
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Standardize features
    scaler = StandardScaler()
    x_scaled = ps.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # Convert to numpy arrays
    X = x_scaled.values.astype("float32")
    y_arr = y.values.astype("int32")

    print("\nPrepared data:")
    print("  Features shape:", X.shape)
    print("  Labels shape:", y_arr.shape)
    print("  Positives:", int(y_arr.sum()))
    print("  Negatives:", int(len(y_arr) - y_arr.sum()))

# 3. Basic data visualization

    # Target distribution (encoded wine type)
    print("\nTarget column:", target_col)
    print("Value counts for wine type (encoded):")
    print(df[target_col].value_counts())

    pt.figure(figsize=(5, 4))
    df[target_col].value_counts().sort_index().plot(kind="bar")
    pt.title("Wine Type Distribution (Encoded)")
    pt.xlabel("Wine Type (0/1)")
    pt.ylabel("Count")
    pt.tight_layout()
    pt.show()

    # Histograms of numeric features (excluding target for clarity)
    numeric_feature_cols = df.drop(columns=[target_col]).select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    pt.figure(figsize=(12, 8))
    df[numeric_feature_cols].hist(figsize=(12, 8), bins=20)
    pt.suptitle("Feature Distributions (Numeric Columns)", y=1.02)
    pt.tight_layout()
    pt.show()

    # Full correlation heatmap 
    corr = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).corr()

    pt.figure(figsize=(10, 8))
    import seaborn as sns  # local import for plotting

    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
    pt.title("Correlation Heatmap (Features + Encoded Target)")
    pt.tight_layout()
    pt.show()


# 4. 10-fold cross-validation (RF, KNN, Conv1D)

    kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM)

    results = {"RF": [], "KNN": [], "Conv1D": []}
    n_features = X.shape[1]

    for fold, (train_indices, test_indices) in enumerate(
        kfold.split(X, y_arr), start=1
    ):
        x_train, x_test = X[train_indices], X[test_indices]
        y_train, y_test = y_arr[train_indices], y_arr[test_indices]

        # Baseline for Brier Skill Score 
        base_prob = y_train.mean()
        baseline_prob = ny.full_like(y_test, fill_value=base_prob, dtype="float32")

        fold_metrics = {}

        # Random Forest and KNN
        classic_models = {
            "RF": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM
            ),
            "KNN": KNeighborsClassifier(n_neighbors=13),
        }

        for name, model in classic_models.items():
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            cm = confusion_matrix(y_test, pred)
            metrics_dict = calc_metrics_from_cm(cm)

            # Brier score and Brier Skill Score
            bs = brier_score_loss(y_test, prob)
            bs_ref = brier_score_loss(y_test, baseline_prob)
            bss = 1.0 - bs / bs_ref if bs_ref > 0 else 0.0

            metrics_dict["AUC"] = roc_auc_score(y_test, prob)
            metrics_dict["BS"] = bs
            metrics_dict["BSS"] = bss

            results[name].append(metrics_dict)
            fold_metrics[name] = metrics_dict

        # Conv1D
        x_train_seq = x_train.reshape(-1, n_features, 1)
        x_test_seq = x_test.reshape(-1, n_features, 1)

        conv_model = build_conv1d(n_features)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        conv_model.fit(
            x_train_seq,
            y_train,
            epochs=40,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )

        conv_prob = conv_model.predict(x_test_seq).ravel()
        conv_pred = (conv_prob >= 0.5).astype(int)

        cm_conv = confusion_matrix(y_test, conv_pred)
        metrics_conv = calc_metrics_from_cm(cm_conv)

        bs_conv = brier_score_loss(y_test, conv_prob)
        bs_ref_conv = brier_score_loss(y_test, baseline_prob)
        bss_conv = 1.0 - bs_conv / bs_ref_conv if bs_ref_conv > 0 else 0.0

        metrics_conv["AUC"] = roc_auc_score(y_test, conv_prob)
        metrics_conv["BS"] = bs_conv
        metrics_conv["BSS"] = bss_conv

        results["Conv1D"].append(metrics_conv)
        fold_metrics["Conv1D"] = metrics_conv

        # Per fold summary table
        cols_show = [
            "TP", "TN", "FP", "FN",
            "P", "N",
            "TPR", "TNR", "FPR", "FNR",
            "Precision", "F1",
            "Accuracy", "Error_rate",
            "BACC", "TSS", "HSS",
            "AUC", "BS", "BSS",
        ]

        fold_df = ps.DataFrame(fold_metrics).T[cols_show]

        print(f"\n-- Fold {fold} results --")
        print(fold_df.round(3))

        
    # 5. Show all 10 folds for each algorithm (RF,KNN,Conv1D)
    # metrics
    cols_show = [
        "TP", "TN", "FP", "FN", "P", "N",
        "TPR", "TNR", "FPR", "FNR",
        "Precision", "F1", "Accuracy", "Error_rate",
        "BACC", "TSS", "HSS",
        "AUC", "BS", "BSS",
    ]

    # fold names
    num_folds = len(next(iter(results.values())))  # length of any list in results
    fold_names = [f"fold{i}" for i in range(1, num_folds + 1)]

    for model_name in ["KNN", "RF", "Conv1D"]:
        df_model = ps.DataFrame(results[model_name])[cols_show]
        df_model.index = fold_names

        print(f"\nAll {model_name} folds metrics:\n")
        print(df_model.T.round(3))
        print("\n")

    # 6. Average metrics over all folds

    metrics_order = [
        "TP", "TN", "FP", "FN",
        "P", "N",
        "TPR", "TNR", "FPR", "FNR",
        "Precision", "F1",
        "Accuracy", "Error_rate",
        "BACC", "TSS", "HSS",
        "AUC", "BS", "BSS",
    ]

    avg_tables = {}
    for name, metrics_list in results.items():
        df_metrics = ps.DataFrame(metrics_list)
        avg_tables[name] = df_metrics[metrics_order].mean()

    avg_df = ps.DataFrame(avg_tables, index=metrics_order)

    print("\n========== Mean metric values over 10 folds ==========")
    print(avg_df.round(3))

    # 7. ROC curves on 20% hold-out test set

    x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        random_state=RANDOM,
        stratify=y_arr,
    )

    # Random Forest ROC
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM,
    )
    rf_model.fit(x_train_all, y_train_all)
    rf_prob = rf_model.predict_proba(x_test_all)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test_all, rf_prob)
    auc_rf = auc(fpr_rf, tpr_rf)

    pt.figure(figsize=(6, 6))
    pt.plot(fpr_rf, tpr_rf, label=f"ROC curve (area = {auc_rf:.2f})")
    pt.plot([0, 1], [0, 1], linestyle="--")
    pt.xlim([0.0, 1.0])
    pt.ylim([0.0, 1.05])
    pt.xlabel("False Positive Rate")
    pt.ylabel("True Positive Rate")
    pt.title("Random Forest ROC Curve")
    pt.legend(loc="lower right")
    pt.grid(True)
    pt.show()

    # KNN ROC
    knn_model = KNeighborsClassifier(n_neighbors=13)
    knn_model.fit(x_train_all, y_train_all)
    knn_prob = knn_model.predict_proba(x_test_all)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_test_all, knn_prob)
    auc_knn = auc(fpr_knn, tpr_knn)

    pt.figure(figsize=(6, 6))
    pt.plot(fpr_knn, tpr_knn, label=f"ROC curve (area = {auc_knn:.2f})")
    pt.plot([0, 1], [0, 1], linestyle="--")
    pt.xlim([0.0, 1.0])
    pt.ylim([0.0, 1.05])
    pt.xlabel("False Positive Rate")
    pt.ylabel("True Positive Rate")
    pt.title("KNN ROC Curve")
    pt.legend(loc="lower right")
    pt.grid(True)
    pt.show()

    # Conv1D ROC
    x_train_seq_all = x_train_all.reshape(-1, X.shape[1], 1)
    x_test_seq_all = x_test_all.reshape(-1, X.shape[1], 1)

    conv_model_roc = build_conv1d(X.shape[1])
    early_stop_roc = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    conv_model_roc.fit(
        x_train_seq_all,
        y_train_all,
        epochs=40,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop_roc],
        verbose=0,
    )

    conv_prob = conv_model_roc.predict(x_test_seq_all).ravel()
    fpr_conv, tpr_conv, _ = roc_curve(y_test_all, conv_prob)
    auc_conv = auc(fpr_conv, tpr_conv)

    pt.figure(figsize=(6, 6))
    pt.plot(fpr_conv, tpr_conv, label=f"ROC curve (area = {auc_conv:.2f})")
    pt.plot([0, 1], [0, 1], linestyle="--")
    pt.xlim([0.0, 1.0])
    pt.ylim([0.0, 1.05])
    pt.xlabel("False Positive Rate")
    pt.ylabel("True Positive Rate")
    pt.title("Conv1D ROC Curve")
    pt.legend(loc="lower right")
    pt.grid(True)
    pt.show()


if __name__ == "__main__":
    main()
