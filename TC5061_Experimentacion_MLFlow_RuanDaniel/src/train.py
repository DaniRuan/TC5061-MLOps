# Entrenamiento parametrizado de un baseline de clasificación con MLflow.
# Daniel R.

# librerías
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"


def parse_args():
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una regresión logística sobre Breast Cancer Wisconsin "
            "y registra parámetros, métricas, artefactos y modelo en MLflow."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/breast_cancer.csv"),
        help="Ruta al archivo CSV.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="URI del backend de MLflow Tracking.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="TC5061_Experimentacion_LogisticRegression",
        help="Nombre del experimento de MLflow.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre descriptivo del run. Si se omite, se genera automáticamente.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Algoritmo de optimización de LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0001,
        help="Tolerancia para el criterio de convergencia.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporción reservada para prueba.",
    )

    args = parser.parse_args()

    if args.C <= 0:
        parser.error("--C debe ser mayor que cero.")
    if args.max_iter <= 0:
        parser.error("--max-iter debe ser mayor que cero.")
    if args.tol <= 0:
        parser.error("--tol debe ser mayor que cero.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1.")

    return args


def set_random_seed(seed):
    """Fija las semillas disponibles para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


def load_data(data_path):
    """Carga y valida el dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "El dataset contiene valores faltantes. "
            "Revise la estrategia de limpieza antes de entrenar."
        )

    df = df.drop_duplicates().copy()
    df = df.set_index(ID_COLUMN)

    observed_classes = set(df[TARGET_COLUMN].unique())
    if observed_classes != {"B", "M"}:
        raise ValueError(
            f"Clases inesperadas en {TARGET_COLUMN}: {observed_classes}"
        )

    return df


def prepare_data(df, test_size, random_state):
    """Separa predictores y objetivo, y crea los conjuntos train/test."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"B": 0, "M": 1})

    if y.isna().any():
        raise ValueError("No fue posible codificar todas las clases.")

    non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"Se encontraron predictores no numéricos: {non_numeric}"
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model(C, solver, max_iter, tol, random_state):
    """Construye el pipeline de escalado y regresión logística."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test):
    """Calcula predicciones y métricas de clasificación."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_proba)
        ),
    }

    return y_pred, y_proba, metrics


def save_artifacts(
    y_test,
    y_pred,
    y_proba,
    metrics,
    output_dir,
):
    """Genera matriz de confusión y archivos de resultados."""
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = output_dir / "confusion_matrix.png"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benigno", "Maligno"],
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close()

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    predictions_df = pd.DataFrame(
        {
            "diagnostico_real": y_test,
            "diagnostico_predicho": y_pred,
            "probabilidad_maligno": y_proba,
        },
        index=y_test.index,
    )
    predictions_df.index.name = ID_COLUMN
    predictions_df.to_csv(predictions_path)

    return confusion_path, metrics_path, predictions_path


def create_run_name(args):
    """Genera un nombre reproducible y descriptivo para la corrida."""
    if args.run_name:
        return args.run_name

    c_value = str(args.C).replace(".", "p")
    tol_value = str(args.tol).replace(".", "p")

    return (
        f"logreg_C{c_value}_{args.solver}_"
        f"iter{args.max_iter}_tol{tol_value}_"
        f"seed{args.random_state}"
    )


def log_experiment(
    args,
    model,
    X_train,
    X_test,
    metrics,
    artifact_paths,
):
    """Registra parámetros, métricas, artefactos y modelo en MLflow."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = create_run_name(args)

    parameters = {
        "model_type": "LogisticRegression",
        "scaler": "StandardScaler",
        "C": args.C,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "n_train_rows": X_train.shape[0],
        "n_test_rows": X_test.shape[0],
        "n_features": X_train.shape[1],
        "data_path": str(args.data_path),
    }

    signature = infer_signature(
        X_train,
        model.predict(X_train),
    )
    input_example = X_train.head(5).copy()

    with mlflow.start_run(run_name=run_name) as run:
        # log_param y log_metric
        for name, value in parameters.items():
            mlflow.log_param(name, value)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.set_tags(
            {
                "course": "TC5061",
                "student": "Daniel Ruan Aguilar",
                "task": "binary_classification",
                "stage": "experimentation",
                "dataset": "Breast Cancer Wisconsin Diagnostic",
            }
        )

        for artifact_path in artifact_paths:
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="evaluation",
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id

    return run_id, run_name


def main():
    # Flujo completo de entrenamiento y registro.
    args = parse_args()
    set_random_seed(args.random_state)

    df = load_data(args.data_path)

    X_train, X_test, y_train, y_test = prepare_data(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_model(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    y_pred, y_proba, metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    run_output_dir = Path("outputs") / create_run_name(args)

    artifact_paths = save_artifacts(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        output_dir=run_output_dir,
    )

    run_id, run_name = log_experiment(
        args=args,
        model=model,
        X_train=X_train,
        X_test=X_test,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )

    print("\nEntrenamiento y registro completados.")
    print(f"Run name: {run_name}")
    print(f"Run ID: {run_id}")
    print(f"Experimento: {args.experiment_name}")
    print("\nMétricas:")

    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()