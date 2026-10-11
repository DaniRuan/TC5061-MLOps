# TC5061 | Experimentación base con MLflow

## Autor

Daniel Ruán Aguilar  
Matrícula: A01731921

## Descripción

Este repositorio convierte el baseline exploratorio del dataset Breast Cancer Wisconsin (Diagnostic) en un script de entrenamiento parametrizado e instrumentado con MLflow Tracking.

El pipeline incluye:

- Estandarización con StandardScaler.
- Clasificación con LogisticRegression.
- División estratificada en entrenamiento y prueba.
- Registro de parámetros, métricas, artefactos y modelo en MLflow.
- Control de la semilla para reproducibilidad.

## Estructura del repositorio

```text
.
├── data/
│   └── breast_cancer.csv
├── evidence/
├── notebooks/
│   └── TC5061_Notebook_Baseline_RuanDaniel.ipynb
├── report/
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Requisitos

- Python 3.14
- MLflow 3.16.1
- Dependencias especificadas en requirements.txt

## Instalación de dependencias

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
```

## Ayuda del script

```powershell
python src/train.py --help
```

## Ejecución con valores predeterminados

```powershell
python src/train.py
```

## Ejecución parametrizada

```powershell
python src/train.py --C 0.1 --solver lbfgs --max-iter 1000 --tol 0.0001 --random-state 42 --run-name C01_lbfgs
```

## Parámetros principales

- `--C`: inverso de la fuerza de regularización.
- `--solver`: algoritmo de optimización.
- `--max-iter`: máximo de iteraciones.
- `--tol`: tolerancia de convergencia.
- `--random-state`: semilla aleatoria.
- `--test-size`: proporción reservada para prueba.
- `--run-name`: nombre descriptivo de la corrida.
- `--experiment-name`: nombre del experimento en MLflow.

## Interfaz de MLflow

Después de ejecutar las corridas, iniciar el servidor desde la raíz del repositorio:

```powershell
mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador:

```text
http://127.0.0.1:5000
```

## Experimento

Todas las corridas se agrupan en:

```text
TC5061_Experimentacion_LogisticRegression
```

## Métricas registradas

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

## Artefactos registrados

Cada corrida registra:

- Matriz de confusión en formato PNG.
- Métricas en formato JSON.
- Predicciones en formato CSV.
- Pipeline entrenado con StandardScaler y LogisticRegression.

## Reproducibilidad

La semilla se controla mediante `--random-state`. La división de los datos usa estratificación y la misma semilla. Dos ejecuciones con argumentos idénticos deben generar las mismas métricas.

Las corridas `reproducibility_A` y `reproducibility_B` se ejecutaron con la misma configuración y produjeron resultados idénticos.
