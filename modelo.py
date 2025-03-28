import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("fake_news_spanish_clean.csv")

# Load precomputed TF-IDF matrix
with open("tfidf_vectorized.pkl", "rb") as f:
    X_tfidf = pickle.load(f)

# Define target variable
y = df['Label']  # Ensure it's the correct label column (1 = Fake, 0 = Real)

# Añadir columnas de longitud
X_extra = df[['Titulo_len', 'Descripcion_len']].values

# Combinar la matriz TF-IDF con las columnas adicionales
from scipy.sparse import hstack
X_combined = hstack([X_tfidf, X_extra])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Definir el modelo base de XGBoost con parámetros especificados
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=300,        # Número de árboles
    max_depth=10,            # Profundidad máxima de los árboles
    learning_rate=0.1,      # Tasa de aprendizaje
    subsample=1.0,          # Porcentaje de muestras usadas por árbol
    colsample_bytree=1.0    # Porcentaje de características usadas por árbol
)

# Entrenar el modelo
xgb.fit(X_train, y_train)

# Predecir con el modelo entrenado
y_pred = xgb.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado como archivo .pkl
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("Modelo XGBoost guardado exitosamente como 'xgboost_model.pkl'")

# Matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - XGBoost')
plt.show()
