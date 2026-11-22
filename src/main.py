
import os
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie konfiguracji z pliku .env
load_dotenv()

# Pobranie parametrów konfiguracyjnych
data_path = os.getenv("DATA_PATH", "data/iris.csv")
test_size = float(os.getenv("TEST_SIZE", "0.2"))
random_seed = int(os.getenv("RANDOM_SEED", "42"))

# Wczytanie danych z pliku CSV
df = pd.read_csv(data_path)

# Przygotowanie cech i zmiennej docelowej
X = df.drop(columns=["target"])
y = df["target"]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_seed,
    stratify=y
)

# Utworzenie i trenowanie modelu
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["setosa", "versicolor", "virginica"]
)

# Wyświetlenie wyników
print("PARAMETRY KONFIGURACYJNE:")
print("Ścieżka do danych:", data_path)
print("Wielkość zbioru testowego:", test_size)
print("Ziarno losowości:", random_seed)

print("\nDOKŁADNOŚĆ MODELU:")
print(accuracy)

print("\nRAPORT KLASYFIKACJI:")
print(report)

# Zapis wyników do pliku
with open("wyniki.txt", "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy}\n\n")
    file.write(report)

print("\nWyniki zapisano do pliku wyniki.txt")
