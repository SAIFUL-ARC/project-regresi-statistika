# =============================================================================
#  TINGKAT KASUS KEKERASAN ANAK DI SULAWESI 
#  Metode: Support Vector Machine (SVM/SVC) — Klasifikasi Kategori Risiko
#  Dataset: kekerasan_anak_sulawesi_2023.csv
# =============================================================================

# =============================================================================
# 1. IMPOR LIBRARY YANG DIBUTUHKAN & PENYESUAIAN TAMPILAN
# =============================================================================
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold"})

# =============================================================================
# 2. MEMUAT DATA
# =============================================================================
FILE_PATH =  "C:/project/project/kekerasan_anak_sulawesi_2023.csv" 

df = pd.read_csv(FILE_PATH)
print(f"[INFO] Total data: {len(df)} record")

# =============================================================================
# 3. MENENTUKAN FITUR DAN TARGET
# =============================================================================
FITUR = [
    "kekerasan_fisik", "kekerasan_psikis", "kekerasan_seksual",
    "eksploitasi", "tppo", "penelantaran", "kekerasan_lainnya"
]
TARGET = "kategori_risiko"

X_Raw = df[FITUR].values
y_Raw = df[TARGET].values

# =============================================================================
# 4.MEMPROSES DATA
# =============================================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y_Raw)

penskala = StandardScaler()
X = penskala.fit_transform(X_Raw)

print(f"[INFO] Bentuk fitur X: {X.shape}")
print(f"[INFO] Kelas target: {list(encoder.classes_)}")

X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"[INFO] Data latih: {len(X_latih)} record")
print(f"[INFO] Data uji: {len(X_uji)} record")

# =============================================================================
# 5. MEMBUAT MODEL SVM
# =============================================================================

model_svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
model_svm.fit(X_latih, y_latih)

y_pred = model_svm.predict(X_uji)

akurasi = accuracy_score(y_uji, y_pred)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nilai_cv = cross_val_score(model_svm, X, y, cv=cv, scoring="accuracy")

print("\n" + "=" * 56)
print("  HASIL EVALUASI MODEL SVM - SULAWESI")
print("=" * 56)
print(f"  Akurasi Data Uji      : {akurasi * 100:.2f}%")
print(f"  Rata-rata CV (5 lipat) : {nilai_cv.mean() * 100:.2f}%")
print(f"  Standar Deviasi CV    : {nilai_cv.std() * 100:.2f}%")
print("-" * 56)
print("\nLAPORAN KLASIFIKASI:")
print(classification_report(y_uji, y_pred, target_names=encoder.classes_))
print("=" * 56)