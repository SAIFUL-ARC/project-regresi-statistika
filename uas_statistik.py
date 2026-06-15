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