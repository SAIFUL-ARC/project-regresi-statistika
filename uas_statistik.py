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

# =============================================================================
# 6. VISUALISASI - GRAFIK BATANG TUMPUK PER PROVINSI
# =============================================================================
print("\n[INFO] Membuat visualisasi...")

fig, ax = plt.subplots(figsize=(14, 7))

df_provinsi = df.groupby("nama_provinsi")[FITUR].sum()
df_provinsi_persen = df_provinsi.div(df_provinsi.sum(axis=1), axis=0) * 100

WARNA = ["#2171b5", "#fd8d3c", "#31a354", "#e31a1c", "#c994c7", "#a65628", "#6a51a3"]

df_provinsi_persen.plot(
    kind="bar", stacked=True, ax=ax,
    color=WARNA[:len(FITUR)],
    edgecolor="white", linewidth=0.6, width=0.7
)

ax.set_title("Proporsi Jenis Kekerasan Anak per Provinsi di Sulawesi (%)",
              fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Provinsi", fontsize=12, labelpad=10)
ax.set_ylabel("Persentase (%)", fontsize=12, labelpad=10)

ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_locator(plt.MultipleLocator(20))

ax.tick_params(axis="x", labelrotation=25, labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    title="Jenis Kekerasan",
    bbox_to_anchor=(1.01, 1), loc="upper left",
    fontsize=9, title_fontsize=10, frameon=False
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("kekerasan_anak_sulawesi.png", bbox_inches="tight", dpi=150, facecolor="white")
plt.show()

print("[INFO] Plot tersimpan → kekerasan_anak_sulawesi.png")

# =============================================================================
# 7. VISUALISASI - CONFUSION MATRIX
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(7, 5))

cm = confusion_matrix(y_uji, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(ax=ax2, cmap="Blues", values_format="d")

ax2.set_title("Confusion Matrix - Klasifikasi Risiko Kekerasan Anak", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix_sulawesi.png", bbox_inches="tight", dpi=150, facecolor="white")
plt.show()

print("[INFO] Confusion Matrix tersimpan → confusion_matrix_sulawesi.png")

# =============================================================================
# 10. RINGKASAN STATISTIK
# =============================================================================
print("\n" + "=" * 56)
print("  RINGKASAN STATISTIK KASUS KEKERASAN ANAK")
print("=" * 56)

for provinsi in df['nama_provinsi'].unique():
    data_prov = df[df['nama_provinsi'] == provinsi]
    total = data_prov['total_kasus'].sum()
    rata_rata = data_prov['total_kasus'].mean()
    print(f"  {provinsi:20s} : Total={total:4d} kasus, Rata-rata={rata_rata:5.1f}")

print("=" * 56)
print("[INFO] Analisis selesai!")