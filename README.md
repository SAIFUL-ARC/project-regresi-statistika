# Klasifikasi Tingkat Risiko Kekerasan Anak di Sulawesi Menggunakan Support Vector Machine (SVM)
 
Repositori ini berisi implementasi proyek komputasi Machine Learning untuk menganalisis dan mengklasifikasikan tingkat risiko kasus kekerasan terhadap anak di wilayah Sulawesi berdasarkan data tahun 2023. Proyek ini dikembangkan menggunakan bahasa pemrograman Python untuk mengeksplorasi penerapan algoritma klasifikasi terhadap data sosial-demografi berbasis provinsi.Proyek ini disusun sebagai bagian dari pemenuhan tugas mata kuliah Statistik dan Probabilitas.

## Latar Belakang 
 
Kekerasan terhadap anak merupakan permasalahan sosial yang tersebar secara tidak merata di berbagai wilayah Indonesia, termasuk di Sulawesi. Perbedaan kondisi geografis, sosial-ekonomi, dan budaya antar provinsi menciptakan pola kekerasan yang kompleks dan non-linear, sehingga pendekatan klasifikasi berbasis statistik tradisional sering kali tidak mampu menangkap pola tersebut secara menyeluruh.
 
Oleh karena itu, proyek ini mengimplementasikan algoritma **Support Vector Machine (SVM)** dengan kernel RBF (*Radial Basis Function*) yang mampu memetakan hubungan non-linear antara jenis-jenis kekerasan dengan kategori risiko wilayah. Model dilatih menggunakan data observasi multivariat yang mencakup tujuh dimensi jenis kekerasan anak di seluruh provinsi di Sulawesi.

## Sumber Data & Cakupan Wilayah
 
Data yang digunakan bersumber dari dataset **kekerasan anak tingkat provinsi di Sulawesi tahun 2023**, yang mencakup enam provinsi utama:
 
- Sulawesi Selatan
- Sulawesi Tengah
- Sulawesi Tenggara
- Sulawesi Utara
- Sulawesi Barat
- Gorontalo

## Alur Kerja Machine Learning
 
Sistem klasifikasi dalam repositori ini dibangun melalui tahapan pemrosesan yang terstruktur:
 
1. **Pemuatan & Inspeksi Data:** Membaca dataset CSV dan memeriksa dimensi serta integritas data.
2. **Seleksi Fitur & Target:** Mendefinisikan tujuh jenis kekerasan sebagai matriks fitur prediktor (X) dan kategori_risiko sebagai target (y).
3. **Encoding Label:** Mengonversi label kategori risiko bertipe string menjadi representasi numerik menggunakan LabelEncoder.
4. **Standardisasi Fitur:** Menggunakan StandardScaler untuk menyamakan skala antar fitur numerik agar bobot antar variabel kekerasan proporsional dalam ruang fitur SVM.
5. **Pembagian Dataset:** Membagi data menjadi Training Set (75%) dan Testing Set (25%) dengan stratifikasi kelas untuk menjaga proporsi label.
6. **Pelatihan Model SVM:** Melatih model SVC dengan kernel RBF, C=10, dan gamma='scale' untuk menangkap batas keputusan non-linear antar kelas risiko.
7. **Evaluasi Metrik:** Mengukur performa model menggunakan Akurasi, Laporan Klasifikasi (precision, recall, F1-score), dan Stratified K-Fold Cross-Validation (5 lipatan) untuk validasi stabilitas model.
8. **Visualisasi Data:** Menggunakan Matplotlib dan Seaborn untuk memproyeksikan proporsi jenis kekerasan per provinsi dalam bentuk Stacked Bar Chart serta performa klasifikasi dalam bentuk Confusion Matrix.
