* **Nama:** Anisa Julianti
* **NPM:** G1A023052
* **Program Studi:** Informatika, Universitas Bengkulu

# Studi Kasus: Prediksi Keterlambatan Pembayaran Invoice (*Late Payment Prediction*)

Studi kasus ini dikembangkan berdasarkan materi **Informatics Practitioner Lecture Series** dengan narasumber **Bapak Andria Arisal** (Peneliti Pusat Riset Sains Data dan Informasi Badan Riset dan Inovasi Nasional / BRIN) dengan tema **"Make Sense of Data with Analysis and AI"**.

---

## 1. Latar Belakang & Konteks Bisnis
Sebuah perusahaan distribusi fiktif bernama **PT Maju Bersama** menghadapi teka-teki finansial:
* Berdasarkan laporan 2 tahun berjalan, **Sales Revenue** (pendapatan penjualan) mengalami kenaikan.
* Namun, faktanya **Net Profit** (keuntungan bersih) justru turun 14%.
* Piutang dagang (*Accounts Receivable*) melonjak hingga 300% dan **Operating Cash Flow** berkurang drastis.

**Pertanyaan Bisnis:** 
* Apakah kondisi keuangan PT Maju Bersama ini sehat?
* *Invoice* atau pelanggan mana yang berisiko tinggi mengalami keterlambatan pembayaran (*late payment*) di masa depan agar manajemen dapat mengambil tindakan preventif (*prescriptive action*)?

---

## 2. Alur Proses Analisis (CRISP-DM)
Untuk menyelesaikan masalah bisnis di atas, proses analitik dijalankan secara terstruktur menggunakan metodologi **CRISP-DM**:
1. **Business Understanding:** Menyamakan persepsi terminologi bisnis (misal: definisi pendapatan bersih vs kotor) dan menerjemahkannya ke dalam pertanyaan analitik.
2. **Data Understanding & Preparation:** 
   * Mengelola *granularities* data (tingkat per produk, *invoice line*, *invoice*, hingga *customer*).
   * Membersihkan anomali data (seperti tanggal pembayaran yang mendahului tanggal penerbitan *invoice*).
   * **Menghindari Data Leakage:** Membuang fitur/kolom yang mencatat informasi masa depan (seperti jumlah hari keterlambatan aktual) agar model tidak bias.
3. **Modeling:** Menerapkan algoritma *Machine Learning* klasifikasi (*Random Forest Classifier*) untuk memprediksi probabilitas keterlambatan pembayaran.
4. **Evaluation:** Mengukur performa model menggunakan *Classification Report*, *ROC-AUC Score*, dan *Confusion Matrix*.
5. **Deployment & Decision (Human-in-the-Loop):** Model dan AI berfungsi sebagai pemberi wawasan (*insight*), sementara keputusan strategis akhir tetap dikendalikan oleh manusia.

---

## 3. Implementasi Kode Studi Kasus (`analisis_pt_maju.py`)

Berikut adalah ringkasan tahapan implementasi teknis yang digunakan dalam skrip Python:

```python
# 1. Generate Dataset Sintetik PT Maju Bersama & Data Cleaning
# 2. Feature Engineering & Pembuatan Target Biner (is_late > 30 hari)
# 3. Pencegahan Data Leakage (Drop kolom masa depan)
# 4. Training Model Machine Learning (Random Forest)
# 5. Evaluasi Performa Model (Classification Report & ROC-AUC)
