import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("==================================================")
print(" STUDI KASUS: PREDIKSI KETERLAMBATAN INVOICE")
print(" Berdasarkan Materi Praktisi Mengajar BRIN")
print("==================================================\n")

# ==========================================
# 1. SETUP & GENERATE DATASET SINTETIK
# ==========================================
np.random.seed(42)
n_records = 5000

customer_ids = [f"CUST-{str(i).zfill(3)}" for i in range(1, 121)]
payment_methods = ["Transfer Bank", "Giro", "Tempo 30 Hari", "Cash"]

data = {
    'invoice_id': [f"INV-{str(i).zfill(5)}" for i in range(1, n_records + 1)],
    'customer_id': np.random.choice(customer_ids, n_records),
    'invoice_date': pd.date_range(start='2024-07-01', periods=n_records, freq='h').date,
    'total_amount': np.random.uniform(500000, 15000000, n_records),
    'discount_given': np.random.choice([0, 50000, 100000, 500000], n_records, p=[0.6, 0.2, 0.15, 0.05]),
    'payment_method': np.random.choice(payment_methods, n_records, p=[0.5, 0.2, 0.2, 0.1])
}

df = pd.DataFrame(data)
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df['due_date'] = df['invoice_date'] + pd.Timedelta(days=30)

# Simulasi tanggal pembayaran
delay_days = np.random.choice([0, 5, 10, 35, 50, 90], n_records, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
df['payment_date'] = df['due_date'] + pd.to_timedelta(delay_days, unit='D')

print(f"[Info] Dataset berhasil dibuat dengan total baris: {len(df)}")
print(df.head(), "\n")

# ==========================================
# 2. DATA CLEANING & VALIDATION
# ==========================================
print("--- [Langkah 2] Melakukan Data Cleaning ---")
initial_len = len(df)

# A. Menghapus anomali bisnis (tanggal bayar sebelum tanggal invoice)
df = df[df['payment_date'] >= df['invoice_date']]
print(f"Baris setelah membersihkan anomali tanggal: {len(df)} (Dihapus: {initial_len - len(df)})")

# B. Menangani Missing Value
df = df.dropna()
print("Data bersih dan siap diproses.\n")

# ==========================================
# 3. FEATURE ENGINEERING & TARGET CREATION
# ==========================================
print("--- [Langkah 3] Feature Engineering ---")
# Target: Apakah invoice terlambat > 30 hari dari due date? (1 = Late, 0 = Not Late)
df['days_overdue'] = (df['payment_date'] - df['due_date']).dt.days
df['is_late'] = np.where(df['days_overdue'] > 30, 1, 0)

# Mengubah due_date menjadi format ordinal agar bisa dibaca model Machine Learning
df['due_date_ordinal'] = pd.to_datetime(df['due_date']).apply(lambda x: x.toordinal())

print("Distribusi Target (is_late):")
print(df['is_late'].value_counts(normalize=True) * 100, "\n")

# ==========================================
# 4. MENGHINDARI DATA LEAKAGE
# ==========================================
# Kolom masa depan (data leakage) dibuang dari fitur pelatihan
features_to_drop = [
    'invoice_id', 'customer_id', 'invoice_date', 
    'due_date', 'payment_date', 'days_overdue', 'is_late'
]

X = df.drop(columns=features_to_drop)
# One-hot encoding untuk kategori teks
X = pd.get_dummies(X, columns=['payment_method'], drop_first=True)
y = df['is_late']

print(f"Fitur aktif untuk model: {list(X.columns)}\n")

# ==========================================
# 5. MODELING & TRAINING (Random Forest)
# ==========================================
print("--- [Langkah 4] Melatih Model Machine Learning ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model Random Forest berhasil dilatih!\n")

# ==========================================
# 6. EVALUASI MODEL

# ==========================================
print("--- [Langkah 5] Evaluasi Model ---")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==================================================")
print(" PROSES SELESAI")
print("==================================================")