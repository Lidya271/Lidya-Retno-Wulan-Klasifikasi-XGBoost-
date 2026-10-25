import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

st.set_page_config(page_title="Kolkata Weather - XGBoost", page_icon="🌦️", layout="wide")
st.title("🌦️ Klasifikasi Hujan di Kolkata dengan XGBoost")
st.write(
    "Dashboard klasifikasi **Hujan/Tidak Hujan** menggunakan dataset cuaca Kolkata. "
    "Target dibuat dari `precipitation_sum`: hujan jika nilainya > 0 mm, selain itu tidak hujan."
)

DEFAULT_FILE = "kolkata_weather_jan_june_2026.csv"

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv(DEFAULT_FILE)
        except FileNotFoundError:
            return None
    return data

uploaded = st.sidebar.file_uploader("Unggah file CSV dataset", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.warning(
        f"File `{DEFAULT_FILE}` belum ditemukan. Letakkan file CSV di folder yang sama dengan app.py "
        "atau unggah file melalui sidebar."
    )
    st.stop()

required = {"precipitation_sum"}
if not required.issubset(df.columns):
    st.error("Dataset harus memiliki kolom `precipitation_sum` untuk membentuk target hujan/tidak hujan.")
    st.stop()

# Rapikan nama kolom dan data dasar
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

# Bentuk target dari curah hujan. Kolom target dan kolom yang berpotensi membocorkan target
# tidak digunakan sebagai fitur.
df["Target"] = np.where(pd.to_numeric(df["precipitation_sum"], errors="coerce") > 0, 1, 0)
df["Target_Label"] = df["Target"].map({0: "Tidak Hujan", 1: "Hujan"})

# Gunakan fitur yang tersedia sebelum/selama kondisi cuaca dan tidak menggunakan
# kolom curah hujan, kode cuaca, jam hujan, atau kolom waktu matahari.
numeric_candidates = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "sunshine_duration",
]
feature_cols = [c for c in numeric_candidates if c in df.columns]

# Tambahkan fitur kalender yang dapat diketahui tanpa melihat hasil hujan.
if "time" in df.columns:
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = parsed_time.dt.month
    df["day_of_year"] = parsed_time.dt.dayofyear
    feature_cols += ["month", "day_of_year"]

if len(feature_cols) < 2:
    st.error("Dataset tidak memiliki cukup kolom fitur numerik yang didukung untuk membangun model.")
    st.stop()

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

model_data = df[feature_cols + ["Target"]].replace([np.inf, -np.inf], np.nan)
model_data = model_data.dropna(subset=["Target"])
# Simpan data penelitian ke CSV
model_data.to_csv(
    "Nama_Test1_KlasifikasiXGBoost.csv",
    index=False
)
print("Data penelitian berhasil disimpan!")
X = model_data[feature_cols]
y = model_data["Target"].astype(int)

if y.nunique() < 2:
    st.error("Target hanya memiliki satu kelas. Model klasifikasi membutuhkan kelas Hujan dan Tidak Hujan.")
    st.stop()

# Pisahkan data terlebih dahulu. Imputer dipasang di dalam pipeline sehingga median
# dihitung hanya dari data latih, bukan dari keseluruhan dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

@st.cache_resource
def train_model(X_train, y_train):
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
            objective="binary:logistic",
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

# Ringkasan data
st.header("1. Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah baris setelah duplikat dihapus", f"{len(df):,}")
c2.metric("Jumlah fitur model", len(feature_cols))
c3.metric("Data latih", f"{len(X_train):,} (80%)")
c4.metric("Data uji", f"{len(X_test):,} (20%)")

if "time" in df.columns:
    dates = pd.to_datetime(df["time"], errors="coerce")
    st.caption(f"Rentang tanggal pada file: {dates.min().date()} sampai {dates.max().date()}.")

st.subheader("Struktur dan statistik dataset")
with st.expander("Lihat data awal"):
    st.dataframe(df.head(10), use_container_width=True)
with st.expander("Tipe data dan missing values"):
    info = pd.DataFrame({
        "Tipe Data": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Persentase Missing (%)": (df.isna().mean() * 100).round(2)
    })
    st.dataframe(info, use_container_width=True)
with st.expander("Statistik deskriptif"):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

st.subheader("Proporsi kelas target")
class_counts = df["Target_Label"].value_counts().reindex(["Tidak Hujan", "Hujan"]).fillna(0)
fig, ax = plt.subplots()
ax.bar(class_counts.index, class_counts.values)
ax.set_ylabel("Jumlah hari")
ax.set_title("Distribusi Target Hujan/Tidak Hujan")
st.pyplot(fig)
plt.close(fig)
st.write("Proporsi kelas (%):")
st.dataframe((class_counts / class_counts.sum() * 100).round(2).rename("Persentase").to_frame())

st.header("2. Eksplorasi Data (EDA)")
selected_feature = st.selectbox("Pilih fitur untuk melihat distribusinya", feature_cols)
fig, ax = plt.subplots()
sns.histplot(data=df, x=selected_feature, hue="Target_Label", bins=30, kde=True, ax=ax)
ax.set_title(f"Distribusi {selected_feature} berdasarkan kelas")
st.pyplot(fig)
plt.close(fig)

st.subheader("Korelasi antarfitur")
corr = model_data[feature_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Heatmap Korelasi Fitur")
st.pyplot(fig)
plt.close(fig)

st.header("3. Pra-pemrosesan dan Pencegahan Data Leakage")
st.markdown(
    f"""
- Target: `Target = 1` jika `precipitation_sum > 0`, dan `Target = 0` jika curah hujan 0 mm.
- Fitur model: `{', '.join(feature_cols)}`.
- Kolom `precipitation_sum`, `rain_sum`, `precipitation_hours`, dan `weather_code` tidak dijadikan fitur karena dapat secara langsung mengungkapkan kondisi/target hujan.
- Kolom `sunrise`, `sunset`, `city`, `latitude`, dan `longitude` tidak digunakan sebagai fitur model ini.
- Data dibagi secara stratified: 80% data latih dan 20% data uji, dengan `random_state=42`.
- Missing values pada fitur diisi menggunakan median melalui `Pipeline`. Median dipelajari dari data latih saja untuk menghindari kebocoran informasi.
- XGBoost tidak memerlukan standarisasi fitur untuk dapat bekerja.
"""
)
st.caption(
    "Catatan metodologis: model ini mengklasifikasikan kondisi harian berdasarkan fitur cuaca harian yang tersedia, "
    "bukan prakiraan hujan beberapa hari ke depan. Untuk klaim prediksi masa depan, gunakan fitur yang benar-benar "
    "tersedia sebelum waktu prediksi dan evaluasi berbasis urutan waktu."
)

st.header("4. Hasil Model XGBoost Classifier")
st.write("Parameter utama: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`.")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.3f}")
m2.metric("Precision", f"{precision:.3f}")
m3.metric("Recall", f"{recall:.3f}")
m4.metric("F1-Score", f"{f1:.3f}")
m5.metric("ROC-AUC", f"{roc_auc:.3f}")

metrics_df = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Nilai": [accuracy, precision, recall, f1, roc_auc]
})
st.subheader("Tabel metrik evaluasi")
st.dataframe(metrics_df.style.format({"Nilai": "{:.4f}"}), use_container_width=True)

st.header("5. Analisis Hasil Klasifikasi")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
    xticklabels=["Prediksi Tidak Hujan", "Prediksi Hujan"],
    yticklabels=["Aktual Tidak Hujan", "Aktual Hujan"]
)
ax.set_xlabel("Kelas Prediksi")
ax.set_ylabel("Kelas Aktual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)
plt.close(fig)
tn, fp, fn, tp = cm.ravel()
st.markdown(
    f"- **True Negative (TN): {tn}** — hari tidak hujan yang diprediksi tidak hujan.\n"
    f"- **False Positive (FP): {fp}** — hari tidak hujan yang salah diprediksi hujan.\n"
    f"- **False Negative (FN): {fn}** — hari hujan yang salah diprediksi tidak hujan.\n"
    f"- **True Positive (TP): {tp}** — hari hujan yang diprediksi hujan."
)

fig, ax = plt.subplots()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Garis acak")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)
plt.close(fig)
st.write(
    f"ROC-AUC sebesar **{roc_auc:.3f}** mengukur kemampuan model membedakan kelas hujan dan tidak hujan "
    "pada berbagai ambang probabilitas. Nilai mendekati 1 lebih baik, sedangkan sekitar 0,5 setara tebakan acak."
)

st.subheader("Feature importance")
xgb_model = model.named_steps["model"]
importance = pd.DataFrame({
    "Fitur": feature_cols,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=importance, x="Importance", y="Fitur", ax=ax)
ax.set_title("Feature Importance XGBoost")
st.pyplot(fig)
plt.close(fig)
st.dataframe(importance, use_container_width=True)
st.caption(
    "Feature importance menunjukkan kontribusi relatif fitur terhadap pemisahan dalam model; "
    "ini bukan bukti sebab-akibat."
)

with st.expander("Laporan klasifikasi lengkap"):
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Tidak Hujan", "Hujan"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.header("6. Coba Prediksi Manual")
st.write("Masukkan nilai fitur di bawah ini. Form ini menggunakan fitur yang sama dengan model.")
with st.form("manual_prediction"):
    input_values = {}
    default_values = X.median(numeric_only=True)
    cols = st.columns(2)
    for i, feature in enumerate(feature_cols):
        value = float(default_values.get(feature, 0))
        input_values[feature] = cols[i % 2].number_input(
            feature, value=value, format="%.3f", key=f"input_{feature}"
        )
    submitted = st.form_submit_button("Prediksi kondisi")
if submitted:
    manual_df = pd.DataFrame([input_values], columns=feature_cols)
    pred = int(model.predict(manual_df)[0])
    prob = float(model.predict_proba(manual_df)[0, 1])
    if pred == 1:
        st.success(f"Prediksi model: **HUJAN** (probabilitas kelas hujan: {prob:.1%})")
    else:
        st.info(f"Prediksi model: **TIDAK HUJAN** (probabilitas kelas hujan: {prob:.1%})")
    st.caption("Hasil ini merupakan keluaran model pembelajaran mesin, bukan jaminan kondisi cuaca sebenarnya.")

st.header("7. Kesimpulan Otomatis")
st.write(
    f"Pada data uji sebanyak {len(X_test)} baris, model memperoleh Accuracy {accuracy:.3f}, "
    f"Precision {precision:.3f}, Recall {recall:.3f}, F1-Score {f1:.3f}, dan ROC-AUC {roc_auc:.3f}. "
    "Gunakan confusion matrix untuk melihat jenis kesalahan yang paling sering terjadi dan feature importance "
    "untuk mengetahui fitur yang paling banyak digunakan model. Nilai tersebut adalah hasil evaluasi pada pembagian "
    "data ini; evaluasi tambahan dengan data dari periode berbeda diperlukan untuk menilai kemampuan generalisasi."
)
