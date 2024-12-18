Proyek ini bertujuan untuk menganalisis data kesehatan menggunakan algoritma Support Vector Machine (SVM). Dalam tutorial ini, kita akan melakukan preprocessing data, membangun model prediksi untuk mendeteksi risiko serangan jantung, dan mengevaluasi performa model menggunakan berbagai metrik. Selain itu, tutorial ini juga mencakup langkah-langkah visualisasi data dan hyperplane SVM. Pada akhir tutorial, Anda akan belajar bagaimana mengunggah proyek ini ke GitHub untuk dokumentasi atau kolaborasi lebih lanjut.

---

### **1. Importing Required Libraries**
Kode memuat sejumlah pustaka R yang relevan untuk analisis, pemrosesan data, dan visualisasi, seperti `ggplot2` untuk plot, `caret` untuk pelatihan model machine learning, dan `e1071` untuk algoritma SVM.

```r
library(ggplot2)
library(caret)
library(e1071)
library(pROC)
library(gridExtra)
library(reshape2)
```

---

### **2. Load Dataset**
File dataset diimpor dari path lokal. Disarankan mengganti path dengan format relatif atau menempatkan dataset dalam direktori kerja agar file dapat diakses secara universal.

```r
df <-  read.csv("C:/Users/Asus/OneDrive/Documents/CCIT/TIES SEM 3/projekkkkkkkk/heart.csv")
```
**penjelasan:** Fungsi read.csv() digunakan untuk memuat data dari file CSV yang berlokasi di path "C:/Users/Asus/OneDrive/Documents/CCIT/TIES SEM 3/projekkkkkkkk/heart.csv"  Data yang diimpor akan disimpan dalam objek df, yang nantinya dapat digunakan untuk analisis lebih lanjut seperti preprocessing, pelatihan model, atau evaluasi.

---

### **3. Preprocessing**
Kolom diubah namanya agar lebih mudah digunakan, missing values diidentifikasi, dan fitur kategorikal dikonversi menjadi tipe faktor. Langkah ini memastikan dataset bersih dan siap digunakan.

```r
colnames(df) <- c("age", "sex", "chest_pain", "blood_pressure", "cholesterol", 
                  "fasting_blood_sugar", "restecg", "max_heart_rate", "angina", 
                  "oldpeak", "slope", "n_vessels", "thall", "heart_attack")
df$sex <- as.factor(df$sex)
df$fasting_blood_sugar <- as.factor(df$fasting_blood_sugar)
df$angina <- as.factor(df$angina)
df$n_vessels <- as.factor(df$n_vessels)
df$thall <- as.factor(df$thall)
df$heart_attack <- as.factor(df$heart_attack)
```
**Penjelasan**: Kode tersebut digunakan untuk mempersiapkan dataset df dengan mengganti nama kolom menjadi lebih deskriptif dan mengonversi beberapa kolom menjadi tipe data faktor. Kolom sex, fasting_blood_sugar, angina, n_vessels, thall, dan heart_attack diubah menjadi faktor karena kolom-kolom ini merepresentasikan data kategori, bukan numerik. Langkah ini penting untuk memastikan bahwa algoritma machine learning dapat memahami dan memproses data dengan benar sesuai dengan jenis variabelnya.

---

### **4. Exploratory Data Analysis (EDA)**
Melakukan analisis korelasi antar fitur numerik dan menghasilkan heatmap korelasi menggunakan `ggplot2`.

```r
numeric_cols <- sapply(df, is.numeric)
df_corr <- cor(df[, numeric_cols], use = "complete.obs")
melted_corr <- melt(df_corr)
ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Heatmap Korelasi")
```
**Penjelasan**: Kode tersebut digunakan untuk membuat heatmap korelasi yang menggambarkan hubungan antara variabel numerik dalam dataset df. Pertama, kolom numerik diidentifikasi menggunakan sapply() dengan fungsi is.numeric. Kemudian, matriks korelasi dihitung menggunakan fungsi cor() dengan hanya menggunakan data lengkap (use = "complete.obs"). Matriks tersebut dilelehkan ke dalam format panjang menggunakan fungsi melt() agar dapat digunakan dalam visualisasi. Heatmap dibuat dengan ggplot2, menggunakan geom_tile() untuk menampilkan hubungan korelasi dalam bentuk warna, di mana gradien warna menunjukkan kekuatan dan arah korelasi (biru untuk negatif, merah untuk positif, putih untuk nol). Visualisasi ini membantu memahami hubungan antarvariabel dalam dataset

Output Korelasi
![alt text](https://github.com/azhrrpa/SVM-R/blob/main/gambar/Korelasi.png?raw=true)
---

### **5. Data Standardization and Outlier Detection**
Semua fitur numerik diskalakan menggunakan `scale()` untuk meningkatkan kinerja model SVM. Z-score digunakan untuk mendeteksi outlier.

```r
df[numeric_cols] <- scale(df[numeric_cols])
z_scores <- scale(df[, numeric_cols])
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)
df_no_outliers <- df[apply(z_scores, 1, function(x) all(abs(x) <= 3)), ]
```
**Penjelasan**: Kode tersebut digunakan untuk melakukan scaling pada variabel numerik dalam dataset df agar memiliki rata-rata 0 dan standar deviasi 1, menggunakan fungsi scale(). Setelah itu, z-scores dihitung untuk mengidentifikasi outlier, yaitu data dengan nilai absolut lebih besar dari 3. Indeks outlier disimpan dalam outliers. Dataset kemudian difilter untuk menghapus semua baris yang mengandung outlier, menghasilkan dataset baru df_no_outliers yang hanya berisi data tanpa outlier. Langkah ini bertujuan untuk meningkatkan kualitas data sehingga analisis atau model yang dibangun tidak dipengaruhi oleh nilai ekstrem.

---

### **6. Dataset Splitting**
Dataset dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian menggunakan fungsi `createDataPartition`.

```r
set.seed(42)
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```
**Penjelasan**: Kode tersebut digunakan untuk membagi dataset df menjadi data latih (df_train) dan data uji (df_test) dengan proporsi 80:20, menggunakan fungsi createDataPartition(). Fungsi set.seed(42) memastikan pembagian data bersifat reproducible, sehingga hasilnya akan sama setiap kali kode dijalankan. Pembagian ini dilakukan secara stratified berdasarkan variabel target heart_attack, memastikan distribusi kelas tetap seimbang di data latih dan data uji. Data latih digunakan untuk melatih model, sementara data uji digunakan untuk mengevaluasi kinerja mode

---

### **7. Model Training (SVM)**
Model SVM dilatih menggunakan kernel linear dan cross-validation (K-Fold). Hyperparameter C disesuaikan untuk mencari performa terbaik.

```r
ctrl <- trainControl(method = "cv", number = 5)
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmLinear", 
                   trControl = ctrl,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10)))
```
**Penjelasan**: Kode tersebut digunakan untuk melatih model Support Vector Machine (SVM) dengan kernel linear menggunakan fungsi train() dari paket caret. Parameter pelatihan dikontrol oleh objek ctrl, yang menggunakan cross-validation (cv) dengan 5 lipatan untuk mengevaluasi performa model selama pelatihan. Model dilatih pada data latih (df_train), dengan variabel target heart_attack dan semua variabel lainnya sebagai fitur. Parameter C (regularisasi) dioptimalkan melalui grid search dengan nilai 0.1, 1, dan 10 yang didefinisikan dalam tuneGrid. Pendekatan ini memastikan model yang dihasilkan adalah yang paling optimal berdasarkan evaluasi cross-validation.

---

### **8. Evaluation Metrics**
Model dievaluasi menggunakan data pengujian. Metrik seperti akurasi, precision, recall, F1-score, dan AUC dihitung dan ditampilkan.

```r
pred <- predict(svm_model, df_test)
conf_matrix <- confusionMatrix(pred, df_test$heart_attack)
roc_obj <- roc(as.numeric(df_test$heart_attack), as.numeric(pred))
auc_val <- auc(roc_obj)
```
**Penjelasan**: Kode tersebut digunakan untuk mengevaluasi kinerja model SVM yang telah dilatih. Prediksi pada data uji (df_test) dilakukan menggunakan fungsi predict(), dan hasilnya disimpan dalam pred. Matriks kebingungan (confusion matrix) dihitung menggunakan confusionMatrix() untuk mengevaluasi metrik seperti akurasi, sensitivitas, dan spesifisitas. Selain itu, kurva ROC dihitung menggunakan fungsi roc(), yang membandingkan nilai aktual dan prediksi sebagai numerik, untuk mengukur kemampuan model dalam membedakan kelas. Nilai Area Under the Curve (AUC) dihitung dengan auc() untuk memberikan satu angka yang mencerminkan performa model, di mana nilai yang lebih tinggi menunjukkan performa yang lebih baik.

---

### **9. Visualizing Hyperplanes**
Plot hyperplane SVM dibuat menggunakan kombinasi 2 fitur dengan `plot.svm`. Ini membantu memahami klasifikasi dalam ruang 2D.

```r
subset_df <- df[, c("age", "blood_pressure", "heart_attack")]
svm_model <- svm(heart_attack ~ age + blood_pressure, data = subset_df, kernel = "linear", cost = 1)
plot(svm_model, subset_df, age ~ blood_pressure)
```
**Penjelasan**: 
Kode tersebut digunakan untuk membangun model Support Vector Machine (SVM) dengan kernel linear untuk memprediksi variabel target heart_attack berdasarkan dua fitur, yaitu age dan blood_pressure, yang diambil dari subset dataset df. Model dilatih menggunakan fungsi svm() dengan parameter cost = 1, yang mengatur tingkat penalti untuk kesalahan klasifikasi. Setelah model dilatih, fungsi plot() digunakan untuk memvisualisasikan hasil model SVM, dengan plot yang menggambarkan pemisahan kelas (heart_attack) berdasarkan variabel age dan blood_pressure dalam ruang dua dimensi. Visualisasi ini membantu memahami bagaimana model memisahkan dua kelas dalam data

---
