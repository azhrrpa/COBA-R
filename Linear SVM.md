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
**penjelasan:** 
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

---

### **5. Data Standardization and Outlier Detection**
Semua fitur numerik diskalakan menggunakan `scale()` untuk meningkatkan kinerja model SVM. Z-score digunakan untuk mendeteksi outlier.

```r
df[numeric_cols] <- scale(df[numeric_cols])
z_scores <- scale(df[, numeric_cols])
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)
df_no_outliers <- df[apply(z_scores, 1, function(x) all(abs(x) <= 3)), ]
```

---

### **6. Dataset Splitting**
Dataset dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian menggunakan fungsi `createDataPartition`.

```r
set.seed(42)
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```

---

### **7. Model Training (SVM)**
Model SVM dilatih menggunakan kernel linear dan cross-validation (K-Fold). Hyperparameter C disesuaikan untuk mencari performa terbaik.

```r
ctrl <- trainControl(method = "cv", number = 5)
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmLinear", 
                   trControl = ctrl,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10)))
```

---

### **8. Evaluation Metrics**
Model dievaluasi menggunakan data pengujian. Metrik seperti akurasi, precision, recall, F1-score, dan AUC dihitung dan ditampilkan.

```r
pred <- predict(svm_model, df_test)
conf_matrix <- confusionMatrix(pred, df_test$heart_attack)
roc_obj <- roc(as.numeric(df_test$heart_attack), as.numeric(pred))
auc_val <- auc(roc_obj)
```

---

### **9. Visualizing Hyperplanes**
Plot hyperplane SVM dibuat menggunakan kombinasi 2 fitur dengan `plot.svm`. Ini membantu memahami klasifikasi dalam ruang 2D.

```r
subset_df <- df[, c("age", "blood_pressure", "heart_attack")]
svm_model <- svm(heart_attack ~ age + blood_pressure, data = subset_df, kernel = "linear", cost = 1)
plot(svm_model, subset_df, age ~ blood_pressure)
```

