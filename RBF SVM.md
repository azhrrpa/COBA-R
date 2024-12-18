Berikut adalah tutorial berbasis kode untuk analisis dan prediksi serangan jantung menggunakan **SVM (Support Vector Machine)** dengan R. Tutorial ini dapat digunakan sebagai panduan pemula untuk memahami analisis data, preprocessing, dan implementasi SVM.

### Tutorial: Analisis Data dan Prediksi Serangan Jantung Menggunakan SVM Kernel RBF di R

#### **1. Memuat Library**
Kita membutuhkan beberapa library penting:
```R
library(ggplot2)
library(caret)
library(e1071)
library(pROC)
library(gridExtra)
library(reshape2)
```

#### **2. Memuat Dataset**
Dataset dapat diakses dari file CSV:
```R
df <- read.csv("path/ke/dataset/heart.csv")
```

#### **3. Preprocessing Data**
- **Penamaan Kolom**: Menamai ulang kolom untuk memudahkan pemrosesan.
```R
colnames(df) <- c("age", "sex", "chest_pain", "blood_pressure", "cholesterol", 
                  "fasting_blood_sugar", "restecg", "max_heart_rate", "angina", 
                  "oldpeak", "slope", "n_vessels", "thall", "heart_attack")
```

- **Handling Missing Values**:
```R
cat("Jumlah nilai yang hilang:", sum(is.na(df)), "\n")
```

- **Konversi Kolom Kategorikal**:
```R
df$sex <- as.factor(df$sex)
df$fasting_blood_sugar <- as.factor(df$fasting_blood_sugar)
df$angina <- as.factor(df$angina)
df$n_vessels <- as.factor(df$n_vessels)
df$thall <- as.factor(df$thall)
df$heart_attack <- as.factor(df$heart_attack)
```

#### **4. Analisis Korelasi**
Korelasi antar fitur numerik divisualisasikan dengan heatmap:
```R
numeric_cols <- sapply(df, is.numeric)
df_corr <- cor(df[, numeric_cols], use = "complete.obs")
melted_corr <- melt(df_corr)
ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal()
```

#### **5. Standarisasi dan Identifikasi Outlier**
- **Standarisasi Data Numerik**:
```R
df[numeric_cols] <- scale(df[numeric_cols])
```

- **Deteksi Outlier**:
```R
z_scores <- scale(df[, numeric_cols])
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)
cat("Jumlah outlier berdasarkan Z-score:", length(outliers), "\n")
```

- **Menghapus Outlier**:
```R
df_no_outliers <- df[apply(z_scores, 1, function(x) all(abs(x) <= 3)), ]
```

#### **6. Membagi Data (Training dan Testing)**
Split dataset menjadi 80% training dan 20% testing:
```R
set.seed(42)
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```

#### **7. Melatih Model SVM**
Melatih SVM dengan kernel **RBF** menggunakan Cross-Validation:
```R
ctrl <- trainControl(method = "cv", number = 5)
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmRadial",
                   trControl = ctrl,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.05, 0.1)))
```

#### **8. Evaluasi Model**
- **Matriks Kebingungan**:
```R
pred <- predict(svm_model, df_test)
conf_matrix <- confusionMatrix(pred, df_test$heart_attack)
print(conf_matrix)
```

- **Metrik Tambahan (Precision, Recall, F1-Score)**:
```R
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1 <- 2 * (precision * recall) / (precision + recall)
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1, "\n")
```

#### **9. Visualisasi**
- **ROC Curve**:
```R
roc_obj <- roc(as.numeric(df_test$heart_attack), as.numeric(pred))
plot(roc_obj, main = "ROC Curve", col = "blue")
auc_val <- auc(roc_obj)
cat("AUC:", auc_val, "\n")
```

- **Hyperplane SVM (Visualisasi 2D)**:
Menggunakan kombinasi fitur tertentu untuk memvisualisasikan hyperplane:
```R
subset_df <- df[, c("age", "blood_pressure", "heart_attack")]
svm_model <- svm(heart_attack ~ age + blood_pressure, data = subset_df, kernel = "radial", cost = 1)
plot(svm_model, subset_df, age ~ blood_pressure)
```

#### **10. Kesimpulan**
Hasil evaluasi model menunjukkan akurasi, precision, recall, F1-Score, dan AUC dari model SVM yang digunakan. Tutorial ini mengajarkan langkah-langkah preprocessing, pelatihan model, dan evaluasi performa.

> **Catatan:** Pastikan dataset yang digunakan sesuai dengan kode di atas untuk menghindari error.
