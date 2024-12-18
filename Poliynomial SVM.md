---

## **SVM dengan Kernel Polynomial**

### **Pendahuluan**
Support Vector Machine (SVM) adalah algoritma machine learning yang dapat digunakan untuk klasifikasi dan regresi. Salah satu kekuatan utama SVM terletak pada kemampuan kernel untuk memetakan data ke dimensi yang lebih tinggi, sehingga data yang tidak terpisahkan secara linear menjadi lebih mudah untuk dipisahkan.

Kernel polynomial adalah salah satu jenis kernel yang sering digunakan dalam SVM. Kernel ini memperluas kemampuan model dengan menambahkan dimensi-dimensi polinomial pada fitur asli, memungkinkan pembentukan batas keputusan yang lebih kompleks dibandingkan dengan kernel linear.

### **Penerapan Kernel Polynomial**

Dalam eksperimen ini, kernel polynomial digunakan untuk memodelkan hubungan kompleks antar fitur dataset kesehatan yang terkait dengan risiko serangan jantung.

#### **Langkah-Langkah**
1. **Standarisasi Data**  
   Semua fitur numerik distandarisasi untuk memastikan bahwa skala fitur tidak memengaruhi performa model.

2. **Penghapusan Outliers**  
   Data outlier diidentifikasi menggunakan Z-score, dan outlier dihapus agar model lebih stabil.

3. **Pelatihan Model**  
   Model SVM dilatih menggunakan kernel polynomial dengan hyperparameter berikut:
   - **C**: Parameter regulasi yang mengontrol trade-off antara margin maksimum dan kesalahan klasifikasi.
   - **Degree**: Derajat polinomial yang menentukan kompleksitas fungsi kernel.
   - **Scale**: Faktor penskalaan untuk data.

4. **Evaluasi Model**  
   Model dievaluasi menggunakan data uji, menghasilkan metrik seperti akurasi, precision, recall, F1-score, dan ROC curve.

#### **Visualisasi Hasil**
- **ROC Curve**: Menampilkan performa prediksi model dengan area di bawah kurva (AUC) sebesar *x.x*. 
- **Hyperplane**: Menunjukkan pembagian kelas dalam beberapa kombinasi fitur, seperti `age` vs `blood_pressure` dan `max_heart_rate` vs `oldpeak`.

---

### **Perbedaan dengan Kernel Linear**

| Aspek                 | Kernel Linear                              | Kernel Polynomial                              |
|-----------------------|--------------------------------------------|-----------------------------------------------|
| **Pendekatan**        | Menggunakan garis lurus sebagai hyperplane | Menggunakan fungsi polinomial sebagai hyperplane |
| **Kecocokan Data**    | Cocok untuk data yang dapat dipisahkan secara linear | Cocok untuk data dengan hubungan non-linear kompleks |
| **Kompleksitas**      | Rendah                                    | Lebih tinggi karena melibatkan dimensi tambahan |
| **Hyperparameter**    | Hanya melibatkan parameter regulasi (C)    | Melibatkan derajat polinomial (degree) dan skala |
| **Contoh Visualisasi**| ![Linear Kernel](images/svm_hyperplane_linear.png) | ![Polynomial Kernel](images/svm_hyperplane_age_bp.png) |

---
Berikut penjelasan untuk setiap bagian kode yang digunakan dalam implementasi SVM dengan kernel polynomial:
---

## **1. Memuat Library yang Dibutuhkan**

```r
library(ggplot2)
library(caret)
library(e1071)
library(pROC)
library(gridExtra)
library(reshape2)
```

**Penjelasan**:  
Library yang digunakan:
- **ggplot2**: Untuk membuat visualisasi data.
- **caret**: Untuk pelatihan model machine learning dan cross-validation.
- **e1071**: Untuk algoritma SVM.
- **pROC**: Untuk menghitung dan memvisualisasikan ROC curve.
- **gridExtra**: Untuk mengatur tata letak beberapa plot.
- **reshape2**: Untuk mengolah data agar sesuai dengan format input visualisasi.

---

## **2. Memuat Dataset**

```r
df <- read.csv("C:/Users/Asus/OneDrive/Documents/CCIT/TIES SEM 3/projekkkkkkkk/heart.csv")
```

**Penjelasan**: digunakan untuk membaca file CSV bernama heart.csv dari jalur file yang ditentukan dan memuatnya ke dalam variabel df dalam bentuk data frame. Data frame adalah struktur data berbentuk tabel yang terdiri dari baris (observasi) dan kolom (atribut). File heart.csv kemungkinan berisi data terkait kesehatan jantung, seperti usia, tekanan darah, kadar kolesterol, atau indikator penyakit jantung. Setelah data dimuat, langkah berikutnya biasanya adalah mengeksplorasi data dengan fungsi seperti head(df) untuk melihat beberapa baris awal, atau str(df) untuk memeriksa struktur dataset. Pastikan jalur file benar, file berformat CSV valid, dan data dapat dibaca tanpa kesalahan.
---

## **3. Penamaan Ulang Kolom**

```r
colnames(df) <- c("age", "sex", "chest_pain", "blood_pressure", "cholesterol", 
                  "fasting_blood_sugar", "restecg", "max_heart_rate", "angina", 
                  "oldpeak", "slope", "n_vessels", "thall", "heart_attack")
```

**Penjelasan**: Nama kolom diubah agar lebih mudah diakses dan dipahami. Penyesuaian ini penting agar kolom sesuai dengan sintaks R.

---

## **4. Menghapus Nilai yang Hilang**

```r
cat("Jumlah nilai yang hilang:", sum(is.na(df)), "\n")
```

**Penjelasan**: Mengecek jumlah nilai yang hilang dalam dataset.

---

## **5. Konversi Kolom Kategorikal Menjadi Faktor**

```r
df$sex <- as.factor(df$sex)
df$fasting_blood_sugar <- as.factor(df$fasting_blood_sugar)
df$angina <- as.factor(df$angina)
df$n_vessels <- as.factor(df$n_vessels)
df$thall <- as.factor(df$thall)
df$heart_attack <- as.factor(df$heart_attack)
```

**Penjelasan**: Kolom dengan data kategorikal diubah menjadi tipe `factor` agar SVM dapat mengolahnya dengan benar.

---

## **6. Korelasi Antar Fitur Numerik**

```r
numeric_cols <- sapply(df, is.numeric)
df_corr <- cor(df[, numeric_cols], use = "complete.obs")
```

**Penjelasan**:  
- `is.numeric`: Memilih kolom dengan tipe data numerik.
- `cor()`: Menghitung korelasi antar fitur numerik untuk melihat hubungan antar variabel.

---

## **7. Visualisasi Heatmap Korelasi**

```r
melted_corr <- melt(df_corr)
ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Korelasi") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  labs(title = "Heatmap Korelasi")
```

**Penjelasan**:  
Heatmap digunakan untuk menunjukkan korelasi antar fitur numerik, memudahkan identifikasi hubungan antar variabel.

---

## **8. Standarisasi Fitur Numerik**

```r
df[numeric_cols] <- scale(df[numeric_cols])
```

**Penjelasan**: Semua fitur numerik dinormalisasi agar berada dalam rentang yang seragam (mean = 0, standar deviasi = 1). Hal ini penting untuk SVM, terutama kernel polynomial, agar performa model lebih optimal.

---

## **9. Identifikasi Outliers**

```r
z_scores <- scale(df[, numeric_cols])
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)
```

**Penjelasan**:  
- `scale()`: Menghitung Z-score setiap fitur.
- `which()`: Menentukan data yang memiliki Z-score lebih dari 3 sebagai outlier.

---

## **10. Membagi Dataset**

```r
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```

**Penjelasan**: Dataset dibagi menjadi:
- 80% data pelatihan.
- 20% data pengujian.

---

## **11. Pelatihan Model SVM dengan Kernel Polynomial**

```r
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmPoly", 
                   trControl = ctrl,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10), degree = c(2, 3), scale = c(0.01, 0.1)))
```

**Penjelasan**:  
Model dilatih menggunakan kernel polynomial. Hyperparameter yang dituning:
- `C`: Pengendali regulasi.
- `degree`: Derajat polinomial.
- `scale`: Faktor penskalaan fitur.

---

## **12. Evaluasi Model**

```r
conf_matrix <- confusionMatrix(pred, df_test$heart_attack)
```

**Penjelasan**: Confusion matrix digunakan untuk mengukur performa model, mencakup metrik akurasi, precision, recall, dan F1-score.

---

## **13. Visualisasi ROC Curve**

```r
roc_obj <- roc(as.numeric(df_test$heart_attack), as.numeric(pred))
plot(roc_obj, main = "ROC Curve", col = "blue")
auc_val <- auc(roc_obj)
```

**Penjelasan**:  
- `roc()`: Menghitung ROC curve untuk memvisualisasikan performa model.
- `auc()`: Menghitung nilai AUC (Area Under Curve) sebagai ukuran performa klasifikasi.

---

## **14. Visualisasi Hyperplane SVM**

```r
plot(svm_model1, subset_df1, age ~ blood_pressure, 
     main = "SVM Hyperplane: Age vs Blood Pressure",
     col = c("lightblue", "lightpink"), 
     symbolPalette = c("red", "blue"))
```

**Penjelasan**: Visualisasi hyperplane untuk kombinasi fitur seperti `age` dan `blood_pressure`, menunjukkan bagaimana SVM memisahkan kelas berdasarkan kernel polynomial.

---
