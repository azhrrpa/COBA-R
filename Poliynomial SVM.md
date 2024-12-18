Berikut adalah **Markdown GitHub** yang sudah ditata dengan format dan struktur rapi agar tampil sempurna di GitHub tanpa masalah **bold** atau penataan:

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

**Penjelasan**:  
Digunakan untuk membaca file CSV bernama `heart.csv` dari jalur file yang ditentukan dan memuatnya ke dalam variabel `df` dalam bentuk **data frame**. Data frame adalah struktur data berbentuk tabel yang terdiri dari baris (observasi) dan kolom (atribut).  

---

## **3. Penamaan Ulang Kolom**

```r
colnames(df) <- c("age", "sex", "chest_pain", "blood_pressure", "cholesterol", 
                  "fasting_blood_sugar", "restecg", "max_heart_rate", "angina", 
                  "oldpeak", "slope", "n_vessels", "thall", "heart_attack") 
```

**Penjelasan**:  digunakan untuk mengganti nama kolom dalam data frame df. Penyesuaian ini dilakukan untuk memberikan nama kolom yang lebih deskriptif, konsisten, dan mudah diakses saat melakukan analisis. Misalnya, kolom yang sebelumnya memiliki nama teknis atau tidak jelas kini menjadi lebih intuitif, seperti "age" untuk usia atau "cholesterol" untuk kadar kolesterol. Hal ini juga penting untuk memastikan nama kolom sesuai dengan sintaks R, khususnya jika nama asli kolom mengandung spasi atau karakter khusus. Dengan perubahan ini, analisis dan manipulasi data menjadi lebih efisien dan mudah dipahami.

---

## **4. Menghapus Nilai yang Hilang**

```r
cat("Jumlah nilai yang hilang:", sum(is.na(df)), "\n")
```

**Penjelasan**: Perintah ini digunakan untuk menghitung dan menampilkan jumlah nilai yang hilang (NA) dalam dataset df. Fungsi is.na(df) menghasilkan nilai TRUE untuk setiap elemen yang hilang, kemudian sum() menjumlahkan seluruh nilai TRUE tersebut. Outputnya menunjukkan total nilai yang hilang dalam dataset.

## **5. Konversi Kolom Kategorikal Menjadi Faktor**

```r
df$sex <- as.factor(df$sex)
df$fasting_blood_sugar <- as.factor(df$fasting_blood_sugar)
df$angina <- as.factor(df$angina)
df$n_vessels <- as.factor(df$n_vessels)
df$thall <- as.factor(df$thall)
df$heart_attack <- as.factor(df$heart_attack)
```

**Penjelasan**:  
Kolom-kolom dengan data kategorikal, seperti sex, fasting_blood_sugar, angina, dan lainnya, diubah menjadi tipe data factor. Konversi ini penting untuk analisis dan pemodelan, terutama dalam algoritma seperti SVM (Support Vector Machine), yang memerlukan tipe data faktor untuk memproses variabel kategorikal secara benar.

---

## **6. Korelasi Antar Fitur Numerik**

```r
numeric_cols <- sapply(df, is.numeric)
df_corr <- cor(df[, numeric_cols], use = "complete.obs")
```
**Penjelasan**: Kode ini digunakan untuk menghitung matriks korelasi dari kolom numerik dalam dataset df. Fungsi sapply(df, is.numeric) mengidentifikasi kolom yang bertipe numerik dan menyimpannya dalam vektor logika numeric_cols. Kemudian, fungsi cor() menghitung nilai korelasi antar kolom numerik tersebut dengan hanya menggunakan observasi lengkap (tanpa nilai yang hilang) sesuai parameter use = "complete.obs". Hasilnya, df_corr berisi matriks korelasi yang menunjukkan hubungan linear antar kolom numerik.

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
**Penjelasan**: Kode ini digunakan untuk membuat visualisasi heatmap korelasi menggunakan ggplot2. Matriks korelasi df_corr yang berisi hubungan antar variabel numerik diubah menjadi format panjang dengan fungsi melt, sehingga dapat divisualisasikan. Heatmap dibuat menggunakan geom_tile, di mana nilai korelasi diwakili oleh gradasi warna: biru untuk korelasi negatif, merah untuk positif, dan putih untuk nol. Label angka pada setiap sel ditambahkan dengan geom_text untuk menampilkan nilai korelasi secara langsung. Tema minimalis (theme_minimal) diterapkan untuk tampilan bersih, sementara teks pada sumbu x dimiringkan agar label variabel tidak bertumpuk. Visualisasi ini membantu memahami hubungan linear antar variabel numerik dalam dataset dengan cepat dan intuitif.

---

## **8. Standarisasi Fitur Numerik**

```r
df[numeric_cols] <- scale(df[numeric_cols])
```
**Penjelasan**: Kode ini digunakan untuk melakukan normalisasi atau standarisasi pada kolom numerik dalam dataset df. Fungsi scale() mengubah setiap kolom numerik sehingga memiliki mean (rata-rata) 0 dan standard deviation (simpangan baku) 1. Normalisasi ini penting untuk memastikan semua variabel numerik berada pada skala yang sama, terutama saat digunakan dalam algoritma machine learning seperti SVM atau k-Means yang sensitif terhadap skala variabel. Setelah eksekusi, semua kolom numerik dalam df akan berada pada skala yang seragam, sehingga meminimalkan bias akibat perbedaan skala antar variabel.

---

## **9. Identifikasi Outliers**

```r
z_scores <- scale(df[, numeric_cols])
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)
```
**Penjelasan**: Kode ini digunakan untuk mendeteksi outlier pada kolom numerik dalam dataset df menggunakan metode z-score. Fungsi scale() menghitung z-score, yaitu ukuran deviasi setiap nilai dari rata-rata dalam satuan simpangan baku. Nilai z-score yang lebih besar dari 3 (atau kurang dari -3) dianggap sebagai outlier karena berada jauh dari rata-rata. Perintah which(..., arr.ind = TRUE) digunakan untuk menemukan posisi elemen-elemen tersebut dalam dataset, sehingga menghasilkan indeks baris dan kolom yang mengindikasikan lokasi outlier. Proses ini membantu mengidentifikasi nilai ekstrem yang dapat memengaruhi hasil analisis atau model.

---

## **10. Membagi Dataset**

```r
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```
**Penjelasan**: Kode tersebut bertujuan untuk membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan fungsi createDataPartition(), yang memastikan pembagian stratifikasi berdasarkan variabel target (heart_attack). Data latih (df_train) digunakan untuk melatih model, sementara data uji (df_test) digunakan untuk mengukur kinerja model pada data yang belum pernah dilihat, sehingga membantu mengevaluasi kemampuan generalisasi model.

---

## **11. Pelatihan Model SVM dengan Kernel Polynomial**

```r
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmPoly", 
                   tuneGrid = expand.grid(C = c(0.1, 1, 10), degree = c(2, 3), scale = c(0.01, 0.1)))
```
**Penjelasan**: Model dilatih menggunakan fungsi train() dari paket caret, di mana heart_attack ~ . menunjukkan bahwa variabel target adalah heart_attack, sementara semua variabel lainnya digunakan sebagai fitur. Metode svmPoly digunakan untuk memilih kernel polinomial. Parameter C, degree, dan scale disesuaikan berdasarkan nilai-nilai dalam tuneGrid untuk menemukan kombinasi terbaik yang menghasilkan kinerja optimal.

---

Cukup copas langsung ke **GitHub** ya! Formatnya pasti sesuai.
