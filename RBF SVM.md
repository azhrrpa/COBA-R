Berikut adalah **SVM dengan Kenerl RBF** 

---

### **Pendahuluan**
Support Vector Machine (SVM) adalah algoritma machine learning yang dapat digunakan untuk klasifikasi dan regresi. Salah satu kekuatan utama SVM terletak pada kemampuan kernel untuk memetakan data ke dimensi yang lebih tinggi, sehingga data yang tidak terpisahkan secara linear menjadi lebih mudah untuk dipisahkan.

RBF (Radial Basis Function) kernel adalah salah satu jenis kernel yang paling populer dan sering digunakan dalam Support Vector Machine (SVM). Kernel ini mampu menangani data non-linear dengan sangat baik, karena dapat memetakan data dari ruang input ke ruang fitur berdimensi lebih tinggi, di mana data tersebut menjadi lebih mudah dipisahkan.

RBF kernel bekerja dengan mengukur jarak antara dua titik data dalam ruang input dan mengonversi jarak tersebut menjadi nilai kesamaan (similarity). Fungsi RBF mengasumsikan bahwa titik data yang lebih dekat satu sama lain memiliki kesamaan yang lebih tinggi, sedangkan titik data yang lebih jauh memiliki kesamaan yang rendah.

#### **Mengapa Menggunakan RBF Kernel?**
Mengatasi data non-linear:
RBF kernel mampu membentuk batas keputusan yang melengkung, sehingga lebih fleksibel dibandingkan dengan kernel linear.

Fleksibilitas yang tinggi:
Kernel ini dapat menyesuaikan dengan berbagai jenis distribusi data, sehingga cocok untuk beragam masalah klasifikasi dan regresi.

Parameter γ (gamma):
RBF kernel memiliki parameter gamma (γ) yang menentukan seberapa jauh pengaruh dari satu titik data dapat menjangkau titik lain.

Gamma kecil: Model lebih sederhana dan cenderung menghasilkan batas keputusan yang lebih mulus.
Gamma besar: Model menjadi lebih kompleks dan dapat memisahkan data yang lebih rumit, tetapi berisiko mengalami overfitting.

---

#### **Visualisasi Hasil**
- **ROC Curve**: Menampilkan performa prediksi model dengan area di bawah kurva (AUC) sebesar *x.x*. 
- **Hyperplane**: Menunjukkan pembagian kelas dalam beberapa kombinasi fitur, seperti `age` vs `blood_pressure` dan `max_heart_rate` vs `oldpeak`.

---

### 🗂 Dataset

Model SVM ini dilatih menggunakan dataset **Heart Disease** yang dapat dilihat di repositori berikut:  
🔗 [heart.csv](https://github.com/azhrrpa/SVM-R/blob/main/heart.csv)

---

## **1. Importing Required Libraries**
Kode memuat sejumlah pustaka R yang relevan untuk analisis, pemrosesan data, dan visualisasi, seperti `ggplot2` untuk plot, `caret` untuk pelatihan model machine learning, dan `e1071` untuk algoritma SVM.

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

## **2. Load Dataset**
File dataset diimpor dari path lokal. Disarankan mengganti path dengan format relatif atau menempatkan dataset dalam direktori kerja agar file dapat diakses secara universal.

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

**Penjelasan**:  
digunakan untuk mengganti nama kolom dalam data frame df. Penyesuaian ini dilakukan untuk memberikan nama kolom yang lebih deskriptif, konsisten, dan mudah diakses saat melakukan analisis. Misalnya, kolom yang sebelumnya memiliki nama teknis atau tidak jelas kini menjadi lebih intuitif, seperti "age" untuk usia atau "cholesterol" untuk kadar kolesterol. Hal ini juga penting untuk memastikan nama kolom sesuai dengan sintaks R, khususnya jika nama asli kolom mengandung spasi atau karakter khusus. Dengan perubahan ini, analisis dan manipulasi data menjadi lebih efisien dan mudah dipahami.

---

## **4. Menghapus Nilai yang Hilang**

```r
cat("Jumlah nilai yang hilang:", sum(is.na(df)), "\n")
```

**Penjelasan**: 
Perintah ini digunakan untuk menghitung dan menampilkan jumlah nilai yang hilang (NA) dalam dataset df. Fungsi is.na(df) menghasilkan nilai TRUE untuk setiap elemen yang hilang, kemudian sum() menjumlahkan seluruh nilai TRUE tersebut. Outputnya menunjukkan total nilai yang hilang dalam dataset.

## **5. Preprocessing**
Kolom diubah namanya agar lebih mudah digunakan, missing values diidentifikasi, dan fitur kategorikal dikonversi menjadi tipe faktor. Langkah ini memastikan dataset bersih dan siap digunakan.

```r
df$sex <- as.factor(df$sex)
df$fasting_blood_sugar <- as.factor(df$fasting_blood_sugar)
df$angina <- as.factor(df$angina)
df$n_vessels <- as.factor(df$n_vessels)
df$thall <- as.factor(df$thall)
df$heart_attack <- as.factor(df$heart_attack)
```

**Penjelasan**:  
Kode tersebut digunakan untuk mempersiapkan dataset df dengan mengganti nama kolom menjadi lebih deskriptif dan mengonversi beberapa kolom menjadi tipe data faktor. Kolom sex, fasting_blood_sugar, angina, n_vessels, thall, dan heart_attack diubah menjadi faktor karena kolom-kolom ini merepresentasikan data kategori, bukan numerik. Langkah ini penting untuk memastikan bahwa algoritma machine learning dapat memahami dan memproses data dengan benar sesuai dengan jenis variabelnya.

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

**Output Korelasi**

![alt text](https://github.com/azhrrpa/SVM-R/blob/main/gambar/Korelasi.png?raw=true)

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

### **10. Dataset Splitting**
Dataset dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian menggunakan fungsi `createDataPartition`.

```r
set.seed(42)
trainIndex <- createDataPartition(df$heart_attack, p = 0.8, list = FALSE)
df_train <- df[trainIndex, ]
df_test <- df[-trainIndex, ]
```
**Penjelasan**: Kode tersebut digunakan untuk membagi dataset df menjadi data latih (df_train) dan data uji (df_test) dengan proporsi 80:20, menggunakan fungsi createDataPartition(). Fungsi set.seed(42) memastikan pembagian data bersifat reproducible, sehingga hasilnya akan sama setiap kali kode dijalankan. Pembagian ini dilakukan secara stratified berdasarkan variabel target heart_attack, memastikan distribusi kelas tetap seimbang di data latih dan data uji. Data latih digunakan untuk melatih model, sementara data uji digunakan untuk mengevaluasi kinerja mode

---

## **11. Model Training SVM dengan RBF Kernel **

```r
ctrl <- trainControl(method = "cv", number = 5)  # Cross-validation 5 kali
svm_model <- train(heart_attack ~ ., data = df_train, method = "svmRadial", 
                   trControl = ctrl,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.05, 0.1)))  # Tuning C dan sigma
```
**Penjelasan**: 
Kode ini digunakan untuk **melatih model SVM dengan kernel RBF**, sambil menerapkan **cross-validation** dan **hyperparameter tuning**. Cross-validation dilakukan dengan membagi dataset menjadi 5 fold untuk memastikan model dapat **menggeneralisasi dengan baik**. Model dilatih menggunakan fungsi `train()` dengan memprediksi variabel target `heart_attack`, dan tuning parameter dilakukan untuk mencari kombinasi terbaik dari **C** dan **sigma**. Parameter **C** mengatur keseimbangan antara margin besar dan kesalahan klasifikasi, sementara **sigma** menentukan jangkauan pengaruh tiap titik data. Kode ini bertujuan untuk menghasilkan **model yang akurat dan stabil** dengan batas keputusan yang kompleks untuk data non-linear.

---

### **12. Evaluasi Akurasi Model SVM dengan Cross-Validation**

Apa Itu Cross-Validation?
Cross-validation adalah metode untuk mengevaluasi performa model machine learning dengan cara membagi dataset ke beberapa bagian (folds). Setiap bagian digunakan bergantian sebagai data uji dan data latih untuk menghindari overfitting dan memastikan model memiliki performa yang konsisten di berbagai subset data.

```r
cat("Hasil Cross-Validation:\n")
```
Menampilkan pesan ke console untuk memberi tahu bahwa hasil evaluasi cross-validation sedang diproses. 

```r
mean_accuracy <- mean(svm_model$results$Accuracy)
```
Rata-rata akurasi dihitung dari hasil cross-validation, yang menunjukkan seberapa baik model dapat memprediksi data dengan benar.

```r
std_dev_accuracy <- sd(svm_model$results$Accuracy)
```
Standar deviasi digunakan untuk mengukur variabilitas hasil akurasi dari berbagai subset data (folds).
>>> Semakin rendah standar deviasi, semakin konsisten model dalam memprediksi data.

```r
cat("Rata-rata Akurasi:", mean_accuracy, "\n")
```
Setelah rata-rata akurasi dihitung, hasil tersebut ditampilkan ke console agar dapat dianalisis.

```r
cat("Standar Deviasi Akurasi:", std_dev_accuracy, "\n")
```
Standar deviasi akurasi juga ditampilkan ke console untuk mengetahui stabilitas performa model.

### **13. Evaluation Metrics**
Model dievaluasi menggunakan data pengujian. Metrik seperti akurasi, precision, recall, F1-score, dan AUC dihitung dan ditampilkan.

```r
pred <- predict(svm_model, df_test)
conf_matrix <- confusionMatrix(pred, df_test$heart_attack)
roc_obj <- roc(as.numeric(df_test$heart_attack), as.numeric(pred))
auc_val <- auc(roc_obj)
```
**Penjelasan**: Kode tersebut digunakan untuk mengevaluasi kinerja model SVM yang telah dilatih. Prediksi pada data uji (df_test) dilakukan menggunakan fungsi predict(), dan hasilnya disimpan dalam pred. Matriks kebingungan (confusion matrix) dihitung menggunakan confusionMatrix() untuk mengevaluasi metrik seperti akurasi, sensitivitas, dan spesifisitas. Selain itu, kurva ROC dihitung menggunakan fungsi roc(), yang membandingkan nilai aktual dan prediksi sebagai numerik, untuk mengukur kemampuan model dalam membedakan kelas. Nilai Area Under the Curve (AUC) dihitung dengan auc() untuk memberikan satu angka yang mencerminkan performa model, di mana nilai yang lebih tinggi menunjukkan performa yang lebih baik.

### **14. Visualizing Hyperplanes**
Plot hyperplane SVM dibuat menggunakan kombinasi 2 fitur dengan `plot.svm`. Ini membantu memahami klasifikasi dalam ruang 2D.

ini merupakan salah satu dari visualisasi:

```r
subset_df1 <- df[, c("age", "blood_pressure", "heart_attack")]
svm_model1 <- svm(heart_attack ~ age + blood_pressure, data = subset_df1, kernel = "radial", cost = 1, scale = TRUE)
plot(svm_model1, subset_df1, age ~ blood_pressure, 
     main = "SVM Hyperplane: Age vs Blood Pressure (RBF Kernel)",
     col = c("lightblue", "lightpink"), 
     symbolPalette = c("red", "blue"))
legend("topright", legend = levels(subset_df1$heart_attack), col = c("red", "blue"), pch = 16)

```
**Penjelasan**: 
Kode ini digunakan untuk melatih model SVM dengan kernel RBF (Radial Basis Function) untuk memprediksi serangan jantung berdasarkan dua fitur, yaitu umur (age) dan tekanan darah (blood pressure). Pertama, data yang digunakan disaring menjadi subset yang hanya mencakup kolom age, blood_pressure, dan heart_attack. Kemudian, model SVM dilatih dengan kernel RBF, dengan parameter cost yang diatur ke 1, yang mengontrol penalti terhadap kesalahan klasifikasi, dan scale = TRUE untuk menyelaraskan fitur agar memiliki skala yang sama, yang penting untuk kinerja kernel. Setelah model dilatih, hasilnya divisualisasikan dalam bentuk plot 2D yang menggambarkan hubungan antara age dan blood_pressure serta batas keputusan yang dihasilkan oleh model. Plot ini menggunakan warna yang berbeda untuk menunjukkan kelas heart_attack yang berbeda, dengan red dan blue mewakili dua kategori target. Legenda ditambahkan di pojok kanan atas untuk menjelaskan warna titik-titik tersebut.


