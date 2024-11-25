Berikut adalah langkah-langkah untuk melakukan klasifikasi menggunakan metode **Support Vector Machine (SVM)** dengan dataset **Iris** di R. 

### 1. **Instal dan Muat Paket yang Dibutuhkan**
Pastikan Anda memiliki paket-paket yang dibutuhkan, seperti `e1071` untuk SVM dan `caret` untuk evaluasi.

```R
# Install packages jika belum terinstal
install.packages("e1071")
install.packages("caret")
```

```R
# Muat library
library(e1071)
library(caret)
```

---

### 2. **Muat Dataset Iris**
Dataset Iris sudah tersedia secara bawaan di R, jadi tidak perlu diunduh.

```R
# Muat dataset Iris
data(iris)

# Tampilkan ringkasan dataset
summary(iris)
```

---

### 3. **Bagi Dataset Menjadi Data Latih dan Uji**
Kita perlu membagi dataset menjadi data latih (training) dan data uji (testing) untuk mengevaluasi model.

```R
# Atur seed untuk hasil acak yang konsisten
set.seed(123)

# Bagi data menjadi 80% training dan 20% testing
index <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[index, ]
test_data <- iris[-index, ]
```

---

### 4. **Latih Model SVM**
Latih model SVM menggunakan data latih.

```R
# Latih model SVM
svm_model <- svm(Species ~ ., data = train_data, kernel = "linear")

# Tampilkan ringkasan model
summary(svm_model)
```

---

### 5. **Lakukan Prediksi**
Gunakan model yang telah dilatih untuk membuat prediksi pada data uji.

```R
# Prediksi pada data uji
predictions <- predict(svm_model, newdata = test_data)

# Tampilkan hasil prediksi
print(predictions)
```

---

### 6. **Evaluasi Model**
Hitung metrik evaluasi seperti akurasi atau confusion matrix.

```R
# Buat confusion matrix
confusion <- confusionMatrix(predictions, test_data$Species)

# Tampilkan hasil evaluasi
print(confusion)
```

---

### 7. **Visualisasi Hasil (Opsional)**
Anda dapat memvisualisasikan hasil klasifikasi jika datasetnya hanya melibatkan dua fitur.

```R
# Visualisasi dengan dua fitur pertama
plot(svm_model, train_data, Petal.Length ~ Petal.Width, slice = list(Sepal.Length = 5, Sepal.Width = 3))
```

---

### 8. **Modifikasi Kernel (Opsional)**
Anda dapat mencoba kernel lain seperti `radial`, `polynomial`, atau `sigmoid`.

```R
# Latih model SVM dengan kernel radial
svm_model_radial <- svm(Species ~ ., data = train_data, kernel = "radial")

# Evaluasi model radial
predictions_radial <- predict(svm_model_radial, newdata = test_data)
confusion_radial <- confusionMatrix(predictions_radial, test_data$Species)

print(confusion_radial)
```

---

### Output yang Diharapkan
- Confusion matrix akan menunjukkan performa model seperti akurasi, presisi, dan recall.
- Visualisasi (jika dilakukan) akan menampilkan hyperplane yang memisahkan kelas.

Semoga tutorial ini membantu! 😊
