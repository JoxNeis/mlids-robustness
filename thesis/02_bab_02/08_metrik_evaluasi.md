## **2.8 Metrik Evaluasi**

Evaluasi performa model klasifikasi pembelajaran mesin memerlukan metrik yang dapat mengukur kualitas prediksi secara kuantitatif. Dalam konteks sistem deteksi intrusi, evaluasi dilakukan dengan membandingkan hasil prediksi model terhadap label kelas yang sebenarnya pada data uji. Perbandingan tersebut menghasilkan empat kemungkinan hasil klasifikasi: *True Positive* (TP), yaitu sampel serangan yang berhasil diprediksi benar sebagai serangan; *True Negative* (TN), yaitu sampel lalu lintas normal yang berhasil diprediksi benar sebagai normal; *False Positive* (FP), yaitu sampel lalu lintas normal yang keliru diprediksi sebagai serangan; dan *False Negative* (FN), yaitu sampel serangan yang keliru diprediksi sebagai normal.

Keempat nilai tersebut disusun ke dalam *confusion matrix*, sebuah matriks yang merangkum seluruh hasil prediksi model terhadap kelas sebenarnya dalam satu tampilan terstruktur. *Confusion matrix* ini menjadi dasar bagi perhitungan seluruh metrik evaluasi yang digunakan dalam penelitian ini, yaitu *accuracy*, *precision*, *recall*, dan *F1-score* (Tharwat, 2021).

### **2.8.1 Akurasi (*Accuracy*)**

Accuracy merupakan metrik paling dasar yang mengukur proporsi prediksi yang benar dari keseluruhan prediksi yang dilakukan oleh model. Metrik ini memberikan gambaran umum mengenai seberapa baik model dalam mengklasifikasikan seluruh data secara keseluruhan (Tharwat, 2021). Persamaan accuracy didefinisikan sebagai berikut:  
$Accuracy\ =\ \frac{TP+TN}{TP+TN+FP+FN}$ (2.8)  
Persamaan 2.4 adalah persamaan akurasi, meskipun akurasi merupakan metrik yang intuitif dan mudah diinterpretasikan, metrik ini dapat memberikan gambaran yang menyesatkan apabila dataset mengalami ketidakseimbangan kelas (class imbalance). Pada kondisi tersebut, model yang hanya memprediksi kelas mayoritas tetap dapat memperoleh nilai accuracy yang tinggi meski kemampuan deteksi terhadap kelas minoritas sangat buruk (Almseidin et al., 2023).

### **2.8.2 Presisi (*Precision*)**

Precision mengukur proporsi prediksi positif yang benar-benar merupakan sampel positif dari seluruh data yang diprediksi sebagai positif oleh model. Metrik ini sangat relevan ketika biaya dari false positive sangat tinggi, seperti pada sistem keamanan jaringan yang tidak ingin menghasilkan terlalu banyak alarm palsu yang dapat mengganggu operasional sistem (Tharwat, 2021). Persamaan precision didefinisikan sebagai berikut:  
$Precision\ =\ \frac{TP}{TP+FP}$  (2.9)  
Persamaan 2.5 adalah persamaan nilai presisi, nilai presisi yang tinggi menunjukkan bahwa model jarang salah mengklasifikasikan sampel negatif (lalu lintas normal) sebagai positif (serangan).

### **2.8.3 *Recall***

*Recall*, yang juga dikenal sebagai *sensitivity* atau *true positive rate*, mengukur proporsi sampel positif yang berhasil diidentifikasi oleh model dari seluruh sampel positif yang sebenarnya ada dalam data. Metrik ini sangat penting dalam konteks deteksi intrusi, karena kegagalan mendeteksi serangan nyata (*false negative*) berpotensi menimbulkan kerugian signifikan bagi keamanan sistem (Almseidin et al., 2023). Persamaan *recall* didefinisikan sebagai berikut:  
$Recall\ =\ \frac{TP}{TP+FN}$ (2.10)  
Persamaan 2.56 adalah persamaan *recall,* nilai recall yang tinggi menunjukkan bahwa model mampu mendeteksi sebagian besar serangan nyata yang terdapat dalam data uji.

### **2.8.4 F1-*Score***

F1-*score* merupakan rata-rata harmonik dari precision dan recall, yang memberikan ukuran tunggal yang menyeimbangkan antara keduanya. Metrik ini sangat berguna dalam situasi di mana terdapat ketidakseimbangan antara *precision* dan *recall*, serta ketika dataset bersifat tidak seimbang secara distribusi kelas (Tharwat, 2021). Persamaan F1-score didefinisikan sebagai berikut:  
$F1\ =\ \frac{2\times Precision\times Recall}{Precision\ +\ Recall}$ (2.11)  
Persamaan 2.7 adalah persamaan F1-*score,* F1-*score* bernilai antara 0 dan 1, di mana nilai mendekati 1 mengindikasikan performa klasifikasi yang optimal. Dalam penelitian yang melibatkan pengujian ketahanan model pada berbagai kondisi *noise*, F1-*score* menjadi metrik yang diutamakan karena kemampuannya merepresentasikan performa model secara komprehensif pada dataset yang tidak seimbang (Almseidin et al., 2023).
