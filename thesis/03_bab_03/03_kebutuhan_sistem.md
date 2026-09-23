## **3.3 Kebutuhan SIstem**

Berdasarkan identifikasi masalah yang telah dilakukan, kebutuhan sistem yang dirumuskan adalah sebagai berikut:

### **3.3.1 Kebutuhan Pembangunan Model ML-IDS yang *Robust***

Sistem harus mampu membangun model *Machine Learning-based Intrusion Detection System* (ML-IDS) yang digunakan untuk mendeteksi aktivitas jaringan normal maupun aktivitas yang terindikasi sebagai serangan. Model yang dibangun akan menjadi dasar dalam proses evaluasi kinerja algoritma klasifikasi. Selain diuji pada kondisi data normal, model juga harus dapat dievaluasi pada data yang telah diberikan gangguan (*noise*) untuk mengetahui perubahan performa yang terjadi akibat penurunan kualitas data.

### **3.3.2 Kebutuhan Simulasi Feature Noise**

Sistem harus menyediakan mekanisme simulasi *feature noise* yang terkontrol pada dataset yang digunakan. Simulasi ini bertujuan untuk merepresentasikan kondisi data yang tidak ideal sebagaimana dapat terjadi pada lingkungan nyata, seperti kesalahan pengukuran, pencatatan, maupun transmisi data. Sistem juga harus memungkinkan penerapan beberapa tingkat *noise* sehingga pengaruh peningkatan gangguan terhadap kinerja model dapat diamati dan dianalisis secara sistematis.

### **3.3.3 Kebutuhan Pelatihan dan Pengujian Algoritma Klasifikasi**

Sistem harus mampu melakukan proses pelatihan dan pengujian beberapa algoritma klasifikasi menggunakan dataset yang sama. Penggunaan dataset dan skenario pengujian yang konsisten diperlukan untuk memastikan bahwa hasil evaluasi yang diperoleh dapat dibandingkan secara adil. Dengan demikian, perbedaan performa yang muncul dapat dikaitkan dengan karakteristik algoritma yang digunakan, bukan karena perbedaan data atau prosedur pengujian.

### **3.3.4 Kebutuhan Komparasi Sistematis Antar Algoritma**

Sistem harus mampu membandingkan kinerja algoritma klasifikasi berdasarkan metrik evaluasi yang telah ditentukan. Metrik yang digunakan meliputi akurasi, presisi, *recall*, dan *F1-score*. Perbandingan dilakukan pada berbagai tingkat *noise* untuk mengetahui tingkat ketahanan masing-masing algoritma terhadap penurunan kualitas data. Hasil perbandingan ini akan digunakan sebagai dasar dalam menentukan algoritma yang memiliki performa paling stabil pada kondisi data yang mengandung gangguan.

### **3.3.5 Kebutuhan Penyajian Hasil Evaluasi**

Sistem harus mampu menyajikan hasil evaluasi dan perbandingan kinerja algoritma dalam bentuk yang mudah dipahami. Penyajian hasil dilakukan melalui tabel dan visualisasi grafik yang menunjukkan perubahan nilai metrik evaluasi pada setiap tingkat *noise*. Dengan adanya visualisasi tersebut, analisis terhadap tren penurunan performa dan ketahanan masing-masing algoritma dapat dilakukan secara lebih jelas dan sistematis.
