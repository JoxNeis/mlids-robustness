## **2.4 Algoritma Klasifikasi**

Algoritma klasifikasi dilatih dengan cara mempelajari pola, karakteristik, dan hubungan implisit dari fitur-fitur data masa lalu (*training data*) untuk kemudian melakukan prediksi secara adaptif terhadap data baru yang belum pernah dikenali sebelumnya. Berikut adalah penjelasan mengenai beberapa algoritma klasifikasi yang akan diterapkan:

### **2.4.1	*K Nearest-Neighbor* (KNN)**

*K Nearest-Neighbor* (KNN) adalah salah satu metode klasifikasi paling sederhana. (Lu et al, 2023). Algoritma K Nearest-Neighbor (KNN) bekerja dengan cara mengklasifikasikan data baru berdasarkan kalkulasi kedekatan jarak geometrisnya terhadap tetangga terdekat sebanyak $K$ di dalam ruang fitur (Corpuz et al, 2026). Efektivitas algoritma *K Nearest-Neighbor* (KNN) sangat bergantung pada proses pemilihan metrik pengukuran jarak yang akan digunakan (Cetin and Buyuklu, 2025). 

![][image3]

Gambar 2.2 Visualisasi Algoritma KNN dengan K=8

Parameter $K$ yang menentukan jumlah tetangga terdekat dalam proses klasifikasi, memegang peranan krusial terhadap performa algoritma KNN. Pemilihan nilai $K$ yang tepat bukan tahapan yang sederhana, melainkan memerlukan pertimbangan serta analisis yang cermat (Cetin and Buyuklu, 2025). Meskipun, teknik *K* *Nearest-Neighbor* memiliki keunggulan berupa kemudahan implementasi serta mampu meminimalkan dampak negatif dari hilangnya informasi potensial, algoritma ini menuntut kapasitas penyimpanan memori yang besar dan memiliki kecepatan klasifikasi yang lebih lambat ketika dihadapkan pada himpunan data dengan distribusi kelas yang tidak seimbang (*skewed class distributions*) (Sisodia and Sisodia, 2022).

### **2.4.2	*Support Vector Machine* (SVM)**

*Support Vector Machine* (SVM) merupakan algoritma klasifikasi yang berpusat pada maksimisasi jarak (*margin*) antar-kelas demi meningkatkan kemampuan generalisasi model, dan minimalisasi kesalahan (*error*) dilakukan dengan menerapkan fungsi *hinge loss function* (Li et al., 2025). Algoritma SVM memanfaatkan *kernel functions* sehingga mampu menangani data yang tidak dapat dipisahkan secara linear melalui pemetaan ke dalam ruang fitur yang berdimensi lebih tinggi (Chen et al., 2025).   
Penggunaan SVM sangat populer untuk menangani tugas-tugas klasifikasi di berbagai industri. Algoritma SVM mampu diterapkan pada bidang biomekanika, bioinformatika, kesehatan, finansial, pemasaran, hingga pemrosesan citra dan audio (Mohasel and Koosha, 2026). Namun, algoritma SVM tidak luput dari kekurangan. Tingginya tingkat kompleksitas komputasi pada SVM konvensional menyebabkan proses eksekusi dan pelatihan data memakan waktu yang sangat lama, serta menuntut konsumsi sumber daya memori yang besar seiring bertambahnya volume dataset (Li et al., 2025).

### **2.4.3 *Logistic Regression***

*Logistic Regression* (LR) merupakan model regresi paling klasik dalam pembelajaran mesin. Algoritma LR bekerja dengan mempelajari koefisien regresi optimal melalui minimalisasi fungsi *logarithmic likelihood function*, sehingga mampu secara langsung menghasilkan nilai probabilitas suatu sampel terhadap kelas label yang sebenarnya (Wang et al, 2023). Meskipun menggunakan istilah "regresi", *Logistic Regression* (LR) merupakan metode klasik yang sangat penting dan digunakan secara luas dalam bidang statistika serta penggalian data (*data mining*), khususnya untuk klasifikasi data biner seperti *spam filtering* dan deteksi intrusi (*intrusion detection*) (Meziane, 2026).  Hal ini disebabkan algoritma *logistic regression* bekerja dengan cara mengestimasi nilai probabilitas suatu sampel tergolong ke suatu kelas (Meziane, 2026).  

Mirip dengan *Binary* *Logistic Regression* (LR), LR multi-kelas (atau dikenal sebagai *softmax regression*) mendefinisikan nilai probabilitas dari setiap sampel untuk tergolong ke dalam kelas tertentu sebagai berikut  

$P(y=k\ |{x}_{i})=\frac{{e}^{{w}_{k}^{T}{x}_{i}+b\ }}{\sum\limits_{l=1}^{c}{e}^{{w}_{l}^{T}{x}_{i}+b\ }};\ k=1,2,..,c$	 (2.2)   

Persamaan 2.2 adalah persamaan *logistic regression*, $P(y=k\ |{x}_{i})$ adalah probabilitas atau peluang bahwa *input* ${x}_{i}$  tergolong pada kelas *k* melalui *output* pada *y*  (*conditional probability*). ${w}_{k}$ adalah vektor bobot dan  *b* adalah bias. Dengan cara yang sama, fungsi tujuan *maximum likelihood* (*maximum likelihood objective function*) dari LR multi-kelas adalah sebagai berikut:    

$max\ log(\prod\limits_{i=1}^{n}P(y={y}_{i}|\ {x}_{i}))\ \Leftrightarrow \ max\ \sum\limits_{i=1}^{n}log\ P(y={y}_{i}|{x}_{i})$ (2.3) 

Persamaan 2.3 adalah persamaan *maximum log-likelihood* persamaan tersebut digunakan untuk menentukan *loss*. Persamaan tersebut tetap merupakan sebuah optimasi konveks (*convex optimization problem*). 

### **2.4.4 *XGBoost***

*Extreme Gradient Boosting* (*XGBoost*) merupakan salah satu algoritma machine learning tingkat lanjut (bersama dengan *Random Forest,* LSTM, dan algoritma lainnya) yang termasuk dalam kategori metode *boosting* (Moskal, 2025). Prinsip utama yang mendasari *XGBoost* adalah penambahan model baru secara berurutan (sequential). Model-model baru tersebut bertujuan untuk memperbaiki kesalahan yang dihasilkan oleh model sebelumnya. *XGBoost* menggunakan kerangka kerja gradient boosting, di mana model baru dibangun untuk memprediksi kesalahan (*residuals*) dari model-model sebelumnya sehingga kinerja prediksi dapat terus ditingkatkan (Zhang et al., 2025). 

*XGBoost* menerapkan regularisasi (*regularization*) yang dirancang untuk mencegah terjadinya *overfitting* sehingga model dapat melakukan generalisasi dengan lebih baik terhadap data baru. Selain itu, XGBoost menggunakan mekanisme *shrinkage*, yaitu pengurangan pengaruh *decision trees* yang ditambahkan pada tahap-tahap berikutnya untuk meningkatkan stabilitas model. *XGBoost* juga menerapkan *subsampling*, yaitu pemilihan fitur dan observasi secara acak selama proses pelatihan guna meningkatkan kemampuan generalisasi serta mengurangi risiko overfitting. Di samping itu, *XGBoost* dikenal memiliki efisiensi komputasi dan skalabilitas yang tinggi, sehingga mampu menangani dataset berukuran besar dengan cepat dan efektif (Moskal, 2025).

### **2.4.5 *Random Forest*** {#2.4.5-random-forest}

Random Forest (RF) merupakan salah satu algoritma *machine learning* (ML) yang bersifat serbaguna dan mudah digunakan. Algoritma ini telah terbukti memiliki kinerja yang cepat, kuat (*robust*), serta sangat kompetitif dalam berbagai macam aplikasi dan permasalahan. Karena kemampuannya dalam menghasilkan prediksi yang akurat dan stabil, *Random Forest* banyak digunakan dalam tugas klasifikasi maupun regresi pada berbagai bidang penelitian dan industri (Shaker, 2025).

Pada algoritma *Random Forest*, setiap *decision tree* dilatih menggunakan subset data dan subset fitur yang dipilih secara acak. Proses pengacakan ini menyebabkan setiap pohon memiliki karakteristik yang berbeda sehingga meningkatkan keberagaman (*diversity*) model. Selanjutnya, hasil prediksi dari seluruh pohon keputusan digabungkan melalui mekanisme pemungutan suara (*voting*) untuk tugas klasifikasi atau perhitungan rata-rata (*averaging*) untuk tugas regresi, sehingga diperoleh hasil prediksi akhir yang lebih akurat dan stabil (He, 2025).