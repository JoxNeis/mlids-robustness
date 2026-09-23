## **2.7 Simulasi *Noise* (*Noise Injection*)**

*Noise injection* merupakan teknik yang telah lama dikenal dalam penelitian *artificial neural network* (ANN), umumnya diterapkan pada tahap pelatihan sebagai strategi augmentasi untuk meningkatkan *robustness* model terhadap gangguan pada data input (Akbiyik, 2019; Abdolazimi et al., 2024). Prinsip yang mendasari pendekatan tersebut adalah bahwa performa model terhadap *noise* sangat dipengaruhi oleh karakteristik gangguan yang dihadapi, baik dari segi jenis maupun intensitasnya (Akbiyik, 2019). Dengan kata lain, respons suatu model terhadap data yang terganggu bukanlah sifat yang seragam, melainkan bergantung pada bagaimana dan seberapa besar gangguan tersebut diterapkan.

Prinsip inilah yang menjadi dasar pendekatan yang digunakan dalam penelitian ini, meskipun dengan tujuan yang berbeda dari penerapan *noise injection* pada umumnya. Alih-alih menyisipkan *noise* pada tahap pelatihan untuk membentuk *robustness* model secara aktif (*noise-augmented training*), penelitian ini menerapkan injeksi *noise* secara terkontrol pada tahap pengujian untuk mengukur tingkat *robustness* intrinsik yang telah dimiliki oleh model yang dilatih pada data bersih. Pendekatan ini merepresentasikan skenario yang lebih realistis pada konteks operasional ML-IDS, di mana suatu model umumnya dilatih menggunakan data historis yang relatif bersih, namun kemudian dihadapkan pada data lalu lintas jaringan *real-time* yang berpotensi mengandung gangguan tidak terduga saat *deployment*. Dengan demikian, sensitivitas model terhadap jenis dan intensitas *noise* yang ditunjukkan oleh Akbiyik (2019) tidak dimanfaatkan untuk merancang strategi augmentasi, melainkan dijadikan dasar untuk merancang skema pengujian bertingkat (10%, 20%, dan 30%) guna mengungkap sejauh mana performa klasifikasi model dapat terdegradasi ketika kondisi data menyimpang dari kondisi ideal saat pelatihan.

Dalam konteks sistem deteksi intrusi berbasis machine learning (ML-IDS), *noise* dapat muncul dalam berbagai bentuk yang merepresentasikan kondisi dunia nyata. *Noise* pada ML-IDS dapat berupa data dengan label yang salah (*mislabelled instances*), outlier, atau nilai ekstrim, dan menentukan sejauh mana efek noise tersebut sangat membantu dalam merancang serta membangun ML-IDS yang lebih *robust* (Al-Gethami et al., 2021). Penelitian empiris menunjukkan bahwa algoritma yang berbeda memiliki tingkat resiliensi yang berbeda terhadap noise; sebagai contoh, algoritma SVM terbukti sebagai algoritma yang paling resilient terhadap noise injection, sedangkan algoritma Random Forest merupakan yang paling rentan terhadap peningkatan level noise (Al-Gethami et al., 2021). Pendekatan ini diperkuat oleh penelitian pada deteksi intrusi berbasis graph, di mana berbagai faktor dapat mempengaruhi performa ML-NIDS, termasuk noise yang berasal dari *noise adversarial* yang disengaja oleh penyerang maupun keterbatasan hardware dan software dalam menangkap serta memproses data jaringan di dunia nyata (Liuliakov et al., 2025). Dengan demikian, simulasi noise secara sintetis pada dataset benchmark seperti NSL-KDD, UNSW-NB15, maupun CSE-CIC-IDS2018 merupakan metodologi yang telah diterima untuk mengevaluasi ketahanan model ML-IDS terhadap kondisi operasional yang tidak ideal (Al-Gethami et al., 2021; Liuliakov et al., 2025).

1\.	*Gaussian Noise*

Gaussian noise adalah gangguan acak yang mengikuti distribusi normal dengan mean $\mu =\ 0$ dan varians ${\sigma }^{2}$ tertentu.

$x'=x+\eta$ (2.4)

Persamaan 2.4  adalah persamaan yang menjelaskan pemberian *noise.* Dengan, $x$ adalah nilai data asli, $x'$ adalah nilai data setelah diberikan noise, dan $\eta$ pada *gaussian noise* adalah nilai *noise* yang mengikuti distribusi normal (Abdolazimi et al., 2024). Pada persamaan 2.4 nilai $\eta$ harus memenuhi syarat,

$\eta \sim \ 𝒩(\mu ,{\sigma }^{2\ })$ (2.5)

Persamaan 2.5 adalah persamaan yang menjelaskan bahwa, nilai $\eta$ berada dalam distribusi normal. Dimana, $\mu$ adalah nilai mean dari distribusi normal, dan ${\sigma }^{2}$ adalah varians dari distribusi normal . Pada implentasinya nilai $\mu$ adalah  $\mu =\ 0$  sehingga noise tidak memperkenalkan bias sistematis, hanya fluktuasi acak di sekitar nilai asli. *Gaussian noise* merepresentasikan kesalahan pengukuran alami pada sensor jaringan misalnya variasi kecil pada pengukuran durasi aliran (fl\_dur) atau ukuran paket (pkt\_len\_avg) akibat keterbatasan presisi *hardware*. 

2\. *Uniform Noise* 

Uniform noise adalah gangguan acak yang terdistribusi merata dalam suatu rentang (rentang telah ditentukan), di mana setiap nilai dalam rentang tersebut memiliki probabilitas yang sama. Persamaan 2.4  pada persamaan *gaussian noise* merupakan bentuk standar dari *feature noise* yang menunjukan pergeseran valuasi secara aditif*.*. Sehingga, persamaan tersebut dapat digunakan pada *uniform noise* dengan, $x$ adalah nilai data asli, $x'$ adalah nilai data setelah diberikan noise. Namun, nilai $\eta$ pada *uniform noise* adalah nilai *noise* yang mengikuti distribusi *uniform* (Abdolazimi et al., 2024). Pada persamaan 2.4  tersebut nilai $\eta$ harus memenuhi syarat distribusi *uniform* agar menjadi *uniform noise*.

$\eta \sim \ U(a,b)$ (2.6)

Persamaan 2.6 adalah persamaan yang menjelaskan bahwa, nilai $\eta$ berada dalam distribusi *uniform*. Di mana, pada persamaan 2.6  $a$ adalah batas bawah distribusi dan $b$ adalah batas atas dari distribusi *uniform*. Pada implementasinya, rentang umumnya ditetapkan simetris sehingga $a=-b$, agar *noise* tidak memperkenalkan *bias* menuju arah tertentu. Merepresentasikan ketidakpastian kuantisasi atau pembulatan nilai pada sistem pencatatan jaringan (e.g. nilai fl\_byt\_s yang dicatat dengan resolusi terbatas, atau variasi terbatas pada fitur timing paket).

3\. *Multiplicative Noise*

*Multiplicative noise* adalah gangguan yang dikalikan langsung dengan nilai data asli, sehingga besarnya gangguan bersifat proporsional terhadap magnitudo nilai asli. 

$x'=x\cdot \eta$ (2.7)

Persamaan 2.7 menjelaskan mekanisme *multiplicative noise*. Dengan $x$ adalah nilai data asli, $x'$ adalah nilai data setelah dikalikan dengan *noise*, dan η adalah nilai noise yang mengikuti distribusi normal serupa dengan persamaan 2.6 (Abdolazimi et al., 2024). Di mana $\mu$ adalah nilai *mean* dari distribusi normal dan ${\sigma }^{2}$adalah varians dari distribusi normal. Pada implementasinya, $\mu =\ 1$sehingga nilai rata-rata noise tidak mengubah skala keseluruhan data, hanya menambahkan fluktuasi proporsional. *Multiplicative noise* merepresentasikan degradasi proporsional (e.g. fitur fl\_pkt\_s yang terdistorsi secara multiplikatif akibat hambatan pada jaringan, atau manipulasi *adversarial* oleh penyeran).
