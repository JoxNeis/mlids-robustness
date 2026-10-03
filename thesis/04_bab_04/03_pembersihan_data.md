## **4.3 Pembersihan Data (*Data Cleaning*)**

Analisis data eksploratif pada Subbab 4.2 menemukan tiga permasalahan kualitas data yang harus ditangani sebelum data digunakan untuk pemodelan, yaitu fitur konstan, nilai hilang, dan nilai tak hingga. Fitur konstan tidak membawa informasi untuk membedakan kelas, menambah dimensi data tanpa manfaat, dan memiliki simpangan baku nol sehingga tidak dapat distandarkan secara bermakna (Subbab 4.7). Nilai hilang dan nilai tak hingga tidak dapat diproses oleh sebagian besar algoritma, misalnya *Principal Component Analysis*, *K Nearest-Neighbor*, *Support Vector Machine*, dan *Logistic Regression*, karena nilai tersebut tidak memiliki besaran yang dapat dihitung jaraknya maupun dikalikan dengan bobot.

Fitur konstan ditangani dengan menghapus kolomnya, sedangkan nilai hilang dan nilai tak hingga ditangani dengan menghapus baris yang memuatnya. Penghapusan baris dipilih dibandingkan imputasi karena tiga alasan. Pertama, jumlah baris yang terdampak sangat kecil, yaitu 95.760 baris atau 0,59% dari seluruh data. Kedua, seluruh baris tersebut merupakan aliran dengan durasi nol (Subbab 4.2), sehingga nilai laju byte dan laju paket pada baris tersebut memang tidak terdefinisi dan tidak memiliki nilai pengganti yang benar. Penggantian dengan nilai tertentu, misalnya median atau nilai maksimum, akan memasukkan nilai buatan yang tidak pernah diukur. Ketiga, imputasi memerlukan statistik yang dipelajari dari data, sehingga harus dilakukan setelah pembagian dataset agar tidak terjadi kebocoran data (*data leakage*), sedangkan penghapusan baris tidak memerlukan statistik apa pun.

Seluruh langkah pembersihan bersifat deterministik pada setiap baris dan setiap kolom, sehingga aman dilaksanakan sebelum pembagian dataset (Subbab 4.5). Penghapusan fitur konstan juga tidak menimbulkan kebocoran data, karena fitur yang bernilai sama pada seluruh dataset pasti bernilai sama pula pada data latih maupun data uji. Daftar delapan fitur konstan ditetapkan dari hasil EDA dan digunakan sebagai konstanta pada seluruh berkas, sehingga setiap berkas memiliki susunan kolom yang identik setelah pembersihan.

Prosedur pembersihan data dilaksanakan melalui algoritma berikut:

1. **Pemuatan daftar berkas**, Seluruh berkas Parquet pada folder `data-pipeline/01-raw-parquet` didaftar dan diurutkan secara alfabetis, dan setiap berkas diproses secara terpisah.
2. **Penghapusan fitur konstan**, Delapan fitur konstan hasil EDA dihapus dari setiap berkas, sehingga setiap baris tersisa 70 fitur dan satu kolom label.
3. **Penghapusan baris dengan nilai hilang**, Baris yang memiliki sedikitnya satu nilai hilang dihapus dan jumlahnya dicatat.
4. **Penghapusan baris dengan nilai tak hingga**, Baris yang memiliki sedikitnya satu nilai +∞ atau −∞ pada kolom numerik dihapus dan jumlahnya dicatat.
5. **Penyimpanan hasil**, Data yang telah dibersihkan ditulis ke folder `data-pipeline/02-cleaned-parquet` dengan nama berkas yang sama dengan berkas sumber, sehingga informasi tanggal pengambilan data pada nama berkas tetap terjaga.
6. **Pelaporan hasil pembersihan**, Jumlah baris sebelum pembersihan, jumlah baris yang dihapus karena nilai hilang dan karena nilai tak hingga, serta jumlah baris setelah pembersihan dikumpulkan dari setiap berkas dan dijumlahkan per hari pengambilan data.

```mermaid
flowchart TD
    mulai(["Mulai"])
    mulai --> daftar[/"Membaca daftar berkas Parquet<br>dari 01-raw-parquet<br>(n = jumlah berkas)"/]
    daftar --> init_i["Inisialisasi i = 1"]
    init_i --> cek_i{"i ≤ n?"}

    cek_i -->|"Ya"| baca[/"Membaca berkas ke-i"/]
    baca --> konstan["Menghapus 8 fitur konstan"]
    konstan --> hilang["Menghapus baris dengan<br>nilai hilang"]
    hilang --> tak_hingga["Menghapus baris dengan<br>nilai +∞ atau −∞"]
    tak_hingga --> tulis[/"Menulis berkas ke format Parquet<br>dan mencatat jumlah baris"/]
    tulis --> simpan[("Penyimpanan<br>02-cleaned-parquet")]
    simpan --> inc_i["i = i + 1"]
    inc_i --> cek_i

    cek_i -->|"Tidak"| laporan[["Menjumlahkan baris sebelum,<br>baris yang dihapus, dan baris sesudah<br>pembersihan per hari"]]
    laporan --> selesai(["Selesai"])
```

Setiap berkas dibersihkan secara independen, sehingga iterasi berkas pada diagram di atas dijalankan secara paralel menggunakan `joblib` dengan backend `loky` dan delapan proses. Setiap proses hanya memuat satu berkas, yaitu maksimum 100.000 baris, sehingga kebutuhan memori tetap terkendali. Data pada folder `data-pipeline/01-raw-parquet` tidak diubah, sehingga data sebelum dan sesudah pembersihan dapat dibandingkan dan proses pembersihan dapat diulang apabila diperlukan.

Pemeriksaan nilai hilang dilaksanakan sebelum pemeriksaan nilai tak hingga. Karena 59.721 baris dengan nilai hilang pada *flow_byts_s* juga memiliki nilai tak hingga pada *flow_pkts_s*, baris tersebut tercatat sebagai baris yang dihapus karena nilai hilang, sedangkan 36.039 baris sisanya tercatat sebagai baris yang dihapus karena nilai tak hingga. Hasil pembersihan dirangkum pada Tabel 4.2.

Tabel 4.2 Hasil Pembersihan Data

| Besaran | Nilai |
| :--- | ---: |
| Jumlah baris sebelum pembersihan | 16.232.943 |
| Baris dihapus karena nilai hilang | 59.721 |
| Baris dihapus karena nilai tak hingga | 36.039 |
| Jumlah baris setelah pembersihan | 16.137.183 |
| Jumlah fitur sebelum pembersihan | 78 |
| Jumlah fitur setelah pembersihan | 70 |

Baris yang dihapus didominasi oleh kelas *Benign*, yaitu sebanyak 94.459 baris, sedangkan sisanya berasal dari kelas *Infilteration* sebanyak 1.295 baris dan kelas *FTP-BruteForce* sebanyak 6 baris. Seluruh kelas serangan lainnya, termasuk kelas minoritas seperti *SQL Injection*, *Brute Force -XSS*, dan *Brute Force -Web*, tidak kehilangan satu baris pun. Hal ini menegaskan bahwa penghapusan baris tidak mengurangi sampel kelas minoritas yang jumlahnya sudah terbatas, sehingga pendekatan penghapusan tidak merugikan kemampuan model untuk mengenali kelas serangan yang langka.
