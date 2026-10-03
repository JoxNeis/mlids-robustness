## **4.2 Analisis Data (*Exploratory Data Analysis*)**

Data hasil konversi pada Subbab 4.1 tidak dapat langsung digunakan untuk pemodelan. Data harus ditelaah terlebih dahulu melalui analisis data (*Exploratory Data Analysis*, EDA) untuk mengetahui karakteristik dari dataset CSE-CIC-IDS2018. Karakterisitik yang dicari yaitu sebaran data, distribusi kelas, statistik deskriptif setiap fitur, serta permasalahan kualitas data seperti nilai tak hingga (*infinite*), nilai hilang (*missing value*), fitur konstan, dan fitur yang saling berkorelasi kuat. Temuan pada tahap ini menjadi dasar keputusan pada tahap-tahap berikutnya, yaitu pembersihan data (Subbab 4.3), penghapusan data duplikat (Subbab 4.4), pembagian dataset (Subbab 4.5), dan rancangan *pipeline* praproses (Subbab 4.7). Dengan demikian, setiap langkah praproses yang diterapkan memiliki alasan yang dapat ditelusuri kembali pada data.

EDA dilakukan pada data dari hasil olahan *parquet*, yang terdiri atas 168 file Parquet dengan total 16.232.943 baris, 78 fitur, dan satu kolom label. Dikarenakan ukuran memori yang terbatas, ukuran data tersebut tidak memungkinkan seluruh data dimuat ke memori sekaligus, sehingga setiap analisis dirancang agar hanya membaca bagian data yang dibutuhkan. Jumlah baris dibaca dari metadata file Parquet tanpa membaca isi data, distribusi kelas dihitung per file, statistik deskriptif dihitung per fitur dengan hanya membaca satu kolom pada setiap perhitungan, dan korelasi antarfitur dihitung per file kemudian digabungkan. Pembacaan per kolom dapat dilakukan karena sifat dari *parquet* yang mendukung *columnar* dan dilakukan secara *lazy* menggunakan library `Polars`, sedangkan pemrosesan per file dijalankan secara paralel menggunakan `joblib`.

Prosedur analisis data eksploratif dilaksanakan melalui algoritma berikut:

1. **Penghitungan baris per hari**, Jumlah baris setiap file dibaca dari *metadata Parquet*, kemudian dijumlahkan menurut tanggal pengambilan data yang diambil dari awal nama file, yaitu pola `YYYY-MM-DD`.
2. **Penyusunan daftar kelas**, Seluruh nilai unik pada kolom label dikumpulkan dari setiap file, diurutkan secara alfabetis, dan disimpan pada file sebagai acuan daftar kelas bagi tahap pengodean label (Subbab 4.6).
3. **Penghitungan distribusi kelas**, Jumlah baris setiap kelas dihitung pada setiap file, kemudian dijumlahkan per hari dan untuk seluruh dataset beserta persentasenya.
4. **Penghitungan statistik deskriptif**, Untuk setiap fitur dihitung jumlah baris, jumlah nilai hilang, jumlah NaN, jumlah nilai tak hingga, nilai minimum, maksimum, rata-rata, median, simpangan baku, jumlah nilai negatif, jumlah nilai nol, dan jumlah nilai unik. Statistik nilai dihitung hanya dari nilai yang terhingga (*finite*) agar tidak terdistorsi oleh nilai tak hingga.
5. **Identifikasi fitur bermasalah**, Fitur yang memuat nilai tak hingga, nilai hilang, atau NaN disaring dari statistik deskriptif beserta persentase barisnya, demikian pula fitur konstan, yaitu fitur yang hanya memiliki satu nilai unik.
6. **Analisis korelasi**, Koefisien korelasi Pearson dihitung antara seluruh pasangan fitur yang tidak konstan, kemudian pasangan dengan nilai mutlak korelasi sedikitnya 0,95 didaftar sebagai pasangan yang berkorelasi tinggi.
7. **Visualisasi**, Hasil analisis disajikan dalam bentuk diagram batang jumlah baris per hari dan per kelas, peta panas (*heatmap*) jumlah baris setiap kelas pada setiap hari, dan peta panas korelasi antarfitur.

```mermaid
flowchart TD
    mulai(["Mulai"])
    mulai --> daftar[/"Membaca daftar file Parquet<br>dari 01-raw-parquet<br>(n = jumlah file)"/]
    daftar --> baris[["Menghitung jumlah baris setiap file<br>dari metadata Parquet"]]
    baris --> per_hari["Menjumlahkan baris<br>per hari pengambilan data"]
    per_hari --> kelas[["Mengumpulkan nilai unik label<br>dari seluruh file"]]
    kelas --> simpan_kelas[/"Menulis daftar kelas ke<br>original-classes.json"/]
    simpan_kelas --> distribusi[["Menghitung jumlah baris setiap kelas<br>per file secara paralel"]]
    distribusi --> init_j["Inisialisasi j = 1<br>(m = jumlah fitur)"]
    init_j --> cek_j{"j ≤ m?"}

    cek_j -->|"Ya"| statistik[["Membaca kolom fitur ke-j dan<br>menghitung statistik deskriptif"]]
    statistik --> inc_j["j = j + 1"]
    inc_j --> cek_j

    cek_j -->|"Tidak"| masalah["Mengidentifikasi fitur dengan nilai<br>tak hingga, nilai hilang, NaN,<br>dan fitur konstan"]
    masalah --> init_i["Inisialisasi i = 1"]
    init_i --> cek_i{"i ≤ n?"}
    cek_i -->|"Ya"| kovarians[["Menghitung n, rata-rata, dan<br>cross-product file ke-i"]]
    kovarians --> gabung["Menggabungkan statistik<br>dengan Persamaan 4.1"]
    gabung --> inc_i["i = i + 1"]
    inc_i --> cek_i

    cek_i -->|"Tidak"| korelasi[["Menghitung korelasi Pearson<br>dan mendaftar pasangan korelasi mutlak ≥ 0,95"]]
    korelasi --> visual[/"Menyajikan diagram batang<br>dan peta panas"/]
    visual --> selesai(["Selesai"])
```

Korelasi tidak dapat dihitung secara langsung pada seluruh data karena membutuhkan seluruh baris di memori. Oleh karena itu, setiap proses dibagi menjadi tiga perhitungan statistik, meliputi jumlah baris $n$, vektor rata-rata $\mu$, dan matriks jumlah hasil kali simpangan (*cross-product*) $C$, yang hanya dihitung dari baris yang seluruh nilainya terhingga. Ringkasan dari dua kelompok data $a$ dan $b$ kemudian digabungkan menggunakan persamaan berikut.

$
n=n_{a}+n_{b}
$ (4.1)

$
\delta ={\mu }_{b}-{\mu }_{a}
$ (4.2)

$
\mu ={\mu }_{a}+\delta \frac{n_{b}}{n} 
$ (4.3)

$
C=C_{a}+C_{b}+\delta {\delta }^{T}\frac{n_{a}n_{b}}{n}
$ (4.4)

Persamaan 4.4 adalah persamaan penggabungan statistik kovarians, dengan $n_{a}$ dan $n_{b}$ adalah jumlah baris pada tiap tahapan, ${\mu }_{a}$ dan ${\mu }_{b}$ adalah vektor rata-rata, serta $C_{a}$ dan $C_{b}$ adalah matriks *cross-product* dari kedua kelompok. Penggabungan dilakukan berurutan untuk seluruh file, sehingga statistik yang diperoleh sama dengan perhitungan pada seluruh data sekaligus tanpa harus memuat seluruh data ke memori. Koefisien korelasi kemudian dihitung dari matriks gabungan tersebut.

$r_{jk}=\frac{C_{jk}}{\sqrt{C_{jj}\,C_{kk}}}$ (4.5)

Persamaan 4.5 adalah koefisien korelasi Pearson antara fitur ke-$j$ dan fitur ke-$k$, dengan $C_{jk}$ adalah elemen matriks *cross-product* gabungan. Nilai $r_{jk}$ berada pada rentang −1 hingga 1, dan nilai mutlak yang mendekati 1 menandakan hubungan linear yang kuat antara kedua fitur.