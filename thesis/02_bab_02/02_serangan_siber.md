## **2.2 Serangan Siber**

Jenis serangan yang dibahas dalam penelitian ini mengacu pada dataset CSE-CIC-IDS2018, yang mencakup berbagai kategori serangan seperti *brute force*, *DoS*, *port scan*, *web attacks*, *botnet, heartbleed*, dan *infiltration*.

### **2.2.1 Brute Force**

*Brute force* adalah salah satu serangan sederhana pada sistem siber, dimana penyerang mencoba seluruh kombinasi hingga mendapatkan solusi yang tepat (Park et al, 2021). Serangan *brute force* tidak melakukan eksploitasi atau memanfaatkan kerentanan spesifik pada sistem, namun mengandalkan pendekatan uji coba (*trial and error*). Pada sistem keamanan siber, serangan *brute force* umumnya ditujukan untuk memperoleh akses tidak sah pada sistem autentikasi (e.g. *secure shell*, *File transfer protocol*, dan *remote desktop protocol*).

Serangan *brute force* pada kata sandi yang panjang dan kompleks, akan meningkatkan jumlah kombinasi yang harus dicoba secara eksponensial. Hal ini disebabkan oleh semakin panjang atau kompleks kata sandi yang digunakan, kombinasi kemungkinan kata sandi yang ada (ruang sampel) akan semakin banyak. Kombinasi kemungkinan kata sandi akan memenuhi persamaan, 

$P=C^l$ (2.1)

Persamaan 2.1 adalah persamaan jumlah kombinasi kata sandi dari jumlah digit dan kemungkinan digit, $P$ adalah jumlah kemungkinan kombinasi kata sandi, $C$ adalah jumlah karakter yang digunakan, dan $l$ adalah panjang dari kata sandi tersebut. 

![][image2]

Gambar 2.1  Kurva fungsi exponen jumlah kombinasi kemungkinan kata sandi.

Pada persamaan 2.1, kombinasi kemungkinan kata sandi meningkat secara eksponensial dari jumlah karakter yang dapat digunakan dan panjang kata sandi mengikuti aturan kombinasi ruang sampel probabilitas. Pada kurva tersebut terlihat peningkatan kompleksitas *brute force* yang meningkat secara eksponensial.

### **2.2.2 *Denial of Service* (DoS)**

*Denial of service* (DoS) adalah serangan siber yang bertujuan mengganggu aktivitas sistem dengan menghalangi atau memblokir layanan sistem (Stallings and Brown, 2024). Serangan DoS bekerja dengan menghabiskan sumber daya sistem (e.g. membanjiri *web server* dengan begitu banyak permintaan \[*request*\]) sehingga sistem tidak dapat mengembalikan respon. DoS menyerang pada aspek ketersediaan (*availability*) suatu sistem, melalui membanjiri *bandwidth* atau lalu lintas data suatu sistem. Hal ini membuat sistem *overload* sehingga layanan tidak bisa diakses oleh pengguna. 

Dalam perkembangannya, taktik ini berevolusi menjadi *Distributed Denial of Service* (DDoS) guna menyamarkan identitas penyerang sekaligus melipatgandakan dampak kerusakan. Serangan DDoS memanfaatkan jaringan komputer yang tersebar di berbagai lokasi geografis—umumnya terdiri dari perangkat-perangkat yang telah terinfeksi dan dikendalikan secara jarak jauh melalui *botnet*—untuk menyerang target secara simultan (Wani et al., 2026). Serangan ini menggunakan lonjakan volume data yang masif, kompleks, dan memicu tingkat kesulitan yang tinggi dalam proses identifikasi (Srivastava & Sinha, 2025).

### **2.2.3. *Port Scan***

*Port scan* atau pemindaian *port* merupakan sebuah metode serangan dengan mengirim banyak permintaan (*request*) ke berbagai *port* untuk mengidentifikasi kelemahan pada suatu sistem. (Wu et al., 2023). Proses ini dilakukan dengan cara mengirimkan paket permintaan koneksi secara sistematis ke serangkaian port spesifik pada alamat IP target untuk menganalisis respons yang dikembalikan. Melalui respons balik tersebut, penyerang dapat memetakan arsitektur jaringan, mengidentifikasi jenis sistem operasi yang digunakan, serta mendeteksi layanan aplikasi yang sedang berjalan secara aktif.

Dalam lanskap keamanan siber, *portscan* dikategorikan sebagai tahap pengumpulan informasi awal (*reconnaissance*) sebelum meluncurkan serangan yang lebih destruktif (Almomani et al., 2026). Penyerang memanfaatkan informasi dari port yang terbuka untuk mencari celah keamanan (*vulnerability*) yang belum ditambal pada layanan tersebut (Wu et al., 2023). *Port Scanning* adalah salah satu serangan yang mudah dilakukan, efektif, namun mudah dideteksi sebab ketika koneksi dibuat, sistem pada umumnya mencatat alamat IP yang melakukan *port scan*. Sehingga, banyak metode *port scan* yang telah dibuat dan beberapa alat yang terkenal seperti *nmap*.

### **2.2.4 *Web Attacks***

*Web attacks* merujuk pada segala bentuk aktivitas berbahaya yang secara khusus menargetkan kelemahan aplikasi berbasis web untuk mengakses sistem jaringan, data, asset, dan informasi suatu organisasi tanpa izin (Zhou et al., 2026). Serangan web sangatlah berbahaya sebab serangan pada aplikasi web sangat mudah untuk mengekspos data sensitif dan mencuri akses. Beberapa bentuk ancaman yang paling umum pada aplikasi web, meliputi *SQL Injection* (SQLi) yang memanipulasi *query* pada *Structured Query Language* (SQL) dalam transaksi data, *Cross-Site Scripting* (XSS) yang memasukan skrip berbahaya ke dalam aplikasi web, dan *Cross Site Request Forgery* (CSRF) dimana penyerang dapat mengakses atau berinteraksi dengan aplikasi diluar akses yang diberi oleh pemilik aplikasi (Kaya et al., 2026).

Dampak dari *web attacks* dapat mengancam integritas dan kerahasiaan data sensitif organisasi secara fatal, termasuk memicu kebocoran data (*data breach*) dalam skala besar (Zhou et al., 2026). Penyerang yang berhasil menembus perimeter aplikasi web sering kali dapat melakukan modifikasi data ilegal atau bahkan mengambil alih hak administrator server..

### **2.2.5 *Botnet***

*Botnet* (*robot network*) adalah sekumpulan perangkat komputer atau infrastruktur *Internet of Things* (IoT) yang telah terinfeksi oleh *malware* dan dikendalikan dari jarak jauh secara kolektif (Wani et al., 2026). Perangkat-perangkat yang terinfeksi, disebut sebagai *bots* atau *zombies*, beroperasi di bawah kendali penyerang utama yang dikenal sebagai *botmaster* (Wani et al., 2026). Mekanisme instruksi pertukaran data antara *botmaster* dan jaringan *zombies* tersebut difasilitasi melalui infrastruktur yang disebut *server Command and Control* (C\&C) (Wani et al., 2026).

Kekuatan utama dari arsitektur botnet terletak pada kemampuannya untuk menggalang sumber daya komputasi secara terdistribusi dan masif guna meluncurkan serangan skala besar, seperti *Distributed Denial of Service* (DDoS) atau penyebaran *spam* secara simultan (Wani et al., 2026). Karena melibatkan ribuan hingga jutaan IP yang tersebar secara geografis, deteksi dan pemblokiran terhadap botnet menjadi tantangan yang sangat kompleks (Wani et al., 2026). Penanganan *botnet* biasanya melibatkan analisis anomali berbasis pembelajaran mesin untuk mengenali pola perilaku tidak wajar pada konsumsi daya atau lalu lintas jaringan perangkat (Wani et al., 2026).

### **2.2.6 *Infiltration***

*Infiltration* dalam konteks keamanan jaringan merujuk pada teknik yang digunakan oleh penyerang untuk memasuki sistem atau jaringan target secara tidak sah dengan cara melewati atau mengeksploitasi mekanisme keamanan yang ada. Infiltrasi dapat dilakukan melalui berbagai metode, seperti eksploitasi kerentanan perangkat lunak, rekayasa sosial (*social engineering*), pencurian kredensial, maupun pemanfaatan *backdoor* yang telah ditanamkan sebelumnya (Stallings and Brown, 2024). Tujuan utama dari infiltrasi adalah memperoleh akses awal ke dalam sistem target sehingga penyerang dapat melanjutkan serangan lebih lanjut, seperti eskalasi hak akses, pencurian data, atau pemasangan malware

Teknik infiltrasi terus berkembang seiring meningkatnya kompleksitas sistem informasi dan jaringan yang digunakan oleh organisasi. Salah satu teknik umum yang banyak digunakan adalah serangan *man-in-the-middle* (MitM), di mana penyerang memposisikan dirinya di antara dua pihak yang berkomunikasi untuk mencegah atau memodifikasi lalu lintas data tanpa sepengetahuan korban (Anderson, 2020). Selain itu, teknik *phishing* dan *spear-phishing* juga merupakan teknik infiltrasi yang sangat efektif, karena menargetkan kelemahan manusia sebagai komponen paling rentan dalam sistem keamanan. Seiring meningkatnya penerapan sistem deteksi intrusi (IDS) dan firewall, penyerang pun mulai menggunakan teknik infiltrasi yang lebih canggih, seperti penggunaan enkripsi untuk menyamarkan lalu lintas berbahaya. 

### **2.2.7 Heartbleed**

*Heartbleed* merupakan kerentanan kritis pada pustaka kriptografi *OpenSSL* yang secara resmi dikatalogkan sebagai CVE-2014-0160. Kerentanan ini terdapat pada implementasi ekstensi *heartbeat* dalam protokol TLS/DTLS, yang berfungsi untuk memverifikasi keaktifan koneksi antara dua pihak. Mekanisme eksploitasi *Heartbleed* memanfaatkan tidak adanya validasi terhadap *payload* pada paket *heartbeat* request. Penyerang mengirimkan paket dengan pesan berukuran kecil namun mengklaim *packet length* yang jauh lebih besar, sehingga server yang rentan akan membaca dan mengembalikan data dari memorinya di luar batas buffer yang seharusnya sehingga berpotensi mengekspos informasi sensitif seperti *private key*, informasi kredensial pengguna, dan data *session* yang aktif (Maseer et al., 2021). 

Dalam dataset CSE-CIC-IDS2018, serangan *Heartbleed* direpresentasikan melalui simulasi eksploitasi terhadap server yang menjalankan versi *OpenSSL* yang rentan. Pola lalu lintas yang dihasilkan bersifat asimetris, di mana volume data yang mengalir dari server ke klien jauh lebih besar dibandingkan arah sebaliknya, dengan ukuran paket *request* yang tidak proporsional terhadap ukuran *response*. Karakteristik ini menjadikan *Heartbleed* sebagai kategori serangan yang dapat dibedakan dari lalu lintas normal melalui pendekatan deteksi anomali berbasis pembelajaran mesin (Kanimozhi & Prem Jacob, 2019).
