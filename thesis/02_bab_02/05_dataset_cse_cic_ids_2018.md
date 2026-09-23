## **2.5 Dataset CSE-CIC-IDS2018**

Dataset yang dikembangkan oleh CIC dan ISCX banyak digunakan di berbagai negara untuk keperluan pengujian keamanan siber dan pencegahan malware (Kanimozhi and Jacob, 2019). Untuk mengakses dataset tersebut, diperlukan pemahaman mengenai layanan *Amazon Web Services (AWS)* karena data disimpan pada sistem bertipe *Amazon S3 Bucket*. Dataset ini dapat diakses melalui *Amazon Resource Name* (ARN) , dengan ketentuan penggunaan mengikuti lisensi yang telah ditetapkan.

Dataset ini mencakup deskripsi rinci mengenai berbagai jenis intrusi beserta model distribusi abstrak yang merepresentasikan aplikasi, protokol, dan entitas jaringan tingkat rendah. Dataset akhir terdiri atas tujuh skenario serangan yang berbeda, yaitu *Brute Force, Heartbleed, Botnet, Denial of Service (DoS), Distributed Denial of Service (DDoS), Web Attacks*, serta *Infiltration*, yaitu penyusupan ke dalam jaringan dari pihak internal. Infrastruktur penyerang yang digunakan dalam simulasi terdiri atas 50 mesin, sedangkan organisasi korban memiliki 5 departemen yang mencakup 420 mesin dan 30 server. Selain itu, dataset ini menyediakan rekaman lalu lintas jaringan (network traffic) dan log sistem dari setiap mesin, serta 80 fitur yang diekstraksi dari lalu lintas jaringan menggunakan *CICFlowMeter*\-V3.  

Fitur-fitur yang terdapat pada dataset berupa,   

Tabel 1.1 Tabel Fitur Dataset CSE-CIC-IDS2018
| Kode Kolom | Deskripsi |
| :---: | :---: |
| fl\_dur | Durasi aliran (*flow*) |
| tot\_fw\_pk | Total paket pada arah maju (*forward*) |
| tot\_bw\_pk | Total paket pada arah mundur (*backward*) |
| tot\_l\_fw\_pkt | Total ukuran paket pada arah maju |
| fw\_pkt\_l\_max | Ukuran maksimum paket pada arah maju |
| fw\_pkt\_l\_min | Ukuran minimum paket pada arah maju |
| fw\_pkt\_l\_avg | Rata-rata ukuran paket pada arah maju |
| fw\_pkt\_l\_std | Standar deviasi ukuran paket pada arah maju |
| bw\_pkt\_l\_max | Ukuran maksimum paket pada arah mundur |
| bw\_pkt\_l\_min | Ukuran minimum paket pada arah mundur |
| bw\_pkt\_l\_avg | Rata-rata ukuran paket pada arah mundur |
| bw\_pkt\_l\_std | Standar deviasi ukuran paket pada arah mundur |
| fl\_byt\_s | Laju *byte* aliran (byte per detik) |
| fl\_pkt\_s | Laju paket aliran (paket per detik) |
| fl\_iat\_avg | Rata-rata waktu antar dua *flow* |
| fl\_iat\_std | Standar deviasi waktu antar *flow* |
| fl\_iat\_max | Waktu maksimum antar *flow* |
| fl\_iat\_min | Waktu minimum antar *flow* |
| fw\_iat\_tot | Total waktu antar paket arah maju |
| fw\_iat\_avg | Rata-rata waktu antar paket arah maju |
| fw\_iat\_std | Standar deviasi waktu antar paket arah maju |
| fw\_iat\_max | Waktu maksimum antar paket arah maju |
| fw\_iat\_min | Waktu minimum antar paket arah maju |
| bw\_iat\_tot | Total waktu antar paket arah mundur |
| bw\_iat\_avg | Rata-rata waktu antar paket arah mundur |
| bw\_iat\_std | Standar deviasi waktu antar paket arah mundur |
| bw\_iat\_max | Waktu maksimum antar paket arah mundur |
| bw\_iat\_min | Waktu minimum antar paket arah mundur |
| fw\_psh\_flag | Jumlah *flag* PSH pada arah maju |
| bw\_psh\_flag | Jumlah *flag* PSH pada arah mundur |
| fw\_urg\_flag | Jumlah *flag* URG pada arah maju |
| bw\_urg\_flag | Jumlah *flag* URG pada arah mundur |
| fw\_hdr\_len | Total *byte* header pada arah maju |
| bw\_hdr\_len | Total *byte* header pada arah mundur |
| fw\_pkt\_s | Jumlah paket per detik arah maju |
| bw\_pkt\_s | Jumlah paket per detik arah mundur |
| pkt\_len\_min | Panjang minimum paket dalam *flow* |
| pkt\_len\_max | Panjang maksimum paket dalam *flow* |
| pkt\_len\_avg | Rata-rata panjang paket dalam *flow* |
| pkt\_len\_std | Standar deviasi panjang paket dalam *flow* |
| pkt\_len\_va | Varians panjang paket dalam *flow*t |
| fin\_cnt | Jumlah paket dengan *flag* FIN |
| syn\_cnt | Jumlah paket dengan *flag* SYN |
| rst\_cnt | Jumlah paket dengan *flag* RST |
| pst\_cnt | Jumlah paket dengan *flag* PSH |
| ack\_cnt | Jumlah paket dengan *flag* ACK |
| urg\_cnt | Jumlah paket dengan *flag* URG |
| cwe\_cnt | Jumlah paket dengan *flag* CWE |
| ece\_cnt | Jumlah paket dengan *flag* ECE |
| down\_up\_ratio | Rasio *download* terhadap *upload* |
| pkt\_size\_avg | Rata-rata ukuran paket |
| fw\_seg\_avg | Rata-rata ukuran segmen arah maju |
| bw\_seg\_avg | Rata-rata ukuran segmen arah mundur |
| fw\_byt\_blk\_avg | Rata-rata byte per *bulk* arah maju |
| fw\_pkt\_blk\_avg | Rata-rata paket per *bulk* arah maju |
| fw\_blk\_rate\_avg | Rata-rata laju *bulk* arah maju |
| bw\_byt\_blk\_avg | Rata-rata byte per *bulk* arah mundur |
| bw\_pkt\_blk\_avg | Rata-rata paket per *bulk* arah mundur |
| bw\_blk\_rate\_avg | Rata-rata laju *bulk* arah mundur |
| subfl\_fw\_pk | Rata-rata paket *subflow* arah maju |
| subfl\_fw\_byt | Rata-rata *byte* *subflow* arah maju |
| subfl\_bw\_pkt | Rata-rata paket *subflow* arah mundur |
| subfl\_bw\_byt | Rata-rata *byte* *subflow* arah mundur |
| fw\_win\_byt | *Byte* pada initial window arah maju |
| bw\_win\_byt | *Byte* pada initial window arah mundur |
| fw\_act\_pkt | Jumlah paket dengan *payload* TCP ≥ 1 *byte* (*forward*) |
| fw\_seg\_min | Ukuran segmen minimum arah maju |
| atv\_avg | Rata-rata waktu *flow* aktif |
| atv\_std | Standar deviasi waktu *flow* aktif |
| atv\_max | Waktu maksimum *flow* aktif |
| atv\_min | Waktu minimum *flow* aktif |
| idl\_avg | Rata-rata waktu *flow idle* |
| idl\_std | Standar deviasi waktu *flow idle* |
| idl\_max | Waktu maksimum *flow idle* |
| idl\_min | Waktu minimum *flow idle* |
| label | Jenis kelas data |