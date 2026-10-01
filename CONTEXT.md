# CONTEXT & DEVELOPER GUIDE: SISTEM DETEKSI APD PDU MIGAS (BACKEND)

> **Dokumen Panduan Konteks untuk Developer & AI Agent yang Melanjutkan Repository Ini**  
> *Versi: 1.0.0 | Terakhir Diperbarui: Oktober 2026*  
> *Repository: [be-apd-pdu](https://github.com/akmalmanggala/be-apd-pdu.git)*

---

## 1. Ringkasan Eksekutif & Misi Proyek

Repository ini adalah backend berbasis **FastAPI, PyTorch, dan Ultralytics YOLO11** untuk **Sistem Pemantauan dan Evaluasi Kepatuhan Alat Pelindung Diri (APD) Pekerja Rig PDU (Power Drill Unit / Drilling Rig Floor)** di industri minyak dan gas bumi (*Oil & Gas*).

Sistem ini bertugas mendeteksi keberadaan dan kepatuhan 4 jenis APD wajib rig secara otomatis dan *real-time*:
1. **`helm`** (*Safety Hard Hat*)
2. **`kacamata`** (*Safety Glasses / Goggles*)
3. **`glove`** (*Heavy-Duty Safety Gloves*)
4. **`sepatu`** (*Steel-Toe Safety Boots*)

---

## 2. Tantangan Domain Lapangan Minyak & Gas (Rig Floor CCTV)

Kamera pengawas CCTV di lantai rig pengeboran (*drill floor*) memiliki karakteristik ekstrem yang membedakannya secara fundamental dari dataset pedestrian jalanan standar (seperti COCO):

1. **Sudut Kamera CCTV dari Atas (*High-Angle / Overhead Rig POV*)**:
   - Model detektor orang biasa (*COCO Person Detector* seperti YOLO11n/s) sering **gagal 100%** mendeteksi manusia karena hanya melihat bagian atas helm bundar dan pundak pekerja, atau membingungkan mesin rig vertikal sebagai tubuh orang.
2. **Latar Belakang Mesin Kompleks & Beroda (*Heavy Industrial Clutter*)**:
   - Dial indikator bulat (*weight indicator gauge*), klem pipa (*tong drill pipe*), meja putar (*rotary table*), clipboard catatan, dan tumpukan katup sering memicu deteksi palsu (*hallucination / false positive*).
3. **Tantangan Deteksi Mikro Kacamata (*Micro-Detection Challenge*)**:
   - Kacamata pelindung transparan/bening berukuran sangat kecil (< 1.5% resolusi frame), sering tertutup bayangan pinggiran helm, atau terhalang saat pekerja membelakangi kamera.
4. **Flickering & Pertukaran Kelas Cepat (*Glitch / Class Swapping*)**:
   - Gerakan cepat tangan dan kaki roughneck saat menangani pipa dapat menyebabkan kotak deteksi putus 1 frame (*flickering*), atau warna sol sepatu di lantai basah tertukar menjadi glove.
5. **Deteksi Ganda pada Saku Celana Kargo (*Cargo Pants Pocket False Positives*)**:
   - Lipatan kain tebal wearpack di bagian paha/kantong celana sering terdeteksi sebagai sarung tangan ke-3 pada 1 pekerja.

---

## 3. Arsitektur & Inovasi Rekayasa (*Engineered Solutions*)

Seluruh logika rekayasa cerdas ditempatkan di [`app/services/video_stabilizer.py`](file:///be-apd-pdu/app/services/video_stabilizer.py) dan [`app/services/apd_detector.py`](file:///be-apd-pdu/app/services/apd_detector.py):

### A. Dual-Anchor Human Association (`WorkerAnatomicalContainer`)
- **Prinsip**: Jangan pernah menggantungkan keberadaan manusia hanya pada detektor orang COCO.
- **Solusi**: Helm (`helm` $\ge 0.22$) adalah jangkar (*anchor*) manusia yang paling akurat (mAP50 98.8%). Sistem memproyeksikan amplop tubuh pekerja ke bawah dari helm, lalu melakukan fusi dengan bounding box COCO person yang valid ($\ge 0.50$). Deteksi person hantu pada pipa vertikal yang ber-confidence rendah ($< 0.50$) atau tumpang tindih secara otomatis ditolak.

### B. Tight Worker Bounding Box (*Kotak Pekerja Presisi*)
- Kotak pekerja membungkus tubuh dan ekstremitas secara ketat (*tight box*) berdasarkan posisi kepala, tangan, dan sepatu.
- **Pembersihan Mesin**: Segala deteksi APD yang berada di luar jangkauan anatomis pekerja (misal glove palsu di tiang bor atau katup) langsung **dihapus**.

### C. Anatomical Quota & Bilateral Glove Pairing
- **Kuota Anatomis Wajib per Pekerja**:
  - Helm: Maksimal 1 (area kepala).
  - Kacamata: Maksimal 1 (area wajah).
  - Sepatu: Maksimal 2 (area kaki).
- **Bilateral Glove Pairing (Anti Saku Celana)**:
  - Jika terdeteksi $\ge 3$ sarung tangan pada satu pekerja, sistem **tidak** memilih hanya berdasarkan confidence score tertinggi (karena lipatan celana bisa memiliki confidence tinggi).
  - Sistem mengevaluasi kesimetrisan lateral kiri dan kanan:
    $$\text{Score}(g_1, g_2) = \text{conf}(g_1) + \text{conf}(g_2) + \text{Bonus}(\text{Bilateral}) + 0.3 \times \frac{|x_1 - x_2|}{\text{lebar body}}$$
  - Pasangan tangan kiri-kanan terpilih, sementara sarung tangan palsu di saku tengah celana dibuang.

### D. Stabilisasi Video Temporal Zero-Flicker (`APDVideoStabilizer`)
- **Linear Velocity Extrapolation**:
  - Jika objek terkonfirmasi hilang sesaat (1 frame dropout karena perubahan sudut cahaya atau gerakan cepat), posisinya diekstrapolasi maju sebesar vektor kecepatan:
    $$\vec{p}_{t} = \vec{p}_{t-1} + \vec{v}, \quad \text{dengan clamping } \|\vec{v}\| \le 18\text{ px/frame}$$
  - Mencegah kotak berkedip (*kedip-kedip*) tanpa menyebabkan lag atau tertinggal di belakang objek.
- **Sticky Class Locking (Hysteresis Margin 25%)**:
  - Tracklet yang sudah terkonfirmasi $\ge 3$ frame mengunci kelasnya. Kelas penantang baru harus mengungguli skor kelas terpasang sebesar minimal 25% untuk dapat mengubah kelas (mencegah glove tiba-tiba berkedip menjadi helm).
- **Motion-Based Static Fixture Suppression**:
  - Deteksi statis pada mesin/dinding rig yang tidak pernah bergerak selama puluhan frame otomatis disupresi tanpa pernah menyupresi pekerja yang sedang duduk diam.

### E. Memori Kepatuhan 90 Frame (*Toleransi Menunduk*)
- Saat pekerja menunduk (*bending over*) untuk menyetel slip rotary atau membelakangi kamera, kacamata atau sepatu mungkin terhalang badan sendiri (*self-occlusion*).
- Sistem menyimpan riwayat kepatuhan selama **90 frame** (sekitar 6 detik pada 15 FPS). Pekerja yang sebelumnya terdeteksi lengkap **tetap berstatus `LENGKAP (100%)`** dan tidak terkena penalti pelanggaran palsu.

---

## 4. Struktur Direktori Repository

```text
be-apd-pdu/
├── app/
│   ├── config.py              # Konfigurasi aplikasi, threshold kelas, path model
│   ├── main.py                # Inisialisasi FastAPI, middleware CORS, router
│   ├── core/
│   │   ├── database.py        # SQLAlchemy async SQLite session & engine
│   │   ├── security.py        # Hashing password bcrypt & pembuatan token JWT
│   │   └── events.py          # Startup & shutdown handlers
│   ├── models/                # Schema database ORM (User, DetectionSession, APDViolation)
│   ├── routers/
│   │   ├── auth.py            # Login, register, token refresh
│   │   ├── detect.py          # Endpoint upload gambar tunggal & frame streaming
│   │   ├── dashboard.py       # Statistik analitik kepatuhan APD
│   │   └── settings.py        # Pengaturan threshold dinamis & ROI
│   ├── schemas/               # Pydantic schema request & response
│   └── services/
│       ├── apd_detector.py    # Singleton CV Engine & visual annotator
│       ├── video_stabilizer.py# Temporal Stabilizer & WorkerAnatomicalContainer
│       ├── video_processor.py # Pemrosesan batch file video
│       └── stream_manager.py  # Handler live streaming RTSP CCTV
├── weights/
│   ├── best_yolo11s_pdu.pt    # Bobot model APD aktif (Tahap 7 Active Learning)
│   ├── yolo11s.pt / yolo11n.pt# Bobot pre-trained detektor orang
│   └── best_yolo11s_pdu_stage*.pt # Bobot historis checkpoint tahapan pelatihan
├── tests/                     # 32 Unit tests lengkap (100% pass)
├── outputs/                   # Folder output visual (bersih, hanya .keep)
├── uploads/                   # Folder upload file sementara (hanya .keep)
├── test_dataset_eval.py       # Script evaluasi kuantitatif dataset test Roboflow
├── pyproject.toml             # Konfigurasi dependensi project (uv / pip)
├── .gitignore                 # Exclude virtualenv, db, outputs, runs, pycache
├── README.md                  # Dokumentasi API umum
└── CONTEXT.md                 # Dokumen ini (Panduan Konteks Developer & AI)
```

---

## 5. Cara Menjalankan & Menjalankan Uji Coba

### A. Setup Lingkungan (*Environment*)
Disarankan menggunakan **Python 3.10**:
```bash
# Menggunakan venv standar
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Instalasi dependensi
pip install -e .
# atau jika menggunakan uv:
uv sync
```

### B. Menjalankan Server Backend
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Dokumentasi Swagger UI: `http://localhost:8000/docs`
- Dokumentasi Redoc: `http://localhost:8000/redoc`

### C. Menjalankan Seluruh Unit Test
```bash
pytest tests/
```
*Status saat ini:* **32 passed** (Meliputi pengujian Auth, API Detect, Settings, Dashboard, APD Detector Engine, dan Video Stabilizer).

### D. Menjalankan Evaluasi Dataset Uji (38 Frame)
```bash
python test_dataset_eval.py
```

---

## 6. Threshold Kalibrasi Saat Ini (`app/config.py`)

Nilai threshold ini adalah hasil kalibrasi optimal dari iterasi pengujian video uji lapangan:
- **`helm`**: `0.25` (Anchor manusia andal; worker container menggunakan `0.22` agar mencakup pekerja duduk).
- **`kacamata`**: `0.15` (Sangat sensitif terhadap kacamata bening di bawah bayangan helm tanpa memicu ilusi lipatan kerah).
- **`glove`**: `0.20` (Menangkap sarung tangan kerja pada tuas kontrol, dengan penolakan saku via *Bilateral Pairing*).
- **`sepatu`**: `0.22` (Menangkap sepatu bot baja di atas lantai rig basah/berminyak).
- **`DEFAULT_PERSON_CONF`**: `0.50` (Mencegah COCO mendeteksi pipa vertikal sebagai orang hantu).

---

## 7. Roadmap & Pekerjaan yang Dapat Dilanjutkan

Jika Anda (atau AI rekan kerja) ingin melanjutkan pengembangan sistem ini, berikut adalah daftar prioritas berikutnya:

1. **Integrasi WebSocket Live CCTV Stream**:
   - Manfaatkan [`app/services/stream_manager.py`](file:///be-apd-pdu/app/services/stream_manager.py) untuk menyalurkan deteksi frame demi frame dari kamera IP/RTSP langsung ke frontend React/Next.js via WebSocket.
2. **Koneksi Frontend Dashboard**:
   - Endpoint analytics di [`app/routers/dashboard.py`](file:///be-apd-pdu/app/routers/dashboard.py) sudah siap menyajikan metrik kepatuhan harian/mingguan, laju pelanggaran per shift, dan jenis APD yang paling sering dilanggar.
3. **Penyimpanan Log Pelanggaran Otomatis**:
   - Implementasikan *background task* (Celery atau FastAPI BackgroundTasks) untuk menyimpan snapshot foto pekerja yang melanggar beserta catatan waktu dan ID kamera ke tabel database `apd_violations`.
4. **Export Laporan PDF K3 (HSE)**:
   - Buat endpoint pelaporan mingguan otomatis yang merangkum persentase kepatuhan APD untuk audit keselamatan kerja rig.
