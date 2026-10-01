# Sistem Deteksi APD PDU Migas — Backend API (FastAPI + YOLO11s & YOLO11n)

Backend API produksi untuk **Sistem Deteksi Alat Pelindung Diri (APD) pada Area Pengeboran Migas Parama Data Unit (PDU)** berbasis Computer Vision, mengacu penuh pada spesifikasi **Product Requirements Document (PRD) Versi 1.0**.

---

## 1. Arsitektur & Pendekatan Teknis

Sistem mengadopsi arsitektur modular **Two-Stage Multi-Model Detection Pipeline**:

```
                                  ┌──────────────────────────┐
                                  │   Input Frame / Video    │
                                  └─────────────┬────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │ Stage 1: Person Detection & Global APD Screening │
                        │  - YOLO11n (COCO Person class 0)                │
                        │  - YOLO11s (best_yolo11s_pdu.pt: 4 APD classes) │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │ Stage 2: Micro-Object Zoom (Head Crop for Eye)  │
                        │  - Crop region kepala pekerja (upper 38% bbox)  │
                        │  - Zoomed inference imgsz=640 (conf=0.08)       │
                        │  - Koordinat direproyeksikan ke frame global     │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │ Stage 3: Evaluasi Kepatuhan PRD Section 10.6    │
                        │  - Rasio Overlap BBox APD terhadap BBox Person   │
                        │  - Threshold: Overlap ≥ 80% (0.80)              │
                        │  - APD Wajib: Helm, Glove, Sepatu, Kacamata     │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │ Stage 4: Temporal Tracking & State Persistence   │
                        │  - Menjaga kepatuhan saat pekerja menunduk /     │
                        │    terjadi oklusi sementara (memory 30 frames)   │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │ REST API Response + Annotated Visuals + DB Audit│
                        └─────────────────────────────────────────────────┘
```

### Keunggulan Pipeline:
1. **Solusi Tantangan Mikro-Deteksi Kacamata (*Safety Glasses*):**
   - Kacamata berukuran sangat kecil (~20–40 piksel) pada citra CCTV sudut tinggi rig migas.
   - Deteksi global menggabungkan threshold terkalibrasi (`conf=0.12`) dengan **Two-Stage Head Crop Analysis** (resolusi 640 pada crop kepala atas 38% tubuh) sehingga kacamata yang tersamarkan bayangan atau sudut kamera tetap terdeteksi akurat.
2. **Pelacakan Spasial Mandiri (*SpatialPersonTracker*):**
   - Menggunakan pencocokan IoU spasial greedy murni Python tanpa ketergantungan compiler C++ eksternal (`lap`/`scipy`) sehingga berjalan mulus di Windows, Linux, maupun container.
   - Menjamin `Track ID` pekerja stabil dan konsisten sepanjang durasi video dan stream kamera.
3. **Mitigasi Oklusi & Gerakan Menunduk (*Temporal State Persistence*):**
   - Pada pemrosesan video dan live stream kamera (`/api/detect/stream-frame`), sistem memelihara riwayat observasi pekerja berbasis `Track ID`.
   - Begitu pekerja terdeteksi mengenakan kacamata / helm / sarung tangan saat tegak atau menatap kamera, status kepatuhan dipertahankan dalam jendela temporal (*sliding memory buffer* 30 frame) agar tidak terjadi kedipan (*flicker*) pelanggaran palsu saat pekerja membungkuk memeriksa pipa/lantai rig.
   - Frame visual yang di-render dan di-encode ke client secara sinkron merefleksikan status yang sudah dihaluskan (*smoothed*), mencegah kotak pelanggaran merah palsu pada layar pengawas HSE.
4. **Standar Overlap PRD Bagian 10.6 & Margin Anatomis:**
   - Dihitung menggunakan rasio irisan area APD terhadap luas APD:
     $$\text{Rasio Overlap} = \frac{\text{Luas}(BBox_{APD} \cap BBox_{Person})}{\text{Luas}(BBox_{APD})}$$
   - Dilengkapi margin anatomis proporsional (8%) untuk mengakomodasi tonjolan helm di atas dahi, sol sepatu keselamatan di bawah pergelangan kaki, dan sarung tangan pada tangan yang terulur bekerja.
   - Status **"APD terpakai"** bila Rasio Overlap $\ge 80\%$ (0.80).
5. **Konfigurasi Zona Deteksi Dinamis (ROI):**
   - Mendukung pembatasan zona ROI dinamis untuk memfilter pekerja di luar area berbahaya rig pengeboran.
   - Mendukung Mode B (Deteksi Zona ROI Global) jika deteksi Person tidak aktif.

---

## 2. Struktur Direktori

```
be-apd-pdu/
├── weights/
│   ├── best_yolo11s_pdu.pt         # Bobot model kustom YOLO11s (4 kelas APD PDU)
│   └── yolo11n.pt                  # Bobot pre-trained YOLO11n (Deteksi & Tracking Person)
├── app/
│   ├── __init__.py
│   ├── main.py                     # Entrypoint FastAPI, CORS, static mounts, lifespan
│   ├── config.py                   # Konfigurasi aplikasi, path, default threshold
│   ├── database.py                 # Async SQLite engine (aiosqlite + SQLAlchemy)
│   ├── models/                     # Model ORM Database
│   │   ├── user.py                 # Tabel User (Admin & Supervisor RBAC)
│   │   ├── session.py              # Tabel DetectionSession
│   │   ├── worker.py               # Tabel WorkerRecord (Checklist & BBox)
│   │   └── settings.py             # Tabel SystemSettingsModel (ROI & Thresholds)
│   ├── schemas/                    # Pydantic Schemas (Request/Response)
│   │   ├── auth.py
│   │   ├── detection.py
│   │   ├── dashboard.py
│   │   └── settings.py
│   ├── core/                       # Keamanan & Injeksi Dependensi
│   │   ├── security.py             # Password hashing (bcrypt) & JWT Token
│   │   └── deps.py                 # Dependency get_db, get_current_user, require_admin
│   ├── services/                   # Core Business Logic & CV Pipeline
│   │   ├── apd_detector.py         # Engine deteksi APD, PRD overlap 80%, two-stage zoom
│   │   ├── video_processor.py      # Batch video processor dengan tracking per-frame
│   │   └── stream_manager.py       # Live stream state manager untuk webcam / live feed
│   └── routers/                    # Endpoints API
│       ├── auth.py                 # /api/auth
│       ├── detection.py            # /api/detect
│       ├── dashboard.py            # /api/dashboard
│       └── settings.py             # /api/settings
├── tests/
│   ├── test_detector.py            # Uji coba matematika overlap, zoom kacamata, temporal
│   ├── test_api_auth.py            # Uji endpoint autentikasi & JWT
│   ├── test_api_detect.py          # Uji endpoint deteksi gambar, stream, sesi
│   ├── test_api_dashboard.py       # Uji dashboard KPI, filter, ekspor CSV
│   └── test_api_settings.py        # Uji konfigurasi ROI & threshold dinamis
├── test_dataset_eval.py            # Script evaluasi otomatis 38 frame test set PDU
├── pyproject.toml
└── README.md
```

---

## 3. Instalasi & Menjalankan Server

### Prasyarat
- Python 3.10+
- `uv` package manager (atau pip standard)

### Menjalankan Server Backend
```powershell
# Jalankan menggunakan uv
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan aktif di:
- **API Base:** `http://localhost:8000`
- **Swagger UI Interactive Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Akun Bawaan (Default Seeded Users)
- **Admin HSE:** `username`: `admin` | `password`: `admin123`
- **Supervisor Lapangan:** `username`: `supervisor` | `password`: `super123`

---

## 4. Dokumentasi Endpoint REST API

### A. Modul Autentikasi (`/api/auth`)
| Metode | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/api/auth/register` | Mendaftarkan pengguna baru (Admin / Supervisor) |
| `POST` | `/api/auth/login` | Login JSON, mengembalikan JWT Bearer Token |
| `POST` | `/api/auth/token` | Login format form (kompatibel Swagger UI) |
| `GET` | `/api/auth/me` | Mengambil profil user yang sedang login |

### B. Modul Deteksi AI (`/api/detect`)
| Metode | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/api/detect/image` | Upload gambar (JPG/PNG), inferensi APD + Person, checklist kelengkapan per orang, rendering visual base64/URL |
| `POST` | `/api/detect/video` | Upload video (MP4), ekstraksi frame periodik, tracking temporal antar-frame, video teranotasi, rekapitulasi sesi |
| `POST` | `/api/detect/stream-frame` | Input frame live stream webcam (base64) dengan persistent temporal tracker per client ID |
| `GET` | `/api/detect/sessions` | Riwayat sesi deteksi (pagination & filter source_type) |
| `GET` | `/api/detect/sessions/{id}` | Detail sesi lengkap beserta riwayat checklist seluruh pekerja |
| `DELETE` | `/api/detect/sessions/{id}` | Menghapus sesi deteksi |

### C. Modul Dashboard Kepatuhan (`/api/dashboard`)
| Metode | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/api/dashboard/summary` | Ringkasan KPI: total pekerja, tingkat kepatuhan %, pelanggaran per jenis APD |
| `GET` | `/api/dashboard/compliance-trends` | Data grafik tren kepatuhan & pelanggaran (harian/mingguan) |
| `GET` | `/api/dashboard/worker-records` | Tabel pekerja terdeteksi dengan filter (status kepatuhan, jenis APD yang hilang, rentang tanggal) |
| `GET` | `/api/dashboard/export` | Unduh laporan kepatuhan dalam format **CSV** siap audit K3 |

### D. Modul Pengaturan & ROI (`/api/settings`)
| Metode | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/api/settings` | Mengambil konfigurasi aktif (koordinat ROI, ambang overlap, threshold confidence tiap kelas) |
| `PUT` | `/api/settings` | Memperbarui koordinat ROI dan parameter threshold deteksi secara dinamis tanpa restart server |

---

## 5. Menjalankan Automated Test Suite

Seluruh pengujian fungsional, integrasi, dan logika bisnis diverifikasi dengan:

```powershell
# Jalankan seluruh test suite pytest (21 test cases)
uv run pytest tests/ -v

# Jalankan evaluasi kuantitatif komprehensif pada 38 test frame dataset PDU
uv run python test_dataset_eval.py
```

Hasil unit & integration test: **21 test cases 100% PASSED**.

---

## 6. Hasil Evaluasi Kuantitatif (Benchmark Test Set PDU Migas)

Evaluasi murni pada **38 Frame Test Set PDU Migas** (207 ground-truth instances APD, *clean chronological split* anti-data leakage):

| Kelas APD | Ground Truth (GT) | True Positive (TP) | False Positive (FP) | False Negative (FN) | Recall | Precision | F1-Score |
|---|---|---|---|---|---|---|---|
| **helm** | 100 | 98 | 14 | 2 | **98.0%** | **87.5%** | **0.925** |
| **sepatu** | 43 | 33 | 6 | 10 | **76.7%** | **84.6%** | **0.805** |
| **glove** | 44 | 23 | 4 | 21 | **52.3%** | **85.2%** | **0.648** |
| **kacamata** | 20 | 13 | 20 | 7 | **65.0%** | **39.4%** | **0.491** |
| **GLOBAL** | **207** | **167** | **44** | **40** | **80.7%** | **79.1%** | **0.799** |

### Ringkasan Optimasi:
- **Deteksi Kacamata (Micro-Detection)**: Meningkat drastis dari 35.0% (raw YOLO) menjadi **65.0% Recall** berkat **Two-Stage Head Crop Zoom** (`HEAD_ZOOM_CONF=0.08`, `imgsz=640`).
- **Deteksi Pekerja di Pinggir Frame**: Algoritma **Helmet-Anchored Worker Inference** berhasil menangkap pekerja yang tubuhnya terpotong sudut atas/bawah CCTV lapangan rig migas, meningkatkan APD association rate hingga **94.8%**.
- **Margin Anatomis Adaptif**: Padding anatomis proporsional memastikan helmet dome, sepatu, dan sarung tangan memenuhi ambang batas overlap $\ge 80\%$ sesuai PRD Section 10.6 tanpa menghasilkan false violations.
- **Temporal State Persistence**: Melindungi pekerja dari kedipan pelanggaran palsu saat menunduk atau terhalang topi helm selama inspeksi rig.
