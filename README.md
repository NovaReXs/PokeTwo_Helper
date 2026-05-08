# Pokétwo Auto-Catcher & Helper Bot 🎮

Bot otomatis berbasis AI untuk mengidentifikasi dan menangkap Pokémon dari bot Pokétwo di Discord. Menggunakan model *Machine Learning* lokal (`skshmjn/Pokemon-classifier-gen9-1025`) yang dapat mendeteksi semua 1025 Pokémon (Generasi 1 - 9) dengan akurasi tinggi dan tanpa perlu API key pihak ketiga.

> ⚠️ **PERINGATAN (Disclaimer):**
> Penggunaan **Self-Bot** melanggar *Terms of Service (ToS)* dari Discord. Bot ini murni untuk tujuan edukasi dan eksperimen. **Gunakan di akun alternatif / akun bot cadangan Anda dengan risiko ditanggung sendiri (Do It At Your Own Risk).**

---

## ✨ Fitur Utama

- **Deteksi Otomatis & Akurat:** Mengunduh gambar spawn Pokémon dan memprosesnya secara lokal menggunakan model AI Vision Transformer.
- **Dua Mode Channel:**
  - **Auto-Catch Mode:** Bot menangkap Pokémon secara otomatis dengan jeda waktu acak (*human-like delay*).
  - **Helper Mode:** Bot tidak menangkap Pokémon, melainkan mengirimkan nama Pokémon ke chat agar Anda/teman Anda bisa menangkapnya.
- **Anti-Bot / CAPTCHA Protection:** Mampu mendeteksi peringatan CAPTCHA dari Pokétwo. Jika terdeteksi, bot akan otomatis melakukan *PAUSE* dan membunyikan alarm di terminal untuk mencegah *ban*.
- **Notifikasi Pokémon Langka (Rare Ping):** Jika Legendary, Mythical, Ultra Beast, atau Pseudo-Legendary muncul, bot akan melakukan `ping/tag` pengguna atau *role* khusus di Discord.
- **Sistem Perintah Langsung (Commands):** Kontrol bot langsung dari Discord menggunakan command seperti `!start`, `!stop`, dll.

---

## 🛠️ Persyaratan Instalasi

Pastikan Anda sudah menginstal **Python 3.10+** di sistem Anda.

1. Buka Terminal / Command Prompt di folder proyek ini.
2. Instal semua dependensi yang dibutuhkan menggunakan `pip`:
   ```bash
   pip install discord.py-self aiohttp transformers torch torchvision pillow python-dotenv
   ```

---

## ⚙️ Konfigurasi (Setup)

Sebelum menjalankan bot, Anda perlu menyesuaikan beberapa pengaturan.

1. **DISCORD_TOKEN (.env file)**
   Token akun Discord Anda tidak boleh disebar. Buat file bernama `.env` di folder yang sama dengan script (Anda bisa mengopi dari `.env.example`).
   Isi file `.env` dengan token Anda:
   ```env
   DISCORD_TOKEN=ISI_TOKEN_DISCORD_ANDA_DI_SINI
   ```
   *(Ingat: JANGAN bagikan token ini kepada siapa pun!)*

2. **Channel Auto-Catch & Helper (`pokemon_autocatch.py`)**
   Buka file `pokemon_autocatch.py` menggunakan Text Editor (seperti VSCode/Notepad). Atur ID dari Channel Discord tempat spawn Pokémon terjadi di bagian atas file:
   - `WATCHED_CHANNEL_IDS_1`: Channel untuk mode *Auto-Catch*
   - `WATCHED_CHANNEL_IDS_2`: Channel untuk mode *Helper*
   *(Anda bisa mengubah pengaturan ini kapan saja melalui Commands di Discord).*

3. **Notifikasi Langka (Rare Ping ID)**
   Masukkan ID User atau Role yang ingin Anda *tag* ketika Pokémon langka muncul.
   ```python
   # Contoh Tag 1 User
   RARE_PING_ID = "<@273351260273508352>"
   
   # Contoh Tag Banyak User
   RARE_PING_ID = "<@273351260273508352>", "<@406091992032870401>"
   ```

4. **Delay Penangkapan**
   Atur jeda waktu bot membalas pesan (dalam detik). Berguna untuk membuat bot terlihat seperti manusia.
   ```python
   CATCH_DELAY_MIN = 1.0
   CATCH_DELAY_MAX = 2.0
   ```

---

## 🚀 Cara Penggunaan

1. Jalankan script menggunakan terminal/CMD:
   ```bash
   python pokemon_autocatch.py
   ```
2. Pada pertama kali berjalan, bot akan **mengunduh model AI** secara otomatis (ukuran ratusan MB). Proses ini hanya terjadi satu kali.
3. Tunggu hingga terminal menampilkan pesan `[OK] Logged in as: NamaAkunAnda`.
4. Jika muncul peringatan *"Warning: You are sending unauthenticated requests to the HF Hub"*, hal tersebut **wajar** dan bisa diabaikan.
5. Bot sudah siap dan otomatis bekerja saat ada Pokétwo spawn!

---

## 🎮 Daftar Perintah (Commands)

Anda dapat mengontrol bot dengan mengetikkan perintah berikut di chat Discord (Bot hanya akan merespon jika dikirim oleh akun bot itu sendiri):

- `!stop` : Menghentikan sementara bot (*PAUSE*).
- `!start` : Melanjutkan kembali bot (*RESUME*).
- `!status` : Menampilkan status bot saat ini (Berjalan/Pause dan Mode masing-masing list) di terminal.
- `!catch1` : Mengubah Channel List 1 menjadi mode **Auto-Catch**.
- `!helper1` : Mengubah Channel List 1 menjadi mode **Helper**.
- `!catch2` : Mengubah Channel List 2 menjadi mode **Auto-Catch**.
- `!helper2` : Mengubah Channel List 2 menjadi mode **Helper**.
- `!catch` : Mengubah **semua** Channel List menjadi mode Auto-Catch.
- `!helper` : Mengubah **semua** Channel List menjadi mode Helper.

---

## 🛑 Penanganan CAPTCHA

Apabila bot mendeteksi peringatan:
> *"Whoa there. Please tell us you're human! https://verify.poketwo.net/captcha/..."*

Bot akan berbunyi `beep` di PC Anda dan men-stop dirinya sendiri (`PAUSE`). 
**Yang harus Anda lakukan:**
1. Klik link CAPTCHA tersebut.
2. Selesaikan *puzzle* secara manual.
3. Kembali ke Discord, ketik `!start` untuk menjalankan bot kembali.
