import sqlite3
import hashlib
from datetime import datetime
import pytz 
import os 

DATABASE_FILE = "users.db"

def _get_db_connection():
    """Mengembalikan objek koneksi database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row 
    return conn

def init_db():
    """
    Menginisialisasi database: membuat tabel 'users' dan 'detection_history' jika belum ada.
    """
    with _get_db_connection() as conn: 
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hashed TEXT NOT NULL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp TEXT NOT NULL, -- Simpan waktu sebagai TEXT ISO format
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT, -- Menyimpan PATH ke file gambar di folder /uploads
                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
            )
        ''')
        conn.commit()
    print(f"Database '{DATABASE_FILE}' siap. Tabel 'users' dan 'detection_history' telah dibuat atau sudah ada.")

def hash_password(password: str) -> str:
    """
    Menghasilkan hash SHA256 dari password.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username: str, password: str) -> bool:
    """
    Menambahkan pengguna baru ke tabel 'users'.
    Mengembalikan True jika berhasil, False jika username sudah ada.
    """
    with _get_db_connection() as conn: 
        c = conn.cursor()
        try:
            hashed_pw = hash_password(password)
            c.execute("INSERT INTO users (username, password_hashed) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            print(f"Pengguna '{username}' berhasil ditambahkan.")
            return True
        except sqlite3.IntegrityError: 
            print(f"Error: Username '{username}' sudah ada.")
            return False
        except Exception as e:
            print(f"Error menambahkan pengguna: {e}")
            return False

def verify_user(username: str, password: str) -> bool:
    """
    Memverifikasi kredensial pengguna dari tabel 'users'.
    """
    with _get_db_connection() as conn: 
        c = conn.cursor()
        c.execute("SELECT password_hashed FROM users WHERE username = ?", (username,))
        result = c.fetchone()

    if result:
        stored_hashed_pw = result[0]
        return stored_hashed_pw == hash_password(password)
    return False

def add_detection_record(username: str, disease_name: str, confidence: float, image_path: str = None) -> bool:
    """
    Menambahkan catatan hasil deteksi ke tabel 'detection_history'.
    image_path adalah jalur ke file gambar di sistem file (folder 'uploads').
    """
    with _get_db_connection() as conn:
        c = conn.cursor()
        try:
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            current_utc_time = datetime.utcnow()
            current_jakarta_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(jakarta_tz)
            timestamp_str = current_jakarta_time.strftime("%Y-%m-%d %H:%M:%S") 

            c.execute("INSERT INTO detection_history (username, timestamp, disease_name, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
                      (username, timestamp_str, disease_name, confidence, image_path))
            conn.commit()
            print(f"Deteksi '{disease_name}' oleh '{username}' pada {timestamp_str} (Gambar: {image_path}) berhasil disimpan.")
            return True
        except Exception as e:
            print(f"Error menyimpan riwayat deteksi: {e}")
            return False

def get_detection_history(username: str) -> list:
    """
    Mengambil semua hasil deteksi untuk pengguna tertentu dari tabel 'detection_history'.
    Mengembalikan list dari dict (id, timestamp, disease_name, confidence, image_path).
    """
    with _get_db_connection() as conn:
        c = conn.cursor()
        try:
            c.execute("SELECT id, timestamp, disease_name, confidence, image_path FROM detection_history WHERE username = ? ORDER BY timestamp DESC", (username,))
            history = [dict(row) for row in c.fetchall()] 
            return history
        except Exception as e:
            print(f"Error mengambil riwayat deteksi: {e}")
            return []

def delete_detection_record(record_id: int) -> bool:
    """
    Menghapus catatan deteksi dari tabel 'detection_history' berdasarkan ID.
    """
    with _get_db_connection() as conn: 
        c = conn.cursor()
        try:
            c.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
            conn.commit()
            print(f"Catatan deteksi dengan ID {record_id} berhasil dihapus.")
            return True
        except Exception as e:
            print(f"Error menghapus catatan deteksi: {e}")
            return False
