import sqlite3
import json
import uuid
import hashlib
import datetime
import numpy as np

DB_FILE = "pivac.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Permite acceder a las columnas por nombre
    return conn

def init_db():
    """
    Crea la tabla 'vacunos' si no existe.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tabla de Usuarios
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nombre_operador TEXT NOT NULL,
            fecha_creacion TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vacunos (
            id TEXT PRIMARY KEY,
            establecimiento_id TEXT NOT NULL,
            id_vacuno_interno TEXT NOT NULL,
            numero_caravana TEXT,
            morro_vector TEXT NOT NULL,      -- Guardado como lista JSON
            morro_encriptado TEXT NOT NULL,  -- Cifrado AES-256
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            fecha_registro TEXT NOT NULL
        )
    """)

        # Crear un usuario por defecto si no existe (admin / pivac2026)
    cursor.execute("SELECT * FROM usuarios WHERE username = 'admin'")
    if not cursor.fetchone():
        pass_hash = hashlib.sha256("pivac2026".encode()).hexdigest()
        cursor.execute("""
            INSERT INTO usuarios (id, username, password_hash, nombre_operador, fecha_creacion)
            VALUES (?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), "admin", pass_hash, "Operador Campo", datetime.datetime.utcnow().isoformat()))
        print("👤 Usuario 'admin' creado por defecto (Clave: pivac2026).")

    conn.commit()
    conn.close()

def save_vacuno(
    establecimiento_id: str,
    id_vacuno_interno: str,
    numero_caravana: str,
    morro_vector: list,
    morro_encriptado: str,
    latitude: float,
    longitude: float
) -> dict:
    """
    Inserta un nuevo vacuno en SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    vacuno_id = str(uuid.uuid4())
    vector_json = json.dumps(morro_vector)
    fecha_iso = datetime.datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO vacunos (
            id, establecimiento_id, id_vacuno_interno, numero_caravana,
            morro_vector, morro_encriptado, latitude, longitude, fecha_registro
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        vacuno_id,
        establecimiento_id,
        id_vacuno_interno,
        numero_caravana,
        vector_json,
        morro_encriptado,
        latitude,
        longitude,
        fecha_iso
    ))

    conn.commit()
    conn.close()

    return {
        "id": vacuno_id,
        "id_vacuno_interno": id_vacuno_interno,
        "numero_caravana": numero_caravana
    }

def verify_user(username: str, password_plain: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    pass_hash = hashlib.sha256(password_plain.encode()).hexdigest()
    
    cursor.execute("SELECT id, username, nombre_operador FROM usuarios WHERE username = ? AND password_hash = ?", (username, pass_hash))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def find_nearest_vacuno(query_vector: np.ndarray, threshold: float = 0.85):
    """
    Realiza la búsqueda k-NN calculando la similitud coseno en memoria usando NumPy.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vacunos")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, 0.0

    best_match = None
    max_similitud = -1.0

    norm_query = np.linalg.norm(query_vector)
    if norm_query == 0:
        return None, 0.0

    for row in rows:
        # Convertir el JSON string de nuevo a NumPy array
        db_vector = np.array(json.loads(row["morro_vector"]), dtype=np.float32)
        norm_db = np.linalg.norm(db_vector)

        if norm_db > 0:
            # Similitud Coseno: (A · B) / (||A|| * ||B||)
            similitud = float(np.dot(query_vector, db_vector) / (norm_query * norm_db))

            if similitud > max_similitud:
                max_similitud = similitud
                best_match = dict(row)

    return best_match, max_similitud
