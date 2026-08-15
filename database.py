import sqlite3
import json
import uuid
import datetime
import hashlib
import numpy as np

DB_FILE = "pivac.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Tabla Usuarios
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nombre_operador TEXT NOT NULL,
            fecha_creacion TEXT NOT NULL
        )
    """)

    # 2. Tabla Establecimiento (Campos / Estancias)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tab_establecimiento (
            id TEXT PRIMARY KEY,
            nombre TEXT NOT NULL,
            domicilio TEXT,
            nombre_contacto TEXT,
            celular TEXT,
            mail TEXT,
            ubicacion_gps TEXT,
            fecha_registro TEXT NOT NULL
        )
    """)

    # 3. Tabla Bovinos (Ficha Biométrica Completa)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tab_bovinos (
            id TEXT PRIMARY KEY,                       -- id_biometrico (UUID)
            id_establecimiento TEXT NOT NULL,          -- FK a tab_establecimiento
            id_vacuno_interno TEXT NOT NULL,
            raza TEXT,
            numero_caravana TEXT,
            morro_vector TEXT NOT NULL,                -- Vector Híbrido JSON
            morro_encriptado TEXT NOT NULL,            -- AES-256
            foto_morro_b64 TEXT,                       -- Imagen/Miniatura Morro
            foto_cara_b64 TEXT,                        -- Imagen/Miniatura Cara
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            fecha_registro TEXT NOT NULL,
            FOREIGN KEY (id_establecimiento) REFERENCES tab_establecimiento(id)
        )
    """)

    # Cargar usuario 'admin' y establecimiento por defecto si no existen
    cursor.execute("SELECT * FROM usuarios WHERE username = 'admin'")
    if not cursor.fetchone():
        pass_hash = hashlib.sha256("pivac2026".encode()).hexdigest()
        cursor.execute("""
            INSERT INTO usuarios (id, username, password_hash, nombre_operador, fecha_creacion)
            VALUES (?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), "admin", pass_hash, "Operador Campo", datetime.datetime.utcnow().isoformat()))

    cursor.execute("SELECT * FROM tab_establecimiento WHERE id = 'EST-TANDIL-01'")
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO tab_establecimiento (id, nombre, domicilio, nombre_contacto, celular, mail, ubicacion_gps, fecha_registro)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "EST-TANDIL-01", "Estancia Las Sierras", "Ruta 226 Km 155, Tandil", 
            "Hernán Moreno", "+54 9 249 4123456", "contacto@lassierras.ar", 
            "-37.3216,-59.1332", datetime.datetime.utcnow().isoformat()
        ))

    conn.commit()
    conn.close()

def save_bovino(data: dict) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()

    id_biometrico = str(uuid.uuid4())
    fecha_iso = datetime.datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO tab_bovinos (
            id, id_establecimiento, id_vacuno_interno, raza, numero_caravana,
            morro_vector, morro_encriptado, foto_morro_b64, foto_cara_b64,
            latitude, longitude, fecha_registro
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id_biometrico,
        data["id_establecimiento"],
        data["id_vacuno"],
        data.get("raza", "Angus"),
        data.get("numero_caravana"),
        json.dumps(data["morro_vector"]),
        data["morro_encriptado"],
        data.get("foto_morro_b64"),
        data.get("foto_cara_b64"),
        data["latitude"],
        data["longitude"],
        fecha_iso
    ))

    conn.commit()
    conn.close()

    return {
        "id_biometrico": id_biometrico,
        "id_vacuno": data["id_vacuno"],
        "caravana": data.get("numero_caravana")
    }

def get_all_bovinos():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            b.id as id_biometrico, 
            b.id_vacuno_interno, 
            b.raza, 
            b.numero_caravana, 
            b.foto_morro_b64, 
            b.foto_cara_b64, 
            b.fecha_registro, 
            b.latitude, 
            b.longitude,
            e.nombre as establecimiento_nombre
        FROM tab_bovinos b
        LEFT JOIN tab_establecimiento e ON b.id_establecimiento = e.id
        ORDER BY b.fecha_registro DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def find_nearest_vacuno(query_vector: np.ndarray, threshold: float = 0.85):
    """
    Compara el vector de la consulta contra todos los bovinos registrados en tab_bovinos
    utilizando similitud Coseno.
    Devuelve la fila de la BD (dict) del animal con mayor similitud y el valor float del score.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Seleccionamos con los nuevos nombres de columna de tab_bovinos
    cursor.execute("""
        SELECT 
            id,
            id_establecimiento,
            id_vacuno_interno,
            raza,
            numero_caravana,
            morro_vector,
            latitude,
            longitude,
            fecha_registro
        FROM tab_bovinos
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, 0.0

    best_match = None
    max_similarity = -1.0

    # Normalización del vector de entrada por seguridad
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    for row in rows:
        db_vacuno = dict(row)
        # Reconstruir el vector desde JSON en la BD
        db_vector = np.array(json.loads(db_vacuno["morro_vector"]), dtype=np.float32)
        db_norm = db_vector / (np.linalg.norm(db_vector) + 1e-10)

        # Similitud Coseno (Producto Punto entre vectores unitarios)
        similarity = float(np.dot(query_norm, db_norm))

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = db_vacuno

    return best_match, max_similarity

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
