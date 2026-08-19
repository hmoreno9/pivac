import json
import sqlite3
import hashlib
import numpy as np
import uuid
import datetime


DB_PATH = 'pivac.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
  conn = sqlite3.connect(DB_PATH)
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
  # Crear/Actualizar tabla de bovinos
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS bovinos (
            id_biometrico TEXT PRIMARY KEY,
            id_establecimiento TEXT,
            id_vacuno_interno TEXT,
            raza TEXT,
            numero_caravana TEXT,
            morro_vectors TEXT,        -- Se guarda lista JSON de vectores [v1, v2, v3]
            foto_morro_b64 TEXT,       -- Foto de morro principal/anotada
            foto_vaca_b64 TEXT,        -- Foto completa del animal
            foto_cara_b64 TEXT,        -- Foto de la caravana
            latitude REAL,
            longitude REAL,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
  
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS tab_razas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL
        )
    """)

   # Insertar razas por defecto si no existen
  razas_iniciales = [
        "Angus", "Hereford", "Brahman", 
        "Brangus", "Braford", "Limousin", "Criollo"
    ]

  for raza in razas_iniciales:
        cursor.execute("INSERT OR IGNORE INTO tab_razas (nombre) VALUES (?)", (raza,))

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

    # Insertar EST-IRAOLA-01 si no existe
  cursor.execute("SELECT * FROM tab_establecimiento WHERE id = 'EST-IRAOLA-01'")
  if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO tab_establecimiento (id, nombre, domicilio, nombre_contacto, celular, mail, ubicacion_gps, fecha_registro)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "EST-IRAOLA-01", "Cabaña Iraola", "Camino Las Acacias S/N, Iraola", 
            "Administración Iraola", "+54 9 249 4987654", "contacto@cabanairaola.ar", 
            "-37.2500,-59.2000", datetime.datetime.utcnow().isoformat()
        ))
  
  # Migración defensiva en caso de que la tabla ya existiera sin 'foto_vaca_b64'
  cursor.execute('PRAGMA table_info(bovinos)')
  columns = [column[1] for column in cursor.fetchall()]
  if 'foto_vaca_b64' not in columns:
    cursor.execute('ALTER TABLE bovinos ADD COLUMN foto_vaca_b64 TEXT')

  conn.commit()
  conn.close()


def save_bovino(data: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    import uuid, json
    id_bio = str(uuid.uuid4())

    fotos_morros_json = json.dumps(data.get("fotos_morros_b64", [data.get("foto_morro_b64", "")]))

    cursor.execute("""
        INSERT INTO bovinos (
            id_biometrico, id_establecimiento, id_vacuno_interno, raza,
            numero_caravana, morro_vectors, foto_morro_b64, foto_vaca_b64,
            foto_cara_b64, latitude, longitude
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id_bio,
        data['id_establecimiento'],
        data['id_vacuno'],
        data.get('raza', 'Angus'),
        data.get('numero_caravana', ''),
        json.dumps(data['morro_vectors']),
        fotos_morros_json, # Guardamos el array de morros en esta columna
        data.get('foto_vaca_b64', ''),
        data.get('foto_cara_b64', ''),
        data.get('latitude', 0.0),
        data.get('longitude', 0.0)
    ))

    conn.commit()
    conn.close()
    return {"id_biometrico": id_bio}

def find_nearest_vacuno(query_vector, threshold=0.85):
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  cursor.execute('SELECT * FROM bovinos')
  rows = cursor.fetchall()
  conn.close()

  best_vacuno = None
  max_similitud = 0.0

  import numpy as np

  q_vec = np.array(query_vector, dtype=np.float32)

  for row in rows:
    bovino = dict(row)
    vectors_list = clean_b64_list(bovino.get('morro_vectors'))

    for vec in vectors_list:
      try:
        v = np.array(vec, dtype=np.float32)
        sim = float(
            np.dot(q_vec, v)
            / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-10)
        )
        if sim > max_similitud:
          max_similitud = sim
          best_vacuno = bovino
      except Exception:
        continue

  # Procesar lista limpia de fotos de morros antes de retornar
  if best_vacuno:
    best_vacuno['fotos_morros_b64'] = clean_b64_list(
        best_vacuno.get('foto_morro_b64')
    )

  return best_vacuno, max_similitud

def get_all_razas():
    """Devuelve la lista de razas ordenadas alfabéticamente."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre FROM tab_razas ORDER BY nombre ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_bovinos():
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  cursor.execute("SELECT * FROM bovinos ORDER BY fecha_registro DESC")
  rows = cursor.fetchall()
  conn.close()

  bovinos = []
  for row in rows:
    b = dict(row)

    # Decodificar el JSON de fotos de morros
    if "foto_morro_b64" in b and b["foto_morro_b64"]:
      try:
        parsed = json.loads(b["foto_morro_b64"])
        if isinstance(parsed, list):
          b["fotos_morros_b64"] = parsed
        else:
          b["fotos_morros_b64"] = [b["foto_morro_b64"]]
      except Exception:
        b["fotos_morros_b64"] = [b["foto_morro_b64"]]

    bovinos.append(b)

  return bovinos

def clean_b64_list(raw_field) -> list:
  """Convierte de forma defensiva cualquier campo de la BD en una lista limpia de strings Base64."""
  if not raw_field:
    return []

  if isinstance(raw_field, list):
    return raw_field

  if isinstance(raw_field, str):
    # Intentar parsear si viene como JSON string ("[\"data:image...\"]")
    try:
      parsed = json.loads(raw_field)
      if isinstance(parsed, list):
        return parsed
      elif isinstance(parsed, str):
        # En caso de que se haya guardado con doble serialización json.dumps(json.dumps(...))
        try:
          second_parse = json.loads(parsed)
          if isinstance(second_parse, list):
            return second_parse
        except Exception:
          pass
        return [parsed]
    except Exception:
      return [raw_field]

  return []


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

def delete_bovino_by_id(id_biometrico: str) -> bool:
  """Elimina un registro de bovino por su ID biométrico (UUID).

  Retorna True si eliminó al menos una fila, False en caso contrario.
  """
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  cursor.execute(
      'DELETE FROM bovinos WHERE id_biometrico = ?', (id_biometrico,)
  )
  deleted_count = cursor.rowcount

  conn.commit()
  conn.close()

  return deleted_count > 0

def get_ultimo_bovino():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Obtenemos el último grabado ordenado por fecha o rowid
    cursor.execute("SELECT id_vacuno_interno FROM bovinos ORDER BY fecha_registro DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row and row['id_vacuno_interno']:
        return row['id_vacuno_interno']
    return None

def get_all_establecimientos():
    """Devuelve la lista de todos los establecimientos disponibles."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre FROM tab_establecimiento ORDER BY nombre ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_ultimo_bovino_por_establecimiento(id_establecimiento: str):
    """Obtiene el último ID de vacuno registrado en un establecimiento específico."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id_vacuno_interno 
        FROM bovinos 
        WHERE id_establecimiento = ? 
        ORDER BY fecha_registro DESC LIMIT 1
    """, (id_establecimiento,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and row['id_vacuno_interno']:
        return row['id_vacuno_interno']
    return None

