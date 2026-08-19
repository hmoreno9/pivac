import base64
from contextlib import asynccontextmanager
from io import BytesIO
import os
import warnings
from typing import List, Optional
from cryptography.fernet import Fernet

from database import delete_bovino_by_id

from database import (
    find_nearest_vacuno,
    get_all_bovinos,
    init_db,
    save_bovino,
    verify_user,
    get_all_establecimientos,
    get_ultimo_bovino_por_establecimiento,
    get_all_razas
)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from motor_biometrico import CattleBiometricEngine
import numpy as np
from PIL import Image
from pydantic import BaseModel
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Obtener la ruta del directorio actual donde está main.py
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"



def generate_thumbnail_b64(image_bytes: bytes, max_size=(180, 180)) -> str:
    """Toma la foto en alta resolución de la cámara del celular,

    la redimensiona a miniatura de 180x180 px y la convierte a string Base64.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.thumbnail(max_size)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return ""


# Context Manager para el ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Motor Biométrico Vacuno API (PIVAC Dev)", lifespan=lifespan)

# Montar carpeta estática usando la ruta absoluta
if STATIC_DIR.exists():
  app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = Fernet.generate_key()
engine = CattleBiometricEngine(encryption_key=SECRET_KEY)


# --- Modelos Pydantic ---
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class IdentificationResponse(BaseModel):
    status: str
    message: str
    similitud_porcentaje: Optional[float] = None
    foto_query_anotada: Optional[str] = (
        None  # Imagen capturada en campo con minucias
    )
    foto_comparacion_1a1: Optional[str] = (
        None  # Imagen lado a lado con líneas de coincidencia
    )
    vacuno: Optional[dict] = None


# Servir archivos estáticos si existe la carpeta
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_home():
  index_path = STATIC_DIR / "index.html"
  if index_path.exists():
    return FileResponse(str(index_path))
  return {
    "message": (
      f"API PIVAC activa. Coloca index.html en {STATIC_DIR} (no encontrado)"
    )
  }

@app.post("/api/v1/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = verify_user(username, password)
    if not user:
        raise HTTPException(
            status_code=401, detail="Usuario o contraseña incorrectos."
        )

    return {
        "status": "success",
        "message": f"Bienvenido {user['nombre_operador']}",
        "user": user,
    }


from typing import List
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


@app.post("/api/v1/enrolar")
async def enroll_cattle(
    id_vacuno: str = Form(...),
    establecimiento_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    raza: str = Form("Angus"),
    fotos_morro: List[UploadFile] = File(...),
    foto_vaca: UploadFile = File(...),
    foto_caravana: UploadFile = File(...)
):
    try:
        if len(fotos_morro) == 0:
            raise HTTPException(status_code=400, detail="Debe enviar al menos 1 foto del morro.")

        vaca_bytes = await foto_vaca.read()
        caravana_bytes = await foto_caravana.read()

        foto_vaca_b64 = generate_thumbnail_b64(vaca_bytes, max_size=(300, 300))
        foto_caravana_b64 = generate_thumbnail_b64(caravana_bytes, max_size=(300, 300))

        vectores_morro = []
        fotos_morros_anotadas_b64 = []  # Lista para guardar TODAS las fotos procesadas con minucias

        for file_morro in fotos_morro:
            m_bytes = await file_morro.read()

            # Extraer vector para el matcher
            processed = engine.preprocess_muzzle(m_bytes)
            vec = engine.extract_vector(processed)
            vectores_morro.append(vec.tolist())

            # Generar anotación de minucias para CADA una de las fotos
            annotated_bytes, _, _ = engine.get_annotated_muzzle(m_bytes)
            m_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"
            fotos_morros_anotadas_b64.append(m_b64)

        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        # Guardar pasando la lista de fotos anotadas
        nuevo = save_bovino({
            "id_establecimiento": establecimiento_id,
            "id_vacuno": id_vacuno,
            "raza": raza,
            "numero_caravana": caravana_detectada,
            "morro_vectors": vectores_morro,
            "fotos_morros_b64": fotos_morros_anotadas_b64, # Guardamos la lista en lugar de una sola
            "foto_morro_b64": fotos_morros_anotadas_b64[0], # Guardamos también la primera por retrocompatibilidad
            "foto_vaca_b64": foto_vaca_b64,
            "foto_cara_b64": foto_caravana_b64,
            "latitude": latitude,
            "longitude": longitude
        })

        return {
            "status": "success",
            "message": f"Enrolamiento exitoso con {len(vectores_morro)} muestras biométricas.",
            "id_biometrico": nuevo["id_biometrico"],
            "id_vacuno": id_vacuno,
            "caravana_ocr": caravana_detectada or "No detectada"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en enrolamiento: {str(e)}")

@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...), umbral_similitud: float = Form(0.85)
):
  try:
    morro_bytes = await foto_morro.read()
    if umbral_similitud > 1.0:
      umbral_similitud = umbral_similitud / 100.0

    # Imagen anotada de la foto tomada en el momento
    annotated_bytes, _, _ = engine.get_annotated_muzzle(morro_bytes)
    foto_query_anotada_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"

    processed_morro = engine.preprocess_muzzle(morro_bytes)
    vector_query = engine.extract_vector(processed_morro)

    vacuno, similitud = find_nearest_vacuno(
        vector_query, threshold=umbral_similitud
    )

    if not vacuno:
      return {
          "status": "no_match",
          "message": "No hay animales registrados aún.",
          "similitud_maxima": 0.0,
      }

    sim_pct = round(similitud * 100, 2)
    umb_pct = round(umbral_similitud * 100, 2)

    # Extraer la lista limpia de morros
    morros_registrados_list = vacuno.get("fotos_morros_b64", [])

    # Match visual 1 a 1
    foto_comparacion_b64 = None
    try:
      if morros_registrados_list:
        first_m = morros_registrados_list[0]
        raw_b64 = (
            first_m.split(",")[1] if "," in first_m else first_m
        )  # Quitar encabezado data:image/jpeg;base64 si existe
        foto_db_bytes = base64.b64decode(raw_b64)
        match_bytes, _ = engine.generate_matching_visual(
            morro_bytes, foto_db_bytes
        )
        foto_comparacion_b64 = f"data:image/jpeg;base64,{base64.b64encode(match_bytes).decode('utf-8')}"
    except Exception as e:
      print(f"Aviso matching visual: {e}")

    # Mapeo completo
    dni_data = {
        "id_biometrico": vacuno["id_biometrico"],
        "id_vacuno_interno": vacuno["id_vacuno_interno"],
        "establecimiento_nombre": vacuno["id_establecimiento"],
        "raza": vacuno.get("raza", "Angus"),
        "numero_caravana": vacuno.get("numero_caravana") or "S/N",
        "fotos_morros_b64": morros_registrados_list,
        "foto_vaca_b64": vacuno.get("foto_vaca_b64"),
        "foto_cara_b64": vacuno.get("foto_cara_b64"),
        "fecha_registro": vacuno["fecha_registro"],
    }

    if similitud >= umbral_similitud:
      return {
          "status": "coincidencia_encontrada",
          "similitud_porcentaje": sim_pct,
          "foto_query_anotada": foto_query_anotada_b64,
          "foto_comparacion_b64": foto_comparacion_b64,
          "vacuno": dni_data,
      }
    else:
      return {
          "status": "sin_coincidencia",
          "message": (
              f"Máxima coincidencia hallada ({sim_pct}%) es menor al umbral"
              f" ({umb_pct}%)."
          ),
          "similitud_maxima": sim_pct,
          "foto_query_anotada": foto_query_anotada_b64,
          "foto_comparacion_b64": foto_comparacion_b64,
      }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/razas")
async def obtener_razas():
    return {"razas": get_all_razas()}

@app.get("/api/v1/registrados")
async def get_registered_cattle():
    try:
        bovinos = get_all_bovinos()
        return {"status": "success", "total": len(bovinos), "bovinos": bovinos}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error consultando registrados: {str(e)}"
        )


@app.get("/api/v1/establecimientos")
async def obtener_establecimientos():
    return {"establecimientos": get_all_establecimientos()}

@app.get("/api/v1/ultimo-vacuno")
async def obtener_ultimo_vacuno(establecimiento_id: str):
    ultimo_id = get_ultimo_bovino_por_establecimiento(establecimiento_id)
    return {"ultimo_id": ultimo_id}


@app.delete("/api/v1/bovinos/{id_biometrico}")
async def delete_bovino(id_biometrico: str):
  try:
    exito = delete_bovino_by_id(id_biometrico)
    if not exito:
      raise HTTPException(
          status_code=404, detail="No se encontró el bovino solicitado."
      )

    return {
        "status": "success",
        "message": f"Bovino {id_biometrico} eliminado correctamente.",
    }
  except HTTPException as he:
    raise he
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Error al eliminar el bovino: {str(e)}"
    )

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
