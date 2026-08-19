import base64
from contextlib import asynccontextmanager
from io import BytesIO
import os
import warnings
from typing import List, Optional

from cryptography.fernet import Fernet

from database import (
    find_nearest_vacuno,
    get_all_bovinos,
    init_db,
    save_bovino,
    verify_user,
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


@app.post("/api/v1/enrolar")
async def enroll_cattle(
    id_vacuno: str = Form(...),
    establecimiento_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    foto_morro: UploadFile = File(...),
    foto_caravana: UploadFile = File(...),
    raza: str = Form("Angus"),
):
    try:
        morro_bytes = await foto_morro.read()
        caravana_bytes = await foto_caravana.read()

        # 1. Generar imagen solapada con minucias/puntos clave
        annotated_bytes, bbox, num_kp = engine.get_annotated_muzzle(
            morro_bytes
        )
        morro_anotado_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"

        # 2. Thumbnails livianos para el DNI
        morro_thumb_b64 = generate_thumbnail_b64(morro_bytes)
        caravana_thumb_b64 = generate_thumbnail_b64(caravana_bytes)

        # 3. Procesar vector biométrico y OCR
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        vector_bytes = vector_biometrico.tobytes()
        vector_encriptado_str = engine.encrypt_data(vector_bytes)

        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        # Guardar registrando foto_morro_b64 (podés guardar el thumbnail o la anotada según prefieras)
        nuevo = save_bovino({
            "id_establecimiento": establecimiento_id,
            "id_vacuno": id_vacuno,
            "raza": raza,
            "numero_caravana": caravana_detectada,
            "morro_vector": vector_biometrico.tolist(),
            "morro_encriptado": vector_encriptado_str,
            "foto_morro_b64": morro_anotado_b64,  # Se guarda la imagen con minucias para el DNI
            "foto_cara_b64": caravana_thumb_b64,
            "latitude": latitude,
            "longitude": longitude,
        })

        return {
            "status": "success",
            "message": "Enrolamiento guardado correctamente.",
            "id_biometrico": nuevo["id_biometrico"],
            "id_vacuno": id_vacuno,
            "caravana_ocr": caravana_detectada or "No detectada",
            "foto_solapada_b64": morro_anotado_b64,
            "total_minucias": num_kp,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en enrolamiento: {str(e)}"
        )


@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...), umbral_similitud: float = Form(0.85)
):
    try:
        morro_bytes = await foto_morro.read()

        if not morro_bytes or len(morro_bytes) == 0:
            raise HTTPException(
                status_code=400, detail="La foto del morro está vacía."
            )

        if umbral_similitud > 1.0:
            umbral_similitud = umbral_similitud / 100.0

        # 1. Generar la vista anotada con minucias de la foto actual capturada
        annotated_bytes, _, num_kp = engine.get_annotated_muzzle(morro_bytes)
        foto_query_anotada_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"

        # 2. Extracción de vector y búsqueda k-NN
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        vacuno, similitud = find_nearest_vacuno(
            vector_query, threshold=umbral_similitud
        )

        if not vacuno:
            return {
                "status": "no_match",
                "message": "No hay animales registrados aún en la base de datos.",
                "similitud_maxima": 0.0,
                "umbral_exigido": round(umbral_similitud * 100, 2),
                "foto_query_anotada": foto_query_anotada_b64,
            }

        sim_pct = round(similitud * 100, 2)
        umb_pct = round(umbral_similitud * 100, 2)

        # 3. Intentar generar la imagen de comparación 1 a 1 con el candidato hallado
        foto_comparacion_b64 = None
        try:
            # Si el registro en BD incluye la imagen guardada en base64:
            b64_db = vacuno.get("foto_morro_b64", "")
            if b64_db and "," in b64_db:
                raw_b64 = b64_db.split(",")[1]
                foto_db_bytes = base64.b64decode(raw_b64)

                match_bytes, _ = engine.generate_matching_visual(
                    morro_bytes, foto_db_bytes
                )
                foto_comparacion_b64 = f"data:image/jpeg;base64,{base64.b64encode(match_bytes).decode('utf-8')}"
        except Exception as err_match:
            print(f"Aviso: No se pudo generar la vista 1a1: {err_match}")

        # 4. Construcción de respuesta
        if similitud >= umbral_similitud:
            return {
                "status": "coincidencia_encontrada",
                "message": f"¡Animal identificado con éxito! ({sim_pct}% de similitud)",
                "similitud_porcentaje": sim_pct,
                "umbral_aplicado_porcentaje": umb_pct,
                "foto_query_anotada": foto_query_anotada_b64,
                "foto_comparacion_b64": foto_comparacion_b64,
                "vacuno": {
                    "id": vacuno["id"],
                    "id_vacuno_interno": vacuno["id_vacuno_interno"],
                    "establecimiento_id": vacuno["id_establecimiento"],
                    "raza": vacuno.get("raza", "N/A"),
                    "numero_caravana": vacuno.get("numero_caravana")
                    or "Sin caravana",
                    "latitud": vacuno["latitude"],
                    "longitud": vacuno["longitude"],
                    "fecha_registro": vacuno["fecha_registro"],
                },
            }
        else:
            return {
                "status": "sin_coincidencia",
                "message": f"La mayor coincidencia hallada ({sim_pct}%) fue menor al umbral exigido ({umb_pct}%).",
                "similitud_maxima": sim_pct,
                "umbral_exigido": umb_pct,
                "foto_query_anotada": foto_query_anotada_b64,
                "foto_comparacion_b64": foto_comparacion_b64,  # Permite analizar visualmente por qué falló
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en identificación: {str(e)}"
        )


@app.get("/api/v1/registrados")
async def get_registered_cattle():
    try:
        bovinos = get_all_bovinos()
        return {"status": "success", "total": len(bovinos), "bovinos": bovinos}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error consultando registrados: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
