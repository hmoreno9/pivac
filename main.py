from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from cryptography.fernet import Fernet
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from motor_biometrico import CattleBiometricEngine

from database import init_db, save_bovino, find_nearest_vacuno, verify_user, get_all_bovinos

import base64
from io import BytesIO
from PIL import Image


def generate_thumbnail_b64(image_bytes: bytes, max_size=(180, 180)) -> str:
    """
    Toma la foto en alta resolución de la cámara del celular,
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
    # Código de startup (al iniciar el servidor)
    init_db()
    yield
    # Código de shutdown (opcional, al apagar el servidor)

app = FastAPI(title="Motor Biométrico Vacuno API (PIVAC Dev)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = Fernet.generate_key()
engine = CattleBiometricEngine(encryption_key=SECRET_KEY)

# Servir archivos estáticos si existe la carpeta
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_home():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "API PIVAC activa. Coloca index.html en /static"}

# Inicializar tabla SQLite al arrancar la app
@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/api/v1/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    user = verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos.")
    
    return {
        "status": "success",
        "message": f"Bienvenido {user['nombre_operador']}",
        "user": user
    }

@app.post("/api/v1/enrolar")
async def enroll_cattle(
    id_vacuno: str = Form(...),
    establecimiento_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    foto_morro: UploadFile = File(...),
    foto_caravana: UploadFile = File(...),
    raza: str = Form("Angus")
):
    try:
        morro_bytes = await foto_morro.read()
        caravana_bytes = await foto_caravana.read()

        # Generar thumbnails Base64 para mostrar en el DNI
        morro_b64 = generate_thumbnail_b64(morro_bytes)
        caravana_b64 = generate_thumbnail_b64(caravana_bytes)

        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        vector_bytes = vector_biometrico.tobytes()
        vector_encriptado_str = engine.encrypt_data(vector_bytes)

        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        # Guardar pasando los Base64
        nuevo = save_bovino({
            "id_establecimiento": establecimiento_id,
            "id_vacuno": id_vacuno,
            "raza": raza,
            "numero_caravana": caravana_detectada,
            "morro_vector": vector_biometrico.tolist(),
            "morro_encriptado": vector_encriptado_str,
            "foto_morro_b64": morro_b64,
            "foto_cara_b64": caravana_b64,
            "latitude": latitude,
            "longitude": longitude
        })

        return {
            "status": "success",
            "message": "Enrolamiento guardado correctamente.",
            "id_biometrico": nuevo["id_biometrico"],
            "id_vacuno": id_vacuno,
            "caravana_ocr": caravana_detectada or "No detectada"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en enrolamiento: {str(e)}")

@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...),
    umbral_similitud: float = Form(0.85)
):
    try:
        morro_bytes = await foto_morro.read()

        if not morro_bytes or len(morro_bytes) == 0:
            raise HTTPException(status_code=400, detail="La foto del morro está vacía.")

        if umbral_similitud > 1.0:
            umbral_similitud = umbral_similitud / 100.0

        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        vacuno, similitud = find_nearest_vacuno(vector_query, threshold=umbral_similitud)

        if not vacuno:
            return {
                "status": "no_match",
                "message": "No hay animales registrados aún en la base de datos.",
                "similitud_maxima": 0.0,
                "umbral_exigido": round(umbral_similitud * 100, 2)
            }

        sim_pct = round(similitud * 100, 2)
        umb_pct = round(umbral_similitud * 100, 2)

        if similitud >= umbral_similitud:
            return {
                "status": "coincidencia_encontrada",
                "message": f"¡Animal identificado con éxito! ({sim_pct}% de similitud)",
                "similitud_porcentaje": sim_pct,
                "umbral_aplicado_porcentaje": umb_pct,
                "vacuno": {
                    "id": vacuno["id"],                                    # id_biometrico (UUID)
                    "id_vacuno_interno": vacuno["id_vacuno_interno"],      # Ej: VAC-2026-001
                    "establecimiento_id": vacuno["id_establecimiento"],    # Mapeado a id_establecimiento
                    "raza": vacuno.get("raza", "N/A"),
                    "numero_caravana": vacuno.get("numero_caravana") or "Sin caravana",
                    "latitud": vacuno["latitude"],
                    "longitud": vacuno["longitude"],
                    "fecha_registro": vacuno["fecha_registro"]
                }
            }
        else:
            return {
                "status": "sin_coincidencia",
                "message": f"La mayor coincidencia hallada ({sim_pct}%) fue menor al umbral exigido ({umb_pct}%).",
                "similitud_maxima": sim_pct,
                "umbral_exigido": umb_pct
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en identificación: {str(e)}")


@app.get("/api/v1/registrados")
async def get_registered_cattle():
    try:
        bovinos = get_all_bovinos()
        return {
            "status": "success",
            "total": len(bovinos),
            "bovinos": bovinos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando registrados: {str(e)}")

if __name__ == "__main__":
        import uvicorn
        # Render asigna dinámicamente la variable PORT
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port)
