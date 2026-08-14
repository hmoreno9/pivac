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
from database import init_db, save_vacuno, find_nearest_vacuno, verify_user



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
    foto_caravana: UploadFile = File(...)
):
    try:
        morro_bytes = await foto_morro.read()
        caravana_bytes = await foto_caravana.read()

        # 1. Preprocesamiento y Extracción de Embedding
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        # 2. Cifrado
        vector_bytes = vector_biometrico.tobytes()
        vector_encriptado_str = engine.encrypt_data(vector_bytes)

        # 3. OCR Caravana
        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        # 4. Guardar en SQLite
        nuevo = save_vacuno(
            establecimiento_id=establecimiento_id,
            id_vacuno_interno=id_vacuno,
            numero_caravana=caravana_detectada,
            morro_vector=vector_biometrico.tolist(),
            morro_encriptado=vector_encriptado_str,
            latitude=latitude,
            longitude=longitude
        )

        return {
            "status": "success",
            "message": "Vacuno enrolado en SQLite correctamente.",
            "id_db": nuevo["id"],
            "id_vacuno": nuevo["id_vacuno_interno"],
            "caravana_ocr": nuevo["numero_caravana"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en enrolamiento: {str(e)}")

@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...),
    umbral_similitud: float = Form(0.85)
):
    try:
        # 1. Leer bytes del archivo subido
        morro_bytes = await foto_morro.read()

        # 2. Validar que la imagen no esté vacía
        if not morro_bytes or len(morro_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo de imagen enviado está vacío.")

        # Normalizar umbral (de 0-100 a 0.0-1.0 si aplica)
        if umbral_similitud > 1.0:
            umbral_similitud = umbral_similitud / 100.0

        # 3. Procesar e identificar
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        vacuno, similitud = find_nearest_vacuno(vector_query, threshold=umbral_similitud)

        if not vacuno:
            return {
                "status": "no_match",
                "message": "No hay animales registrados en la base de datos."
            }

        if similitud >= umbral_similitud:
            return {
                "status": "coincidencia_encontrada",
                "similitud_porcentaje": round(similitud * 100, 2),
                "umbral_aplicado_porcentaje": round(umbral_similitud * 100, 2),
                "vacuno": {
                    "id": vacuno["id"],
                    "id_vacuno_interno": vacuno["id_vacuno_interno"],
                    "establecimiento_id": vacuno["establecimiento_id"],
                    "numero_caravana": vacuno["numero_caravana"],
                    "latitud": vacuno["latitude"],
                    "longitud": vacuno["longitude"],
                    "fecha_registro": vacuno["fecha_registro"]
                }
            }
        else:
            return {
                "status": "sin_coincidencia",
                "message": f"Animal no identificado. Similitud máxima hallada ({round(similitud*100, 2)}%) fue inferior al umbral exigido ({round(umbral_similitud*100, 2)}%).",
                "similitud_maxima": round(similitud * 100, 2),
                "umbral_exigido": round(umbral_similitud * 100, 2)
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en identificación: {str(e)}")

if __name__ == "__main__":
        import uvicorn
        # Render asigna dinámicamente la variable PORT
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port)
