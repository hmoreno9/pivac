from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
import numpy as np

from motor_biometrico import CattleBiometricEngine
from database import init_db, get_db, VacunoModel

app = FastAPI(title="Motor Biométrico Vacuno API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = Fernet.generate_key()
engine = CattleBiometricEngine(encryption_key=SECRET_KEY)

# Inicializar Base de Datos al arrancar
@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/api/v1/enrolar")
async def enroll_cattle(
    id_vacuno: str = Form(...),
    establecimiento_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    foto_morro: UploadFile = File(...),
    foto_caravana: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        morro_bytes = await foto_morro.read()
        caravana_bytes = await foto_caravana.read()

        # 1. Preprocesamiento y Extracción de Embedding
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        # 2. Cifrado AES-256
        vector_bytes = vector_biometrico.tobytes()
        vector_encriptado_str = engine.encrypt_data(vector_bytes)

        # 3. OCR Caravana
        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        # 4. Guardar registro en PostgreSQL + pgvector
        nuevo_vacuno = VacunoModel(
            establecimiento_id=establecimiento_id,
            id_vacuno_interno=id_vacuno,
            numero_caravana=caravana_detectada,
            morro_vector=vector_biometrico.tolist(),
            morro_encriptado=vector_encriptado_str,
            latitude=latitude,
            longitude=longitude
        )
        
        db.add(nuevo_vacuno)
        db.commit()
        db.refresh(nuevo_vacuno)

        return {
            "status": "success",
            "message": "Vacuno enrolado y guardado en BD con éxito.",
            "id_db": nuevo_vacuno.id,
            "id_vacuno": nuevo_vacuno.id_vacuno_interno,
            "caravana_ocr": nuevo_vacuno.numero_caravana
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error en enrolamiento: {str(e)}")

@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...),
    umbral_similitud: float = 0.85, # 85% de coincidencia mínima
    db: Session = Depends(get_db)
):
    """
    Toma la foto del morro tomada en el campo y busca el vacuno más cercano 
    en la BD utilizando la distancia Coseno en pgvector.
    """
    try:
        morro_bytes = await foto_morro.read()
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        # Consulta pgvector: ordenar por distancia coseno (cosine_distance)
        # 0.0 significa 100% idénticos.
        resultado = (
            db.query(
                VacunoModel,
                VacunoModel.morro_vector.cosine_distance(vector_query.tolist()).label("distancia")
            )
            .order_by("distancia")
            .first()
        )

        if not resultado:
            return {"status": "no_match", "message": "No hay animales registrados en la BD."}

        vacuno, distancia = resultado
        similitud = 1.0 - distancia

        if similitud >= umbral_similitud:
            return {
                "status": "coincidencia_encontrada",
                "similitud_porcentaje": round(similitud * 100, 2),
                "vacuno": {
                    "id": vacuno.id,
                    "id_vacuno_interno": vacuno.id_vacuno_interno,
                    "establecimiento_id": vacuno.establecimiento_id,
                    "numero_caravana": vacuno.numero_caravana,
                    "latitud": vacuno.latitude,
                    "longitud": vacuno.longitude,
                    "fecha_registro": vacuno.fecha_registro.isoformat()
                }
            }
        else:
            return {
                "status": "sin_coincidencia",
                "message": f"El animal no está registrado. La mayor similitud encontrada fue {round(similitud*100, 2)}%.",
                "similitud_maxima": round(similitud * 100, 2)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en identificación: {str(e)}")
