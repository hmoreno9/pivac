from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from typing import List, Optional
from cryptography.fernet import Fernet

import json

from motor_biometrico import CattleBiometricEngine

app = FastAPI(
    title="Motor Biométrico Vacuno API",
    version="1.0.0",
    description="API para enrolamiento e identificación biométrica por morro vacuno"
)

# Permitir peticiones desde cualquier origen para testing local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clave de cifrado de prueba (en producción va en Variable de Entorno)
SECRET_KEY = Fernet.generate_key()
engine = CattleBiometricEngine(encryption_key=SECRET_KEY)

class EnrollmentResponse(BaseModel):
    id_vacuno: str
    establecimiento_id: str
    numero_caravana_ocr: str
    vector_dim: int
    vector_preview: List[float]
    morro_encriptado_base64: str
    latitude: float
    longitude: float

@app.post("/api/v1/enrolar", response_model=EnrollmentResponse)
async def enroll_cattle(
    id_vacuno: str = Form(...),
    establecimiento_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    foto_morro: UploadFile = File(...),
    foto_caravana: UploadFile = File(...)
):
    try:
        # 1. Leer bytes de las imágenes enviadas desde la App
        morro_bytes = await foto_morro.read()
        caravana_bytes = await foto_caravana.read()

        # 2. Preprocesar y extraer vector biométrico del morro
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        # 3. Cifrar el vector para almacenamiento seguro
        vector_bytes = vector_biometrico.tobytes()
        vector_encriptado_str = engine.encrypt_data(vector_bytes)

        # 4. OCR sobre la caravana
        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        return EnrollmentResponse(
            id_vacuno=id_vacuno,
            establecimiento_id=establecimiento_id,
            numero_caravana_ocr=caravana_detectada,
            vector_dim=len(vector_biometrico),
            vector_preview=vector_biometrico[:5].tolist(), # Primeros 5 valores
            morro_encriptado_base64=vector_encriptado_str,
            latitude=latitude,
            longitude=longitude
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")

@app.post("/api/v1/identificar")
async def identify_cattle(
    foto_morro: UploadFile = File(...)
):
    """
    Recibe la imagen capturada en el campo y retorna el vector
    listo para hacer la consulta k-NN contra PostgreSQL (pgvector).
    """
    try:
        morro_bytes = await foto_morro.read()
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        return {
            "status": "success",
            "vector_query": vector_query.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
