import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from cryptography.fernet import Fernet

import json

from motor_biometrico import CattleBiometricEngine

app = FastAPI(
    title="Motor Biométrico Vacuno API",
    version="1.1.0",
    description="API para enrolamiento e identificación biométrica por morro vacuno con feedback visual"
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

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class IdentificationResponse(BaseModel):
    status: str
    vector_query: List[float]
    morro_anotado_base64: Optional[str] = None  # Imagen con marcas y delimitación dibujadas
    bbox_morro: Optional[BoundingBox] = None    # Coordenadas del área del morro
    num_keypoints: Optional[int] = None         # Cantidad de minucias/puntos clave detectados


class EnrollmentResponse(BaseModel):
    id_vacuno: str
    establecimiento_id: str
    numero_caravana_ocr: str
    vector_dim: int
    vector_preview: List[float]
    morro_encriptado_base64: str
    latitude: float
    longitude: float

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

        # Extraer vector biométrico
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_biometrico = engine.extract_vector(processed_morro)

        # Generar imagen solapada con minucias para guardarla o mostrarla en DNI
        annotated_bytes, bbox, num_kp = engine.get_annotated_muzzle(morro_bytes)
        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"

        caravana_detectada = engine.process_tag_ocr(caravana_bytes)

        return {
            "id_vacuno": id_vacuno,
            "establecimiento_id": establecimiento_id,
            "numero_caravana_ocr": caravana_detectada,
            "morro_anotado_b64": annotated_b64,
            "minucias_detectadas": num_kp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v1/enrolar_old", response_model=EnrollmentResponse)
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
    foto_morro: UploadFile = File(...),
    umbral_similitud: float = Form(85.0)
):
    try:
        morro_bytes = await foto_morro.read()
        
        # Generar vista con minucias de la foto actual de campo
        annotated_bytes, _, num_kp = engine.get_annotated_muzzle(morro_bytes)
        query_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode('utf-8')}"

        # AQUÍ HACES TU BÚSQUEDA EN LA BASE DE DATOS (pgvector k-NN)
        # Ejemplo simulado de vacuno encontrado en la BD:
        # vacuno_bd_bytes = db.get_foto_morro_original(id_coincidente)
        # sim_porcentaje = 92.4
        
        # Para el ejemplo visual 1 a 1:
        # matching_bytes, num_matches = engine.generate_matching_visual(morro_bytes, vacuno_bd_bytes)
        # match_b64 = f"data:image/jpeg;base64,{base64.b64encode(matching_bytes).decode('utf-8')}"

        return {
            "status": "coincidencia_encontrada",
            "similitud_porcentaje": 92.4,
            "foto_query_anotada": query_b64,
            # "foto_match_1a1": match_b64, # Imagen combinada 1 a 1 con líneas de coincidencia
            "vacuno": {
                "id_vacuno_interno": "VAC-2026-001",
                "establecimiento_id": "EST-TANDIL-01",
                "numero_caravana": "ARG-8841",
                "fecha_registro": "2026-03-01",
                # "foto_morro_b64": f"data:image/jpeg;base64,{...}" # Foto enrolada de la BD
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/identificar_old", response_model=IdentificationResponse)
async def identify_cattle(
    foto_morro: UploadFile = File(...),
    incluir_anotaciones: bool = Form(True) # Flag opcional para pedir o no la imagen dibujada
):
    """
    Recibe la imagen capturada en el campo, extrae el vector para pgvector
    y opcionalmente genera la imagen con la delimitación y minucias detectadas.
    """
    try:
        morro_bytes = await foto_morro.read()

        # 1. Preprocesar y segmentar morro
        processed_morro = engine.preprocess_muzzle(morro_bytes)
        vector_query = engine.extract_vector(processed_morro)

        # 2. Generar anotaciones/dibujo sobre la imagen (Delimitación + Minucias/Puntos)
        morro_anotado_b64 = None
        bbox = None
        num_kp = 0

        if incluir_anotaciones:
            # Métodos a implementar/extender en tu CattleBiometricEngine:
            # - engine.draw_muzzle_annotations(morro_bytes) -> devuelve (bytes_jpg, bbox_dict, count_keypoints)
            annotated_bytes, bbox_data, num_kp = engine.get_annotated_muzzle(morro_bytes)
            morro_anotado_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
            
            if bbox_data:
                bbox = BoundingBox(**bbox_data)

        return IdentificationResponse(
            status="success",
            vector_query=vector_query.tolist(),
            morro_anotado_base64=morro_anotado_b64,
            bbox_morro=bbox,
            num_keypoints=num_kp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")

@app.post("/api/v1/comparar_detalle")
async def compare_detail(
    foto_query: UploadFile = File(...),
    id_vacuno_candidato: str = Form(...)
):
    """
    Endpoint para verificación 1:1 o visualización de coincidencia detallada.
    Dibuja los pareaos de minucias/keypoints entre la foto capturada y la foto enrolada.
    """
    try:
        query_bytes = await foto_query.read()
        
        # 1. Buscar la imagen original de enrolamiento del candidato (ej: desde DB o S3)
        # candidado_bytes = engine.get_enrolled_image(id_vacuno_candidato)
        
        # 2. Generar la imagen de Feature Matching (líneas de coincidencia entre ambas fotos)
        # match_image_bytes, score = engine.match_and_draw(query_bytes, candidado_bytes)
        
        return {
            "status": "success",
            "id_vacuno": id_vacuno_candidato,
            # "score_coincidencia": score,
            # "matching_image_base64": base64.b64encode(match_image_bytes).decode("utf-8")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en comparación detallada: {str(e)}")

@app.get("/api/v1/razas")
async def obtener_razas():
    return {"razas": get_all_razas()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
