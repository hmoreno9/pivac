from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import datetime
import uuid

# Cadena de conexión a PostgreSQL local
DATABASE_URL = "postgresql://pivac_user:pivac_password@localhost:5432/pivac_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VacunoModel(Base):
    __tablename__ = "vacunos"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    establecimiento_id = Column(String(100), nullable=False, index=True)
    id_vacuno_interno = Column(String(50), nullable=False)
    numero_caravana = Column(String(50), nullable=True, index=True)
    
    # Biometría: Vector de 512 dimensiones para pgvector
    morro_vector = Column(Vector(512), nullable=False)
    
    # Backup Cifrado AES-256
    morro_encriptado = Column(Text, nullable=False)
    
    # Ubicación y Auditoría
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    fecha_registro = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    """
    Habilita la extensión pgvector y crea las tablas si no existen.
    """
    with engine.connect() as conn:
        conn.execute(Base.metadata.bind.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
