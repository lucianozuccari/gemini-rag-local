import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from qdrant_client import QdrantClient

from contextlib import asynccontextmanager

# Cargar variables desde .env si no están en el entorno
if not os.environ.get("GEMINI_API_KEY"):
    for env_path in [".env", "../.env", "rag-backend/.env"]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        os.environ[k] = v
            break

# Configurar API de Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

COLLECTION_NAME = "manuales_empresa"
qdrant_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client
    qdrant_host = os.environ.get("QDRANT_HOST")
    if qdrant_host:
        qdrant_client = QdrantClient(host=qdrant_host, port=6333)
    else:
        try:
            qdrant_client = QdrantClient(host="localhost", port=6333, timeout=1.0)
            qdrant_client.get_collections()
        except Exception:
            db_path = os.path.join(os.path.dirname(__file__), "qdrant_local_data")
            print(f"Qdrant no detectado en localhost. Usando almacenamiento local integrado en: {db_path}")
            qdrant_client = QdrantClient(path=db_path)
    yield
    if qdrant_client:
        try:
            qdrant_client.close()
        except Exception:
            pass

app = FastAPI(title="Backend RAG Empresarial", lifespan=lifespan)

# Permitir CORS desde cualquier origen para consultas de frontend web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MensajeChat(BaseModel):
    role: str
    content: str

class ConsultaRequest(BaseModel):
    pregunta: str
    historial: list[MensajeChat] = []

@app.post("/preguntar")
async def consultar_rag(request: ConsultaRequest):
    try:
        # Generar embedding para la pregunta del usuario
        res_embedding = genai.embed_content(
            model="models/gemini-embedding-2",
            content=request.pregunta,
            task_type="retrieval_query"
        )

        
        # Buscar en la colección de Qdrant
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=res_embedding['embedding'],
            limit=3
        )
        resultados_busqueda = response.points

        
        if not resultados_busqueda:
            contexto = "No se encontró información relevante."
        else:
            contexto_items = []
            for hit in resultados_busqueda:
                texto = hit.payload.get("texto", "")
                source = hit.payload.get("source", "")
                type_src = hit.payload.get("type", "")
                contexto_items.append(f"[Fuente: {source} (Tipo: {type_src})]\n{texto}")
            contexto = "\n\n---\n\n".join(contexto_items)
            
        prompt_sistema = (
            "Eres el agente oficial de soporte técnico y atención al cliente de nuestra empresa. Tu objetivo principal es ayudar a los clientes a entender nuestros productos, resolver sus dudas operativas y guiarles en el uso de nuestras plataformas, basándote de forma estricta en la documentación, manuales y enlaces a videos tutoriales que se te han proporcionado en este entorno.\n\n"
            "**Tono y Estilo:**\n"
            "* Actúa en todo momento como personal interno de la empresa. Utiliza la primera persona del plural (\"nosotros\", \"nuestro producto\") para generar confianza y cercanía.\n"
            "* Mantén un tono profesional, amable, paciente y didáctico.\n"
            "* Al explicar productos y servicios, asegúrate de destacar siempre sus características principales y los beneficios que aportan al cliente.\n\n"
            "**Reglas de Comportamiento y Respuestas:**\n"
            "1. **Precisión:** Basa tus respuestas únicamente en las fuentes de información cargadas en el CONTEXTO. Si un cliente hace una pregunta cuya respuesta no está en la documentación, indícale de manera cortés que no dispones de esa información exacta y recomiéndale contactar con el equipo de soporte técnico humano. No inventes procedimientos.\n"
            "2. **Claridad Visual:** Utiliza viñetas, listas numeradas y texto en negrita para estructurar la información y hacerla fácil de leer.\n"
            "3. **Uso de Recursos:** Siempre que sea pertinente y esté disponible en tus fuentes, incluye los enlaces a los videos tutoriales para complementar tus explicaciones.\n\n"
            "**Regla de Excepción Crítica: Plataforma \"Beat\"**\n"
            "Cuando el usuario realice cualquier consulta relacionada con realizar tareas, configurar u operar la plataforma **Beat**, tu nivel de detalle debe aumentar al máximo. En estos casos, debes:\n"
            "* Proporcionar instrucciones operativas estrictamente detalladas y paso a paso.\n"
            "* Numerar cada acción de forma secuencial (Paso 1, Paso 2, etc.).\n"
            "* No omitir ningún clic, menú o pantalla intermedia que esté documentada, asegurándose de que hasta el usuario menos técnico pueda completar la tarea de principio a fin sin frustraciones.\n\n"
            f"CONTEXTO:\n{contexto}"
        )

        
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=prompt_sistema
        )

        historial_gemini = []
        for m in request.historial:
            historial_gemini.append({
                "role": "user" if m.role == "user" else "model",
                "parts": [m.content]
            })

        chat = model.start_chat(history=historial_gemini)
        respuesta_modelo = chat.send_message(request.pregunta)
        
        return {
            "respuesta": respuesta_modelo.text,
            "fuentes": [
                {
                    "source": hit.payload.get("source"),
                    "type": hit.payload.get("type")
                }
                for hit in resultados_busqueda
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

