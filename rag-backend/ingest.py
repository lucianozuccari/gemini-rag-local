import os
import re
import json
import time
import requests
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

def embed_content_with_retry(model, content, task_type):
    for i in range(5):
        try:
            return genai.embed_content(
                model=model,
                content=content,
                task_type=task_type
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Quota" in err_msg or "limit" in err_msg or "ResourceExhausted" in err_msg:
                sleep_time = (i + 1) * 6
                print(f"Límite de cuota alcanzado para embeddings. Esperando {sleep_time} segundos antes de reintentar...")
                time.sleep(sleep_time)
            else:
                raise e
    raise Exception("Excedido el número máximo de reintentos para generar el embedding.")


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

# Configurar API de Gemini y Qdrant
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
qdrant_host = os.environ.get("QDRANT_HOST")
if qdrant_host:
    client = QdrantClient(host=qdrant_host, port=6333)
else:
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=1.0)
        client.get_collections()
    except Exception:
        db_path = os.path.join(os.path.dirname(__file__), "qdrant_local_data")
        print(f"Qdrant no detectado en localhost. Usando almacenamiento local integrado en: {db_path}")
        client = QdrantClient(path=db_path)


COLLECTION_NAME = "manuales_empresa"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documentos")

def inicializar_base_datos():
    print("Inicializando base de datos Qdrant...")
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"Colección '{COLLECTION_NAME}' creada.")
    else:
        print(f"Colección '{COLLECTION_NAME}' ya existe.")

def chunk_text(text, max_chars=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def parse_pdf(file_path):
    print(f"Parseando PDF: {file_path}")
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        print(f"Error al leer PDF {file_path}: {e}")
    return text

def parse_txt(file_path):
    print(f"Parseando TXT: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error al leer TXT {file_path}: {e}")
        return ""

def get_playlist_video_ids(playlist_id):
    print(f"Buscando videos de la lista de reproducción: {playlist_id}")
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if res.status_code == 200:
            video_ids = list(set(re.findall(r'\"videoId\":\"([a-zA-Z0-9_-]{11})\"', res.text)))
            print(f"Se encontraron {len(video_ids)} videos en la lista.")
            return video_ids
    except Exception as e:
        print(f"Error al obtener playlist {playlist_id}: {e}")
    return []

def get_youtube_transcript(video_id):
    try:
        res = YouTubeTranscriptApi().fetch(video_id, languages=['es', 'en'])
        return " ".join([t.text for t in res.snippets])
    except Exception as e:
        print(f"No se pudo descargar transcripción para video {video_id}: {e}")
        return None


def indexar_documentos():
    points = []
    point_id = 0

    # 1. Procesar archivos locales (.txt y .pdf) en la carpeta 'documentos'
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if os.path.isdir(file_path):
                continue
            
            content = ""
            file_type = ""
            if filename.endswith(".pdf"):
                content = parse_pdf(file_path)
                file_type = "pdf"
            elif filename.endswith(".txt") and filename != "README.md":
                content = parse_txt(file_path)
                file_type = "txt"
            
            if content.strip():
                chunks = chunk_text(content)
                print(f"Indexando {len(chunks)} fragmentos de {filename}...")
                for chunk in chunks:
                    try:
                        resultado = embed_content_with_retry(
                            model="models/gemini-embedding-2",
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        points.append(PointStruct(
                            id=point_id,
                            vector=resultado['embedding'],
                            payload={
                                "texto": chunk,
                                "source": filename,
                                "type": file_type
                            }
                        ))
                        point_id += 1
                    except Exception as e:
                        print(f"Error al generar embedding para fragmento de {filename}: {e}")

    # 2. Procesar listas de YouTube si se configura un archivo JSON
    playlists_config = os.path.join(os.path.dirname(__file__), "youtube_playlists.json")
    if os.path.exists(playlists_config):
        try:
            with open(playlists_config, "r", encoding="utf-8") as f:
                playlists = json.load(f)
            
            for playlist_id in playlists:
                video_ids = get_playlist_video_ids(playlist_id)
                for video_id in video_ids:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    print(f"Procesando transcripción de: {video_url}")
                    transcript = get_youtube_transcript(video_id)
                    if transcript:
                        chunks = chunk_text(transcript)
                        print(f"Indexando {len(chunks)} fragmentos de video {video_id}...")
                        for chunk in chunks:
                            try:
                                resultado = embed_content_with_retry(
                                    model="models/gemini-embedding-2",
                                    content=chunk,
                                    task_type="retrieval_document"
                                )
                                points.append(PointStruct(
                                    id=point_id,
                                    vector=resultado['embedding'],
                                    payload={
                                        "texto": chunk,
                                        "source": video_url,
                                        "type": "youtube"
                                    }
                                ))
                                point_id += 1
                            except Exception as e:
                                print(f"Error al generar embedding para video {video_id}: {e}")
        except Exception as e:
            print(f"Error al procesar configuración de playlists: {e}")

    if points:
        print(f"Subiendo {len(points)} puntos/vectores a Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("Indexación completada con éxito.")
    else:
        print("No se encontraron documentos ni videos para indexar.")

if __name__ == "__main__":
    inicializar_base_datos()
    indexar_documentos()
