import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
import ingest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BACKEND_DIR, "documentos")
playlists_config = os.path.join(BACKEND_DIR, "youtube_playlists.json")

if os.path.exists(playlists_config):
    with open(playlists_config, "r", encoding="utf-8") as f:
        playlists = json.load(f)
    
    for playlist_id in playlists:
        video_ids = ingest.get_playlist_video_ids(playlist_id)
        for video_id in video_ids:
            filename = f"video_youtube_{video_id}.txt"
            file_path = os.path.join(DOCS_DIR, filename)
            
            # Si ya existe localmente, no lo volvemos a descargar
            if os.path.exists(file_path):
                print(f"Ya existe localmente (omitido): {filename}")
                continue
                
            print(f"Descargando transcripción para {video_id}...")
            transcript = ingest.get_youtube_transcript(video_id)
            if transcript:
                with open(file_path, "w", encoding="utf-8") as out:
                    out.write(f"Título/URL del Video: https://www.youtube.com/watch?v={video_id}\n\n")
                    out.write(transcript)
                print(f"Guardado exitosamente: {filename}")
            else:
                print(f"No se pudo descargar {video_id}. Se omitirá por ahora.")
else:
    print(f"No se encontró el archivo de configuración en {playlists_config}")
