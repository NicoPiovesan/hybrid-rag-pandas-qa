"""
Pipeline di Ingestione Incrementale per sistemi RAG (Retrieval-Augmented Generation).

Questo script monitora una directory locale, analizza i documenti (Testo, PDF, CSV) 
utilizzando la libreria `unstructured`, esegue il chunking intelligente e aggiorna 
un database vettoriale ChromaDB gestendo le modifiche ai file per evitare duplicati.
"""

import os
import json
import time
from dotenv import load_dotenv

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Caricamento delle variabili d'ambiente (es. API Keys)
load_dotenv()

# --- COSTANTI DI CONFIGURAZIONE ---
CARTELLA_DATI = "novatech_data"
FILE_STATO = "stato_ingestion.json"

def carica_stato() -> dict:
    """
    Carica lo stato dell'ingestione dal file JSON locale.
    
    Returns:
        dict: Dizionario contenente i nomi dei file e i relativi timestamp di ultima modifica.
    """
    if os.path.exists(FILE_STATO):
        with open(FILE_STATO, 'r') as f:
            return json.load(f)
    return {}

def salva_stato(stato: dict) -> None:
    """
    Salva lo stato corrente dell'ingestione su file JSON.
    
    Args:
        stato (dict): Dizionario aggiornato con i timestamp.
    """
    with open(FILE_STATO, 'w') as f:
        json.dump(stato, f, indent=4)

def processa_documenti_aziendali(cartella: str) -> list:
    """
    Analizza la cartella specificata ed elabora solo i file nuovi o modificati.
    Utilizza `unstructured` per il partizionamento e il chunking logico.
    
    Args:
        cartella (str): Percorso della cartella contenente i documenti raw.
        
    Returns:
        list: Lista di dizionari contenenti il testo dei chunk estratti e i relativi metadati.
    """
    tutti_i_chunk = []
    stato_attuale = carica_stato()
    stato_aggiornato = stato_attuale.copy()
    file_processati_in_questa_run = 0

    for nome_file in os.listdir(cartella):
        percorso_completo = os.path.join(cartella, nome_file)
        
        # Filtro per file di sistema e sub-directory
        if not os.path.isfile(percorso_completo) or nome_file.startswith('.'):
            continue
            
        mtime_corrente = os.path.getmtime(percorso_completo)

        # Controllo incrementale: salta il file se non ha subito modifiche
        if nome_file in stato_attuale and stato_attuale[nome_file] >= mtime_corrente:
            print(f"Saltato (Nessuna modifica rilevata): {nome_file}")
            continue

        print(f"\n--- Processando (Nuovo o Modificato): {nome_file} ---")
        
        try:
            # Parsing dinamico basato sull'estensione e struttura del file
            elementi_estratti = partition(filename=percorso_completo)
            # Chunking semantico basato sugli header (Titoli) del documento
            chunks_logici = chunk_by_title(elementi_estratti)
            
            for chunk in chunks_logici:
                dati_puliti = {
                    "testo": chunk.text,
                    "metadati": {
                        "source": nome_file,
                        "tipo_documento": chunk.metadata.filetype,
                        "categoria_strutturale": type(chunk).__name__
                    }
                }
                
                # Conservazione della struttura HTML per i dati tabellari
                if type(chunk).__name__ == "Table" and hasattr(chunk.metadata, "text_as_html"):
                    dati_puliti["testo_html"] = chunk.metadata.text_as_html

                tutti_i_chunk.append(dati_puliti)
                
            print(f"Successo: Estratti {len(chunks_logici)} chunk logici.")
            
            # Aggiornamento dello stato solo in caso di elaborazione completata senza errori
            stato_aggiornato[nome_file] = mtime_corrente
            file_processati_in_questa_run += 1
            
        except Exception as e:
            print(f"Errore durante l'ingestion di {nome_file}: {e}")

    salva_stato(stato_aggiornato)
    print(f"\nOperazione di parsing completata: elaborati {file_processati_in_questa_run} file modificati/nuovi.")
    
    return tutti_i_chunk

def aggiorna_vector_db(dati_pronti_per_db: list) -> None:
    """
    Gestisce la sincronizzazione dei chunk con il database vettoriale ChromaDB.
    Elimina le versioni obsolete dei documenti prima di inserire quelle nuove.
    
    Args:
        dati_pronti_per_db (list): Lista di dizionari contenenti dati e metadati da vettorializzare.
    """
    if not dati_pronti_per_db:
        print("\nNessun nuovo documento da aggiungere al Vector DB.")
        return

    print("\n--- AGGIORNAMENTO CHROMADB ---")
    
    # Inizializzazione della funzione di embedding (Gemini) e del client Chroma
    google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
        model_name="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
    )
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection = chroma_client.get_collection(name="test_collection_wemb2", embedding_function=google_ef)

    # Identificazione dei file aggiornati per la pulizia dei record preesistenti
    file_modificati = set(blocco["metadati"]["source"] for blocco in dati_pronti_per_db)
    
    for file_da_aggiornare in file_modificati:
        print(f"Pulizia vecchi record vettoriali per: {file_da_aggiornare}")
        collection.delete(where={"source": file_da_aggiornare})

    # Preparazione dei batch per l'inserimento
    docs = []
    metadati = []
    ids_ = []
    timestamp_id = int(time.time())

    for i, blocco in enumerate(dati_pronti_per_db):
        docs.append(blocco["testo"])
        metadati.append(blocco["metadati"])
        # Generazione ID univoco (Source + Indice + Timestamp) per prevenire collisioni
        ids_.append(f"{blocco['metadati']['source']}_chunk_{i}_{timestamp_id}")

    # Esecuzione dell'inserimento massivo nel database vettoriale
    collection.add(
        documents=docs,
        metadatas=metadati,
        ids=ids_
    )
    print(f"Salvataggio completato! Inseriti {len(docs)} chunk vettorializzati.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Verifica esistenza directory di input
    if not os.path.exists(CARTELLA_DATI):
        os.makedirs(CARTELLA_DATI)
        print(f"Creata directory: {CARTELLA_DATI}")
        
    chunk_estratti = processa_documenti_aziendali(CARTELLA_DATI)
    aggiorna_vector_db(chunk_estratti)