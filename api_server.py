"""
API FastAPI per architettura ibrida di interrogazione dati.
Integra un sistema RAG (Retrieval-Augmented Generation) basato su ChromaDB per documenti non strutturati
e un Agente Pandas tramite LangChain per l'analisi di dati strutturati (CSV).
Utilizza Ollama come provider LLM locale.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import requests
import os
import pandas as pd
from dotenv import load_dotenv

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import OllamaLLM

# Caricamento variabili d'ambiente
load_dotenv()

# --- CONFIGURAZIONE ---
IP_PC_FISSO = "192.168.1.116"
URL_OLLAMA_GENERATE = f"http://{IP_PC_FISSO}:11434/api/generate"
URL_OLLAMA_BASE = f"http://{IP_PC_FISSO}:11434"
MODELLO = "gemma4:latest"

app = FastAPI(title="NovaTech RAG API (Hybrid)", version="2.0")

# --- INIZIALIZZAZIONE RISORSE GLOBALI ---
# Le risorse vengono caricate in memoria all'avvio dell'applicazione per ridurre la latenza delle richieste.

# 1. Configurazione ChromaDB per documenti testuali
google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_collection(name="test_collection_wemb2", embedding_function=google_ef)

# 2. Configurazione Agente Pandas per dati strutturati (CSV)
try:
    percorso_csv = "novatech_data/ordini_clienti.csv"
    df_aziendale = pd.read_csv(percorso_csv)
    
    # Inizializzazione del modello LLM per LangChain (temperatura 0 per massimizzare il determinismo del codice generato)
    llm_agente = OllamaLLM(
        model=MODELLO, 
        base_url=URL_OLLAMA_BASE,
        temperature=0
    )
    
    agente_dati = create_pandas_dataframe_agent(
        llm_agente, 
        df_aziendale, 
        verbose=True, 
        allow_dangerous_code=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    print("Agente Pandas inizializzato con successo.")
except Exception as e:
    print(f"ATTENZIONE: Inizializzazione Agente Pandas fallita. Motivo: {e}")
    agente_dati = None


# --- MODELLI DATI (Pydantic) ---
class QueryRequest(BaseModel):
    """Schema per la validazione della richiesta in ingresso."""
    domanda: str = Field(..., description="La query testuale dell'utente.")
    storico_conversazione: str = Field(default="", description="Contesto delle interazioni precedenti per il mantenimento della memoria.")
    usa_agente_dati: bool = Field(default=False, description="Flag di routing: True per interrogare il CSV, False per il database vettoriale.")

class QueryResponse(BaseModel):
    """Schema per la formattazione della risposta in uscita."""
    risposta: str
    nuovo_storico: str
    fonti: list[str]
    motore_usato: str


# --- ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
def chiedi_all_assistente(request: QueryRequest):
    """
    Gestisce la query dell'utente instradandola al motore appropriato in base al parametro `usa_agente_dati`.
    
    Flussi:
    - usa_agente_dati=True -> Analisi dati strutturati tramite Pandas AI Agent.
    - usa_agente_dati=False -> Ricerca semantica tramite ChromaDB e generazione testo tramite Ollama.
    """
    
    # ==========================================
    # RAMO 1: ANALISI DATI STRUTTURATI (CSV)
    # ==========================================
    if request.usa_agente_dati:
        if not agente_dati:
            raise HTTPException(status_code=500, detail="Servizio non disponibile: Agente Pandas non inizializzato correttamente.")
            
        try:
            # Prompt ingegnerizzato per forzare l'agente a produrre output deterministici e conformi
            istruzione_pulita = (
                f"Rispondi alla seguente domanda: {request.domanda}\n\n"
                f"REGOLE TASSATIVE DI FORMATTAZIONE:\n"
                f"1. Se devi eseguire del codice per trovare la risposta, scrivi SOLO l'azione e fermati.\n"
                f"2. Se conosci già la risposta, inizia la frase ESATTAMENTE con 'Final Answer: ' seguito dalla tua risposta in lingua italiana.\n"
                f"3. NON INSERIRE MAI 'Action' e 'Final Answer' nello stesso messaggio. Usa sempre 'Final Answer: ' per la risposta definitiva."
            )            
            risposta_agente = agente_dati.invoke(istruzione_pulita)
            testo_generato = risposta_agente["output"]
            fonti_trovate = ["File CSV (Analisi Pandas)"]
            motore = "Pandas AI Agent"
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore di esecuzione dell'Agente Pandas: {e}")

    # ==========================================
    # RAMO 2: RETRIEVAL-AUGMENTED GENERATION (RAG)
    # ==========================================
    else:
        # Recupero del contesto rilevante dal Vector DB
        try:
            risultati = collection.query(
                query_texts=[request.domanda],
                n_results=2,
                include=["documents", "metadatas"]
            )
            testi_trovati = risultati["documents"][0]
            fonti_trovate = [meta.get('source', 'Sconosciuta') for meta in risultati['metadatas'][0]]
            contesto_recuperato = "\n\n".join(testi_trovati)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore di query sul database vettoriale: {e}")

        # Costruzione del prompt basato sul contesto recuperato
        prompt_raw = (
            f"Sei un assistente aziendale oggettivo. Usa ESCLUSIVAMENTE il seguente contesto per rispondere.\n\n"
            f"--- INIZIO CONTESTO ---\n{contesto_recuperato}\n--- FINE CONTESTO ---\n\n"
            f"Domanda dell'utente: {request.domanda}\nRisposta:"
        )

        # Chiamata API diretta a Ollama
        payload = {"model": MODELLO, "prompt": prompt_raw, "stream": False}
        
        try:
            risposta_ollama = requests.post(URL_OLLAMA_GENERATE, json=payload, timeout=120)
            risposta_ollama.raise_for_status()
            testo_generato = risposta_ollama.json().get("response", "").strip()
            motore = "ChromaDB RAG"
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Errore di comunicazione con il servizio Ollama: {e}")

    # --- AGGIORNAMENTO STATO ---
    # Append della transazione corrente allo storico della conversazione
    nuovo_storico = request.storico_conversazione + f"Utente: {request.domanda}\nAssistente: {testo_generato}\n\n"

    return QueryResponse(
        risposta=testo_generato,
        nuovo_storico=nuovo_storico,
        fonti=fonti_trovate,
        motore_usato=motore
    )