# Sistema RAG Ibrido con Analisi di Dati Strutturati

Un sistema di risposta alle domande basato su Python che combina la Generazione Aumentata dal Recupero (RAG) per documenti non strutturati con un Agente Pandas AI per l'analisi di dati CSV strutturati, dotato di una pipeline di ingestione completa e un'interfaccia web.

## Architettura

```
┌─────────────────┐     HTTP API      ┌──────────────────────────────────────┐
│   Streamlit UI  │ ───────────────►  │         Server FastAPI              │
│   (app_ui.py)   │ ◄───────────────  │       (api_server.py)               │
└─────────────────┘   Risposta JSON   │                                      │
                                      │  ┌────────────────────────────────┐  │
                                      │  │ Ramo 1: ChromaDB RAG          │  │
                                      │  │  - Ricerca vettoriale         │  │
                                      │  │  - LLM Ollama                 │  │
                                      │  └────────────────────────────────┘  │
                                      │  ┌────────────────────────────────┐  │
                                      │  │ Ramo 2: Agente Pandas AI      │  │
                                      │  │  - Esecuzione query CSV       │  │
                                      │  │  - Agente LangChain           │  │
                                      │  └────────────────────────────────┘  │
                                      └──────────────────────────────────────┘
                                                    │
                                                    ▼
                                      ┌──────────────────────────────────────┐
                                      │    Database Vettoriale ChromaDB     │
                                      │   (Embedding Google Gemini)         │
                                      └──────────────────────────────────────┘
                                                    ▲
                                                    │
                                      ┌──────────────────────────────────────┐
                                      │    Pipeline di Ingestione            │
                                      │    (ingestion_pipeline.py)           │
                                      │  - Monitora novatech_data/          │
                                      │  - Partiziona con `unstructured`    │
                                      │  - Suddivide e incorpora documenti  │
                                      └──────────────────────────────────────┘
```

## Componenti

### 1. Pipeline di Ingestione (`ingestion_pipeline.py`)
Sistema di elaborazione incrementale dei documenti per preparare i dati al RAG:

- **Monitora** la directory `novatech_data/` alla ricerca di file nuovi o modificati
- **Partiziona** i documenti (PDF, TXT, CSV, HTML) usando la libreria `unstructured`
- **Suddivide** i documenti semanticamente tramite `chunk_by_title()` per una separazione logica
- **Genera embedding** tramite le API di Embedding Google Gemini
- **Archivia** i vettori in ChromaDB con metadati (sorgente, tipo di file, categoria strutturale)
- **Gestione dello stato** tramite `stato_ingestion.json` per evitare la rielaborazione di file invariati
- **Deduplicazione** eliminando i vecchi record vettoriali prima di inserire il contenuto aggiornato

**Funzioni principali:**
- `carica_stato()` / `salva_stato()` – Persistenza dei timestamp di modifica dei file
- `processa_documenti_aziendali()` – Analisi e suddivisione incrementale dei documenti
- `aggiorna_vector_db()` – Sincronizzazione dei chunk su ChromaDB con pulizia

### 2. Server API (`api_server.py`)
Backend FastAPI che implementa un sistema ibrido di instradamento delle query:

**Due rami di query:**
- **Ramo 1 (RAG)**: Ricerca per similarità vettoriale su ChromaDB → recupero del contesto → generazione con LLM Ollama
- **Ramo 2 (Agente Pandas)**: Agente LangChain su DataFrame Pandas → esecuzione di codice Python → analisi CSV

**Endpoint:** `POST /ask`
- **Corpo della richiesta:**
  - `domanda` (str): Query in linguaggio naturale dell'utente
  - `storico_conversazione` (str): Contesto della conversazione precedente
  - `usa_agente_dati` (bool): Flag di instradamento (`True` → Agente Pandas, `False` → RAG)

- **Risposta:**
  - `risposta` (str): Risposta generata
  - `nuovo_storico` (str): Storico della conversazione aggiornato
  - `fonti` (list[str]): Documenti sorgente utilizzati
  - `motore_usato` (str): Nome del motore utilizzato ("ChromaDB RAG" o "Pandas AI Agent")

**Configurazione:**
- Endpoint Ollama: `http://192.168.1.116:11434`
- Modello: `gemma4:latest`
- ChromaDB: `localhost:8000`, collezione `test_collection_wemb2`
- Agente Pandas: Temperatura `0` per la generazione deterministica del codice

### 3. Interfaccia Web (`app_ui.py`)
Frontend Streamlit che offre un'interfaccia di chat interattiva:

- **Selettore di dominio**: Pulsante radio per instradare le query verso "Documenti Testuali" (RAG) o "Dati Aziendali" (Pandas)
- **Interfaccia di chat**: Visualizzazione dei messaggi in tempo reale con ruoli utente/assistente
- **Stato di sessione**: Mantiene `messaggi_chat` (visuale) e `storico_invisibile` (contesto per il LLM)
- **Comunicazione API**: Richieste POST a `http://127.0.0.1:8080/ask`
- **Gestione degli errori**: Errori di connessione e risposte HTTP errate

### 4. Script di Test (`unstructuredTest.py`)
Script sperimentale per testare la libreria `unstructured` e la pipeline RAG:

- Recupera URL di articoli da CNN Lite
- Carica documenti web tramite `UnstructuredURLLoader`
- Crea un vectorstore Chroma con embedding di Google Generative AI
- Esegue ricerche per similarità su argomenti (es. "Aggiornamento sul colpo di stato in Niger")
- Riassume i risultati usando Ollama LLM con `load_summarize_chain` di LangChain

## Installazione

### Prerequisiti
- Python 3.8+
- [Ollama](https://ollama.com/) in esecuzione localmente con il modello `gemma4:latest`
- Server ChromaDB in esecuzione su `localhost:8000`
- Chiave API Google per gli embedding Gemini

### Dipendenze

```bash
pip install fastapi uvicorn streamlit requests
pip install chromadb
pip install langchain-experimental langchain-ollama langchain-community langchain-google-genai
pip install unstructured
pip install pandas python-dotenv
pip install google-generativeai
```

### Variabili d'Ambiente

Crea un file `.env` nella directory principale del progetto:

```
GOOGLE_API_KEY=la_tua_chiave_api_gemini
```

## Utilizzo

### 1. Avviare il server ChromaDB
```bash
chromadb run --host localhost --port 8000
```

### 2. Avviare il server API
```bash
uvicorn api_server:app --host 127.0.0.1 --port 8080 --reload
```

### 3. Eseguire la pipeline di ingestione (per popolare ChromaDB)
```bash
python ingestion_pipeline.py
```
Inserisci i documenti (PDF, TXT, CSV) nella directory `novatech_data/`.

### 4. Avviare l'interfaccia web
```bash
streamlit run app_ui.py
```
Apri `http://localhost:8501` nel tuo browser.

## Esempi di Flusso Dati

**Query: "Quali sono le politiche aziendali sul lavoro da remoto?"** (Ramo RAG)
1. L'interfaccia invia `usa_agente_dati=False` all'API
2. L'API interroga ChromaDB per i chunk di documenti di policy pertinenti
3. Il contesto viene iniettato nel prompt inviato a Ollama
4. Ollama genera la risposta basandosi esclusivamente sul contesto recuperato
5. La risposta include i nomi dei file sorgente

**Query: "Qual è il fatturato totale degli ordini completati?"** (Ramo Pandas)
1. L'interfaccia invia `usa_agente_dati=True` all'API
2. L'agente Pandas riceve la query e il DataFrame CSV
3. L'agente genera ed esegue codice Python (groupby, sum, ecc.)
4. Il risultato viene formattato come "Risposta Finale: ..." in italiano
5. La risposta include "File CSV (Analisi Pandas)" come sorgente

## Tecnologie

| Componente | Tecnologia |
|------------|------------|
| Framework API | FastAPI |
| Framework UI | Streamlit |
| Database Vettoriale | ChromaDB |
| Embedding | Google Gemini Embedding API |
| LLM (Generazione Testo) | Ollama (Gemma 4) |
| Elaborazione Documenti | Libreria `unstructured` |
| Agente Dati Strutturati | LangChain Pandas DataFrame Agent |
| Manipolazione Dati | Pandas |

## Struttura dei File

```
.
├── api_server.py              # Backend FastAPI con routing ibrido RAG/Pandas
├── app_ui.py                  # Interfaccia web Streamlit
├── ingestion_pipeline.py      # Ingestione documenti e popolamento del DB vettoriale
├── unstructuredTest.py        # Script di test per la libreria unstructured
├── novatech_data/             # Directory per i documenti sorgente
│   ├── employees.csv          # Dati strutturati di esempio
│   └── customers_orders.csv   # Dati strutturati di esempio
├── stato_ingestion.json       # File di stato per l'elaborazione incrementale
└── .env                       # Variabili d'ambiente (GOOGLE_API_KEY)
```

## Note

- L'agente Pandas viene eseguito con `allow_dangerous_code=True`
- La temperatura del LLM Ollama è impostata a `0` per output deterministici nel ramo Pandas
- La collezione ChromaDB utilizza gli embedding Google Gemini con `task_type="RETRIEVAL_DOCUMENT"`
- Lo storico della conversazione viene mantenuto come stringa e passato al LLM per il contesto
- La pipeline di ingestione elabora solo i file con timestamp di modifica aggiornati (aggiornamenti incrementali)