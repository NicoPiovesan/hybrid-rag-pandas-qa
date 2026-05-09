"""
NovaTech Assistant - Frontend UI

Questo script implementa l'interfaccia utente (UI) per l'assistente aziendale NovaTech
utilizzando Streamlit. L'applicazione funge da client per un'API RESTful (FastAPI) 
sottostante, gestendo l'input dell'utente, lo stato della sessione (storico chat) e 
l'indirizzamento delle richieste verso due motori distinti:
1. RAG Tradizionale (ChromaDB) per documenti testuali (PDF, Policy).
2. Pandas AI Agent per l'analisi di dati strutturati (CSV, Ordini).
"""

import streamlit as st
import requests

# --- CONFIGURAZIONE AMBIENTE ---
# Indirizzo del backend FastAPI in esecuzione in locale. 
# In produzione, sostituire con l'URL del server (es. https://api.novatech.com/ask)
API_URL = "http://127.0.0.1:8080/ask"

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    page_title="NovaTech Assistant", 
    page_icon="🏢", 
    layout="centered"
)
st.title("🏢 Assistente Aziendale NovaTech")
st.caption("Interroga le policy aziendali o analizza i dati degli ordini in linguaggio naturale.")

# --- INIZIALIZZAZIONE DELLA MEMORIA (SESSION STATE) ---
# Lo stato della sessione è necessario per mantenere la persistenza dei dati 
# durante i ricaricamenti (re-run) automatici della pagina operati da Streamlit.

# 1. Memoria Visiva: Lista dei dizionari contenenti i messaggi da renderizzare nella UI
if "messaggi_chat" not in st.session_state:
    st.session_state.messaggi_chat = []
    
# 2. Memoria di Contesto: Stringa invisibile che mantiene lo storico per l'LLM
if "storico_invisibile" not in st.session_state:
    st.session_state.storico_invisibile = ""

# --- COMPONENTI UI: SELETTORE DEL MOTORE DI RICERCA ---
# Permette all'utente di definire il routing della query (Testo vs Dati Strutturati)
tipo_interrogazione = st.radio(
    "Seleziona il dominio di analisi:",
    ("Documenti Testuali (PDF, Policy)", "Dati Aziendali (Ordini, CSV)"),
    horizontal=True,
    help="Determina se l'assistente userà la ricerca vettoriale o l'agente di analisi dati."
)

# --- RENDERIZZAZIONE DELLO STORICO CHAT ---
# Ricostruisce visivamente la conversazione ad ogni re-run dell'applicazione
for messaggio in st.session_state.messaggi_chat:
    with st.chat_message(messaggio["ruolo"]):
        st.markdown(messaggio["contenuto"])

# --- GESTIONE DELL'INPUT UTENTE E CHIAMATA API ---
domanda_utente = st.chat_input("Scrivi la tua domanda qui...")

if domanda_utente:
    # 1. Renderizzazione immediata dell'input utente
    with st.chat_message("user"):
        st.markdown(domanda_utente)
    
    # Salvataggio nello stato della UI
    st.session_state.messaggi_chat.append({"ruolo": "user", "contenuto": domanda_utente})

    # 2. Renderizzazione del blocco di risposta dell'assistente con indicatore di caricamento
    with st.chat_message("assistant"):
        placeholder_risposta = st.empty()
        placeholder_risposta.markdown("⏳ Elaborazione della richiesta in corso...")
        
        # 3. Costruzione del Payload JSON per la chiamata POST
        payload_api = {
            "domanda": domanda_utente,
            "storico_conversazione": st.session_state.storico_invisibile,
            # Risoluzione del flag booleano in base alla selezione del radio button
            "usa_agente_dati": True if "Dati Aziendali" in tipo_interrogazione else False
        }
        
        try:
            # 4. Esecuzione della chiamata HTTP al backend FastAPI
            response = requests.post(API_URL, json=payload_api)
            
            # Gestione del successo (HTTP 200 OK)
            if response.status_code == 200:
                dati = response.json()
                risposta_testo = dati["risposta"]
                fonti = dati["fonti"]
                
                # Sincronizzazione dello storico invisibile restituito dal backend
                st.session_state.storico_invisibile = dati["nuovo_storico"]
                
                # Formattazione della risposta finale integrando i metadati delle fonti
                # Utilizza un set per rimuovere eventuali fonti duplicate
                risposta_formattata = f"{risposta_testo}\n\n---\n*Fonti consultate: {', '.join(set(fonti))}*"
                
                # Aggiornamento della UI e salvataggio nello stato
                placeholder_risposta.markdown(risposta_formattata)
                st.session_state.messaggi_chat.append({"ruolo": "assistant", "contenuto": risposta_formattata})
                
            # Gestione degli errori logici restituiti dal backend (es. HTTP 400/500)
            else:
                errore_msg = f"⚠️ Errore dal server API: {response.json().get('detail', 'Errore sconosciuto')}"
                placeholder_risposta.error(errore_msg)
                
        # Gestione degli errori di rete (backend non raggiungibile)
        except requests.exceptions.ConnectionError:
            placeholder_risposta.error("⚠️ Impossibile contattare il backend. Verificare che l'API FastAPI (uvicorn) sia in esecuzione sulla porta 8080.")