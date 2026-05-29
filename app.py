import streamlit as st

from agents.router import AgentRouter
from services.memory_service import MemoryService
from services.data_service import DataService
from services.ollama_service import OllamaService
from ui.components import render_chat_history, render_portfolio_summary
from ui.charts import allocation_pie_chart

st.set_page_config(page_title="Casablanca Portfolio Chatbot", page_icon="💬", layout="wide")

st.title("Casablanca Portfolio Chatbot")
st.caption("Assistant de portefeuille pour la Bourse de Casablanca")

MemoryService.initialize()

mode = st.radio(
    "Mode de l'agent",
    ["Local", "Ollama"],
    index=0,
    horizontal=True,
)

if mode == "Ollama" and not OllamaService().is_available():
    st.warning(
        "Ollama n'est pas installé ou n'est pas accessible. Installez le package `ollama`, vérifiez que l’outil Ollama est installé et exécutez `ollama pull mistral`."
    )

router = AgentRouter()

data_service = DataService()
try:
    data_service.load_price_data()
    st.success("Données locales détectées dans data/raw/stock.")
except Exception as exc:
    st.error(f"Aucune donnée exploitable détectée : {exc}")

render_chat_history(MemoryService.get_history())

st.markdown(
    "Entrez votre demande sous forme de texte, par exemple : `12% rendement 15% risque 100000 MAD`."
)

prompt = st.chat_input("Exemple : Je veux 12% de rendement avec 18% de risque et 100000 MAD")

if prompt:
    # Affiche immédiatement la bulle utilisateur (évite duplication avec l'historique)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                agent_mode = "ollama" if mode == "Ollama" else "local"
                response = router.route(agent_mode, prompt)
            except Exception as exc:
                response = None
                st.error(f"Erreur : {exc}")

    if response is not None:
        # Enregistre la paire user/assistant une seule fois en mémoire
        MemoryService.append_message("user", prompt)
        MemoryService.append_message("assistant", response.content)
        with st.chat_message("assistant"):
            st.markdown(response.content)
        if response.structured is not None:
            st.divider()
            render_portfolio_summary(response.structured)
            st.plotly_chart(
                allocation_pie_chart(response.structured.allocations),
                use_container_width=True,
            )
