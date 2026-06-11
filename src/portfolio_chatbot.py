import glob
import os
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import random
import difflib
import google.generativeai as genai

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_GLOB = BASE_DIR / "data" / "raw" / "stock" / "*.csv"
DEFAULT_INFO_CSV = BASE_DIR / "data" / "raw" / "info.csv"
DEFAULT_ENV_FILE = BASE_DIR / ".env"

def _load_dotenv(env_file: Path = DEFAULT_ENV_FILE) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

def _load_company_labels(info_csv: Path) -> dict[str, str]:
    if not info_csv.exists():
        return {}
    labels: dict[str, str] = {}
    df_info = pd.read_csv(info_csv)
    symbol_col = "Symbol" if "Symbol" in df_info.columns else df_info.columns[0]
    name_col = "Asset" if "Asset" in df_info.columns else df_info.columns[min(1, len(df_info.columns) - 1)]
    extra_col = None
    if "Type" in df_info.columns:
        extra_col = "Type"
    elif len(df_info.columns) > 2:
        extra_col = df_info.columns[2]
    for _, row in df_info.iterrows():
        ticker = str(row[symbol_col]).strip()
        name = str(row[name_col]).strip()
        extra = str(row[extra_col]).strip() if extra_col else ""
        labels[ticker] = f"{name} ({extra})" if extra else name
    return labels

def _load_price_frame(stock_glob: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_path in glob.glob(str(stock_glob)):
        ticker = Path(file_path).stem
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        date_col = None
        for candidate in ("Date", "Time", "Datetime", "timestamp"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None or "Close" not in df.columns:
            continue
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
        except Exception:
            continue
        frames.append(df[["Close"]].rename(columns={"Close": ticker}))
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill()
    threshold = max(1, int(len(prices) * 0.5))
    return prices.dropna(axis=1, thresh=threshold)

def data_files_available() -> bool:
    return bool(glob.glob(str(DEFAULT_STOCK_GLOB)))

def get_data_status_message() -> str:
    message = (
        "Donnees detectees dans data/raw/stock."
        if data_files_available()
        else (
            "Aucune donnee detectee dans data/raw/stock. "
            "Ajoute tes CSV de prix et, si possible, data/raw/info.csv."
        )
    )
    if bool(os.getenv("GEMINI_API_KEY")):
        message += "\nAPI Gemini configuree avec succes."
    return message

def _extract_parameters(user_input: str) -> tuple[float | None, float | None, float | None]:
    rendement, risque, capital = None, None, None
    pct_matches = re.findall(r"(\d+(?:[.,]\d+)?)\s*%", user_input)
    if len(pct_matches) >= 2:
        rendement = float(pct_matches[0].replace(",", "."))
        risque = float(pct_matches[1].replace(",", "."))
    text_without_pct = re.sub(r"\d+(?:[.,]\d+)?\s*%", "", user_input)
    other_nums = re.findall(r"\d+(?:[.,]\d+)?", text_without_pct)
    for num_str in other_nums:
        val = float(num_str.replace(",", "."))
        if val >= 100:
            capital = val
            break
    if rendement is None and risque is None:
        all_nums = re.findall(r"\d+(?:[.,]\d+)?", user_input)
        vals = [float(x.replace(",", ".")) for x in all_nums]
        if len(vals) >= 2:
            pcts = [v for v in vals if v < 100]
            caps = [v for v in vals if v >= 100]
            if len(pcts) >= 2:
                rendement, risque = pcts[0], pcts[1]
            if caps and capital is None:
                capital = caps[0]
    return rendement, risque, capital

def _format_allocations_markdown(allocations: pd.DataFrame, latest_prices: pd.Series | None = None, capital: float | None = None) -> str:
    try:
        # Ensure DataFrame has clean indices to avoid "None of [index(...)] are in the [index]" errors
        allocations = allocations.copy().reset_index(drop=True)
        if allocations.empty:
            return "Aucune allocation significative n'a été retenue."
        if capital is None or capital <= 0 or latest_prices is None:
            lines = ["| Entreprise | Ticker | Poids Cible |"]
            lines.append("|---|---|---:|")
            for _, row in allocations.iterrows():
                lines.append(f"| {row['Entreprise']} | {row['Ticker']} | {row['Poids (%)']:.2f}% |")
            return "\n".join(lines)

        prices_dict = latest_prices.to_dict() if isinstance(latest_prices, pd.Series) else dict(latest_prices)
        items = []
        for _, row in allocations.iterrows():
            ticker = row['Ticker']
            price = prices_dict.get(ticker, 0.0)
            target_weight = float(row['Poids (%)'])
            items.append({
                'Entreprise': row['Entreprise'],
                'Ticker': ticker,
                'Prix': price,
                'Cible (%)': target_weight,
                'Shares': 0
            })

        cash_restant = capital
        for item in items:
            if item['Prix'] > 0:
                target_amount = capital * item['Cible (%)'] / 100
                shares = int(target_amount // item['Prix'])
                item['Shares'] = shares
                cash_restant -= shares * item['Prix']

        items.sort(key=lambda x: x['Cible (%)'], reverse=True)
        can_buy = True
        while can_buy:
            can_buy = False
            for item in items:
                if item['Prix'] > 0 and cash_restant >= item['Prix']:
                    item['Shares'] += 1
                    cash_restant -= item['Prix']
                    can_buy = True

        lines = ["| Entreprise | Ticker | Prix unitaire | Poids Réel | Quantité | Montant Investi |"]
        lines.append("|---|---|---:|---:|---:|---:|")
        
        total_invested = capital - cash_restant

        for item in items:
            if item['Shares'] > 0:
                actual_amount = item['Shares'] * item['Prix']
                actual_weight = (actual_amount / capital) * 100
                lines.append(f"| {item['Entreprise']} | {item['Ticker']} | {item['Prix']:,.2f} MAD | {actual_weight:.2f}% | {item['Shares']} | **{actual_amount:,.2f} MAD** |")

        lines.append(f"\n💡 **Bilan du portefeuille :**")
        lines.append(f"- **Capital initial :** {capital:,.2f} MAD")
        lines.append(f"- **Montant réellement investi :** {total_invested:,.2f} MAD")
        lines.append(f"- **Liquidité restante (Cash) :** {cash_restant:,.2f} MAD")
        
        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"Erreur lors du formatage des allocations : {str(e)}\n{traceback.format_exc()}"

def _fetch_ai_response(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Erreur : La variable GEMINI_API_KEY est introuvable."
    try:
        genai.configure(api_key=api_key)
        # Utilisation de la version la plus stable 1.5
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur de l'API : {str(e)}"

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_ai_prompt(user_input: str, context: str | None = None) -> str:
    prompt = "Tu es un assistant financier specialise dans la Bourse de Casablanca. Reponds en francais."
    if context:
        prompt += f"\n\nInformations de contexte :\n{context}"
    prompt += f"\n\nQuestion : {user_input}\nReponse :"
    return prompt

def _fuzzy_contains_any(text: str, keywords: list[str], threshold: float = 0.72) -> bool:
    normalized = _normalize_text(text)
    tokens = normalized.split()
    for kw in keywords:
        kw_norm = _normalize_text(kw)
        if not kw_norm:
            continue
        if kw_norm in normalized:
            return True
        if difflib.SequenceMatcher(None, normalized, kw_norm).ratio() >= threshold:
            return True
        for t in tokens:
            if difflib.SequenceMatcher(None, t, kw_norm).ratio() >= threshold:
                return True
    return False

GREETING_RESPONSES = ["Bonjour ! Je suis ton conseiller financier spécialisé dans la Bourse de Casablanca. 🇲🇦\nIndique-moi quel rendement tu vises (ex: 12%) et le risque maximal que tu tolères (ex: 15%).", "Salut ! Prêt à optimiser tes investissements ? Donne-moi ton rendement cible et ton risque max.", "Coucou ! Je suis connecté aux données du MASI. Que cherches-tu aujourd'hui ?", "Bonjour à toi ! Installe-toi confortablement. Dis-moi, quel est ton objectif de profit et ta limite de perte ?"]
EXPLAIN_RESPONSES = ["J'utilise la Théorie de Markowitz. Je prends tes fichiers CSV, je calcule comment les actions bougent entre elles, puis je simule 5000 portefeuilles pour trouver le meilleur Ratio de Sharpe.", "C'est très mathématique : je calcule le rendement annualisé et la volatilité de chaque entreprise de la bourse de Casablanca, puis je trouve la combinaison parfaite pour toi."]
GENERAL_RESPONSES = ["C'est noté. Si tu veux qu'on passe à l'action, donne-moi simplement tes chiffres : rendement et risque.", "Intéressant. N'hésite pas à me fournir un pourcentage de rendement et de risque pour que je lance l'algorithme.", "Je suis un bot orienté données. Donne-moi tes contraintes de risque et je te trouve les meilleures actions marocaines !"]
CLARIFY_RESPONSES = ["Je n'ai pas bien compris. 🧐 Peux-tu reformuler avec des chiffres clairs ? (ex: '10% de rendement, 15% de risque').", "Hmm, il me manque tes paramètres quantitatifs. Précise ton rendement en % et ton risque max en % s'il te plaît."]
JOKE_RESPONSES = ["Pourquoi les courtiers en bourse sont de mauvais jardiniers ? Parce qu'ils paniquent dès que les actions baissent ! 🌱📉", "C'est l'histoire d'un investisseur qui demande à son ami : 'Comment as-tu fait fortune en bourse ?'. L'ami répond : 'J'ai commencé avec une grosse fortune, et j'en ai perdu la moitié !' 😅", "Que dit une action marocaine quand elle est fatiguée ? 'Je crois que je vais faire une petite cotation...' 😴"]
FRUSTRATION_RESPONSES = ["Je suis désolé si mes réponses ne sont pas parfaites. Je suis un algorithme en apprentissage. Essaie de me donner des pourcentages stricts pour que je t'aide mieux.", "Restons calmes ! 🧘‍♂️ Mon but est de t'aider avec les mathématiques financières. Dis-moi précisément ce qui bloque ou relance une demande avec tes rendements.", "Je comprends ta frustration. L'algorithme est sensible à la façon dont les questions sont posées. Essaie de dire : '12% rendement et 15% risque'."]
IDENTITY_RESPONSES = ["Je suis un agent conversationnel d'intelligence artificielle. J'ai été conçu spécifiquement pour analyser la Bourse de Casablanca et optimiser des portefeuilles.", "Je suis ton assistant quantitatif personnel ! Je ne ressens rien, mais je calcule très vite les matrices de covariance. 🤖", "Je suis un programme Python spécialisé en finance marocaine, boosté au Machine Learning et à l'optimisation mathématique."]
FINANCE_DEF_RESPONSES = ["En finance, le **rendement** est l'argent que tu gagnes sur un an. Le **risque** (ou volatilité) mesure à quel point le prix de l'action fait le yoyo. Plus ça bouge, plus c'est risqué !", "Le **MASI** (Moroccan All Shares Index) est l'indice principal de la Bourse de Casablanca. Il regroupe la performance de toutes les actions cotées au Maroc.", "Une **action** est simplement une petite part de propriété d'une entreprise. Si tu achètes une action Attijariwafa Bank, tu possèdes une minuscule fraction de la banque !"]
THANKS_RESPONSES = ["Avec grand plaisir ! N'hésite pas si tu veux tester d'autres pourcentages.", "Je t'en prie ! On est là pour battre le marché ensemble. 💪", "C'est normal, c'est mon travail ! Dis-moi si tu veux relancer une simulation."]

def _is_range_request(user_input: str) -> bool:
    normalized = _normalize_text(user_input)
    return "plage" in normalized or "ranges" in normalized or "range" in normalized or "valeurs" in normalized or "valeur" in normalized

def _simulate_random_portfolios(annual_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 5000) -> tuple[np.ndarray, list[np.ndarray]]:
    num_assets = len(annual_returns)
    results = np.zeros((3, num_portfolios))
    weights_record: list[np.ndarray] = []
    risk_free_rate = 0.03
    for i in range(num_portfolios):
        weights = np.zeros(num_assets)
        max_assets_in_pf = min(15, num_assets)
        num_selected_assets = np.random.randint(2, max_assets_in_pf + 1)
        selected_indices = np.random.choice(num_assets, num_selected_assets, replace=False)
        random_w = np.random.random(num_selected_assets)
        weights[selected_indices] = random_w / np.sum(random_w)
        weights_record.append(weights)
        portfolio_return = float(np.sum(weights * annual_returns))
        portfolio_std_dev = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        results[0, i] = portfolio_return * 100
        results[1, i] = portfolio_std_dev * 100
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev > 0 else -np.inf
    return results, weights_record

def optimiser_portefeuille_personnalise(rendement_cible_pct: float, risque_max_pct: float, capital: float | None = None) -> str:
    try:
        stock_glob = Path(os.getenv("CSE_STOCK_GLOB", str(DEFAULT_STOCK_GLOB)))
        info_csv = Path(os.getenv("CSE_INFO_CSV", str(DEFAULT_INFO_CSV)))
        company_labels = _load_company_labels(info_csv)
        prices = _load_price_frame(stock_glob)
        if prices.empty:
            return "Erreur technique : aucun historique de prix exploitable n'a été trouvé dans tes fichiers."
        latest_prices = prices.iloc[-1]
    except Exception as e:
        return f"Erreur lors du chargement des données : {str(e)}"
    returns = prices.pct_change().dropna(how="all")
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_portfolios = 5000
    results, weights_record = _simulate_random_portfolios(annual_returns, cov_matrix, num_portfolios)
    valid_indices = np.where((results[0] >= rendement_cible_pct) & (results[1] <= risque_max_pct))[0]
    is_plan_b = False
    if len(valid_indices) == 0:
        safe_indices = np.where(results[1] <= risque_max_pct)[0]
        if len(safe_indices) == 0:
            return (f"⚠️ Même avec un risque de {risque_max_pct:.2f}%, je n'ai trouvé aucune combinaison viable sur ce marché. "
                    f"Tu dois augmenter ton seuil de risque autorisé.")
        best_idx = int(safe_indices[np.argmax(results[0, safe_indices])])
        is_plan_b = True
    else:
        best_idx = int(valid_indices[np.argmax(results[2, valid_indices])])
    try:
        optimal_weights = weights_record[best_idx]
        allocations = pd.DataFrame({"Ticker": returns.columns, "Poids (%)": optimal_weights * 100})
        allocations = allocations[allocations["Poids (%)"] > 1.0].reset_index(drop=True)
        allocations = allocations.sort_values(by="Poids (%)", ascending=False).reset_index(drop=True)
        # Ensure indices are fresh by copying and resetting again to prevent Streamlit caching issues
        allocations = allocations.copy().reset_index(drop=True)
        allocations["Entreprise"] = allocations["Ticker"].map(lambda ticker: company_labels.get(ticker, ticker))
        table_markdown = _format_allocations_markdown(allocations, latest_prices=latest_prices, capital=capital)
    except Exception as e:
        import traceback
        return f"Erreur lors de la création du portefeuille : {str(e)}\n{traceback.format_exc()}"
        if is_plan_b:
            intro_text = (
                f"⚠️ **Compromis nécessaire !**\n\n"
                f"Il m'a été mathématiquement impossible d'atteindre tes {rendement_cible_pct:.2f}% de rendement sans dépasser ta limite de {risque_max_pct:.2f}% de risque.\n"
                f"J'ai donc activé mon **Plan B** : Voici le portefeuille qui t'offre le rendement **le plus élevé possible** tout en restant strictement sous ta limite de risque.\n\n"
            )
        else:
            intro_text = "✅ **Simulation réussie ! J'ai trouvé un portefeuille qui respecte toutes tes contraintes.**\n\n"
        return (
            f"{intro_text}"
            f"📈 **Projections de l'algorithme :**\n"
            f"- Rendement annuel estimé : **{results[0, best_idx]:.2f}%**\n"
            f"- Risque (Volatilité) maximal : **{results[1, best_idx]:.2f}%**\n\n"
            f"🎯 **Voici la répartition recommandée :**\n\n"
            f"{table_markdown}\n"
        )
    except Exception as e:
        import traceback
        return f"Erreur finale : {str(e)}\n{traceback.format_exc()}"

def obtenir_plages_possibles() -> str:
    stock_glob = Path(os.getenv("CSE_STOCK_GLOB", str(DEFAULT_STOCK_GLOB)))
    prices = _load_price_frame(stock_glob)
    if prices.empty:
        return "Je ne trouve pas tes données locales CSV pour faire l'analyse."
    returns = prices.pct_change().dropna(how="all")
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    results, _ = _simulate_random_portfolios(annual_returns, cov_matrix)
    market_return_min = float(np.nanmin(annual_returns) * 100)
    market_return_max = float(np.nanmax(annual_returns) * 100)
    market_risk_min = float(np.min(results[1]))
    best_sharpe_idx = int(np.argmax(results[2]))
    return (
        f"📊 **Analyse globale de la Bourse de Casablanca :**\n\n"
        f"D'après les historiques de prix que tu m'as fournis :\n"
        f"📉 Le titre le moins performant affiche **{market_return_min:.2f}%** par an.\n"
        f"🚀 Le titre le plus performant frôle les **{market_return_max:.2f}%** par an.\n"
        f"🛡️ Le risque (volatilité) le plus faible atteignable en diversifiant est de **{market_risk_min:.2f}%**.\n\n"
        f"⭐ **Le point d'équilibre optimal (Ratio de Sharpe Max) :**\n"
        f"Sans contrainte, le meilleur compromis naturel du marché offre **{float(results[0, best_sharpe_idx]):.2f}%** de rendement pour **{float(results[1, best_sharpe_idx]):.2f}%** de risque.\n\n"
        f"Dis-moi ce que tu en penses et donne-moi tes propres objectifs pour qu'on commence !"
    )

def run_demo(user_input: str, model: str | None = None, prev_assistant: str | None = None) -> str:
    if _is_range_request(user_input):
        return obtenir_plages_possibles()
    extracted = _extract_parameters(user_input)
    if extracted:
        rendement, risque, capital = extracted
        if rendement is not None and risque is not None:
            return optimiser_portefeuille_personnalise(rendement, risque, capital)
    return _fetch_ai_response(_build_ai_prompt(user_input))

def local_chat(user_input: str, capital: float | None = None, prev_assistant: str | None = None) -> str:
    if _is_range_request(user_input):
        return obtenir_plages_possibles()
    rendement, risque, input_capital = _extract_parameters(user_input)
    final_capital = input_capital if input_capital is not None else capital
    if rendement is not None and risque is not None:
        return optimiser_portefeuille_personnalise(rendement, risque, final_capital)
    if _fuzzy_contains_any(user_input, ["bonjour", "bjour", "salut", "slt", "hello", "coucou", "cc", "salam"]):
        return random.choice(GREETING_RESPONSES)
    if _fuzzy_contains_any(user_input, ["merci", "mrc", "mercie", "choukrane", "top", "super", "génial", "genial", "thanks", "thx"]):
        return random.choice(THANKS_RESPONSES)
    if _fuzzy_contains_any(user_input, ["blague", "blag", "rigoler", "humour", "joke", "jok", "rire"]):
        return random.choice(JOKE_RESPONSES)
    if _fuzzy_contains_any(user_input, ["nul", "stupid", "con", "idiot", "bet", "bête", "merd", "chier", "sert a rien", "marche pa", "bug"]):
        return random.choice(FRUSTRATION_RESPONSES)
    if _fuzzy_contains_any(user_input, ["qui es", "createur", "fabrique", "robot", "ia", "intelligence", "nom", "t appel", "t ki", "kies tu"]):
        return random.choice(IDENTITY_RESPONSES)
    if _fuzzy_contains_any(user_input, ["action", "masi", "bourse", "dividende", "risque", "rendement", "definition", "c est quoi", "c koi"]):
        return random.choice(FINANCE_DEF_RESPONSES)
    if _fuzzy_contains_any(user_input, ["comment", "methode", "formule", "markowitz", "algorithme", "expliquer"]):
        return random.choice(EXPLAIN_RESPONSES)
    if user_input.strip().endswith("?") or _fuzzy_contains_any(user_input, ["aide", "que faire"]):
        return random.choice(CLARIFY_RESPONSES)
    return random.choice(GENERAL_RESPONSES)