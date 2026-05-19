# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3
import os
from dotenv import load_dotenv
from groq import Groq


def load_api_key() -> str:
    """Charge la clé API Groq depuis le fichier .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("ERREUR : GROQ_API_KEY non trouvee dans .env")
    return api_key


def create_client(api_key: str) -> Groq:
    """Crée un client Groq avec la clé fournie."""
    return Groq(api_key=api_key)


def build_messages() -> list[dict[str, str]]:
    """Construit la conversation à envoyer au modèle."""
    return [
        {
            "role": "system",
            "content": (
                "Tu es un assistant medical senegalais. "
                "Reponds en francais simple. "
                "Maximum 3 phrases."
            ),
        },
        {
            "role": "user",
            "content": "Quels sont les symptomes du paludisme ?",
        },
    ]


def ask_model(client: Groq, messages: list[dict[str, str]]) -> tuple[str, int]:
    """Envoie la requête à l'API Groq et retourne la réponse texte et le nombre de tokens."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content, response.usage.total_tokens


def main() -> None:
    try:
        api_key = load_api_key()
        client = create_client(api_key)
        messages = build_messages()
        answer, tokens = ask_model(client, messages)

        print("=== Reponse de Llama 3 ===")
        print(answer)
        print(f"\nTokens utilises : {tokens}")

    except EnvironmentError as exc:
        print(exc)
    except Exception as exc:
        print("ERREUR : impossible d'appeler l'API Groq.")
        print(exc)


def build_sensante_messages() -> list[dict[str, str]]:
    """Construit un prompt adapté au format SenSante."""
    return [
        {
            "role": "system",
            "content": (
                "Tu es un assistant medical senegalais. "
                "Tu recois un diagnostic et des donnees patient. "
                "Explique le resultat en francais simple, "
                "comme un medecin parlerait a son patient. "
                "Sois rassurant mais recommande une consultation. "
                "Maximum 3 phrases. "
                "Ne fais JAMAIS de diagnostic toi-meme."
            ),
        },
        {
            "role": "user",
            "content": (
                "Patient : Femme, 28 ans, region Dakar\n"
                "Symptomes : temperature 39.5, toux, fatigue, maux de tete\n"
                "Diagnostic du modele : paludisme (probabilite 72%)\n"
                "Explique ce resultat au patient."
            ),
        },
    ]


def main() -> None:
    try:
        api_key = load_api_key()
        client = create_client(api_key)
        messages = build_messages()
        answer, tokens = ask_model(client, messages)

        print("=== Reponse de Llama 3 ===")
        print(answer)
        print(f"\nTokens utilises : {tokens}")

        sensante_messages = build_sensante_messages()
        explanation, _ = ask_model(client, sensante_messages)
        print("\n=== Explication SenSante ===")
        print(explanation)

    except EnvironmentError as exc:
        print(exc)
    except Exception as exc:
        print("ERREUR : impossible d'appeler l'API Groq.")
        print(exc)


if __name__ == "__main__":
    main()