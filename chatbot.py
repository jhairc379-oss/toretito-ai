import os
from google import genai

# Lee la API key desde la variable de entorno
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("No se encontró la API key. Asegúrate de haber hecho 'set GEMINI_API_KEY=...' en esta terminal.")
    exit()

client = genai.Client(api_key=api_key)

# Guardamos el historial de la conversación
historial = []

print("Chatbot listo. Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        print("¡Chao!")
        break

    historial.append(f"Usuario: {pregunta}")

    respuesta = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="\n".join(historial)
    )

    texto_respuesta = respuesta.text
    print(f"Bot: {texto_respuesta}\n")

    historial.append(f"Bot: {texto_respuesta}")