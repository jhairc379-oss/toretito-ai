import os
import base64
import secrets
from flask import Flask, render_template, request, jsonify, session
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("No se encontró GEMINI_API_KEY. Revisa tu archivo .env.")

client = genai.Client(api_key=api_key)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

INSTRUCCION_SISTEMA = """
Eres un asistente amigable y entusiasta llamado 'Toretito'.
Respondes de forma breve y clara, usas emojis ocasionalmente,
y siempre tratas de ser motivador con el usuario.
Si no sabes algo, lo admites honestamente en vez de inventar.
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.get_json()
    pregunta = data.get("mensaje", "")
    imagen_b64 = data.get("imagen")
    mime_type = data.get("mime_type")

    # Cada usuario tiene su propio historial guardado en su sesión
    if "historial" not in session:
        session["historial"] = []

    historial = session["historial"]

    partes_contenido = []

    if imagen_b64:
        imagen_bytes = base64.b64decode(imagen_b64)
        partes_contenido.append(
            types.Part.from_bytes(data=imagen_bytes, mime_type=mime_type)
        )

    texto_para_historial = pregunta if pregunta else "(el usuario envió una imagen)"
    historial.append(f"Usuario: {texto_para_historial}")

    partes_contenido.append("\n".join(historial))

    respuesta = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=partes_contenido,
        config={"system_instruction": INSTRUCCION_SISTEMA}
    )

    texto_respuesta = respuesta.text
    historial.append(f"Bot: {texto_respuesta}")

    # Guardamos el historial actualizado de vuelta en la sesión
    session["historial"] = historial

    return jsonify({"respuesta": texto_respuesta})

if __name__ == "__main__":
    app.run(debug=True)
    