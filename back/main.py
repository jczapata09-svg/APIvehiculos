import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Vehicle Specs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

MARCAS_VALIDAS = {
    "toyota", "honda", "ford", "chevrolet", "nissan", "volkswagen", "bmw",
    "mercedes", "mercedes-benz", "audi", "hyundai", "kia", "mazda", "subaru",
    "mitsubishi", "suzuki", "renault", "peugeot", "fiat", "jeep", "dodge",
    "ram", "gmc", "buick", "cadillac", "lincoln", "chrysler", "volvo",
    "jaguar", "land rover", "lexus", "infiniti", "acura", "porsche", "ferrari",
    "lamborghini", "maserati", "alfa romeo", "seat", "skoda", "opel", "citroen",
    "tesla", "rivian", "lucid", "genesis", "haval", "chery", "geely", "byd",
    "mg", "ssangyong", "isuzu", "daihatsu", "hino", "kenworth", "freightliner",
    "international", "mack", "peterbilt", "scania", "volvo trucks", "man",
    "yamaha", "kawasaki", "ducati", "harley-davidson", "harley", "ktm",
    "triumph", "royal enfield", "bajaj", "hero", "tvs", "aprilia", "husqvarna",
    "can-am", "indian", "norton", "benelli", "cfmoto", "zontes",
    "mercedes benz", "land-rover",
}

PALABRAS_BLOQUEADAS = {
    "arma", "bomba", "droga", "explosivo", "hackear", "virus", "malware",
    "persona", "político", "presidente", "celebridad", "actor", "actriz",
    "receta", "medicamento", "hack", "crack", "exploit",
}

ALLOWED_PATTERN = re.compile(r"^[a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ\s\-\.]+$")
MAX_FIELD_LENGTH = 60


class VehicleRequest(BaseModel):
    marca: str
    referencia: str

    @field_validator("marca", "referencia")
    @classmethod
    def validar_campo(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El campo no puede estar vacío.")
        if len(v) > MAX_FIELD_LENGTH:
            raise ValueError(f"Máximo {MAX_FIELD_LENGTH} caracteres.")
        if not ALLOWED_PATTERN.match(v):
            raise ValueError("Solo se permiten letras, números, espacios y guiones.")
        if any(p in v.lower() for p in PALABRAS_BLOQUEADAS):
            raise ValueError("Contenido no permitido.")
        return v

    @field_validator("marca")
    @classmethod
    def validar_marca(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if not any(v_lower.startswith(m) or m.startswith(v_lower) or v_lower == m
                   for m in MARCAS_VALIDAS):
            raise ValueError(
                "Marca no reconocida. Solo se aceptan marcas de vehículos (autos, motos, camiones)."
            )
        return v.strip()


SYSTEM_PROMPT = """Eres un asistente especializado EXCLUSIVAMENTE en especificaciones técnicas de vehículos automotores.

REGLAS:
- Solo responde sobre vehículos automotores reales.
- Si el input no corresponde a un vehículo real, devuelve encontrado: false.
- No incluyas precios ni información comercial.
- si esta mal escrito dame opciones para responder


Recibirás fragmentos de búsqueda web. Responde SOLO con JSON válido sin backticks:
{
  "encontrado": true,
  "modelo": "nombre completo del vehículo",
  "specs": [
    { "categoria": "Motor", "items": [{"label": "...", "valor": "..."}, ...] },
    { "categoria": "Transmisión", "items": [...] },
    { "categoria": "Dimensiones", "items": [...] },
    { "categoria": "Rendimiento", "items": [...] },
    { "categoria": "Frenos y suspensión", "items": [...] }
  ]
}
Si no hay info suficiente: {"encontrado": false, "mensaje": "..."}
Campos sin dato: "N/D"."""


@app.post("/specs")
async def get_specs(req: VehicleRequest):
    if not SERP_API_KEY or not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Faltan variables de entorno.")

    query = f"{req.marca} {req.referencia} especificaciones técnicas ficha técnica"

    async with httpx.AsyncClient(timeout=15) as http:
        serp_res = await http.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "hl": "es",
                "gl": "co",
                "num": 5,
                "api_key": SERP_API_KEY,
            },
        )

    if serp_res.status_code != 200:
        raise HTTPException(status_code=502, detail=f"SerpAPI error: {serp_res.status_code}")

    serp_data = serp_res.json()
    organic = serp_data.get("organic_results", [])
    snippets = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}"
        for i, r in enumerate(organic[:5])
    )
    kg = serp_data.get("knowledge_graph", {})
    knowledge = f"Knowledge Graph: {str(kg)[:800]}" if kg else ""

    if not snippets and not knowledge:
        raise HTTPException(status_code=404, detail="Sin resultados de búsqueda.")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Vehículo: {req.marca} {req.referencia}\n\nResultados:\n{knowledge}\n\n{snippets}"}
        ],
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al parsear respuesta de OpenAI.")

    sources = [
        {"title": r.get("title", ""), "link": r.get("link", "")}
        for r in organic[:3]
    ]

    return {"data": parsed, "sources": sources}


@app.get("/health")
def health():
    return {"status": "ok"}