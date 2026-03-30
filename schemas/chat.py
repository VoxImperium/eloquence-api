# schemas/chat.py — Modèles Pydantic pour le chat juridique

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Source(BaseModel):
    type: Literal["jurisprudence", "loi"]
    titre: str
    contenu: str
    url: str
