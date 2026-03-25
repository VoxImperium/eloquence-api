from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx, os

router = APIRouter(prefix="/emails", tags=["emails"])

BREVO_KEY      = os.getenv("BREVO_API_KEY", "")
TEMPLATE_WELCOME = int(os.getenv("BREVO_TEMPLATE_WELCOME", "1"))

class WelcomeEmailRequest(BaseModel):
    email: str
    prenom: str = "cher utilisateur"
    plan: str = "free"

@router.post("/welcome")
async def send_welcome_email(req: WelcomeEmailRequest):
    """Envoie l'email de bienvenue via Brevo"""
    if not BREVO_KEY:
        raise HTTPException(500, "Brevo API key manquante")
    
    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": BREVO_KEY, "Content-Type": "application/json"},
                json={
                    "to": [{"email": req.email, "name": req.prenom}],
                    "templateId": TEMPLATE_WELCOME,
                    "params": {
                        "prenom": req.prenom.capitalize(),
                        "plan":   req.plan,
                        "email":  req.email,
                    },
                    "replyTo": {"email": "contact@eloquence.fr", "name": "Éloquence"},
                },
                timeout=10,
            )
            if r.status_code == 201:
                return {"success": True, "messageId": r.json().get("messageId")}
            else:
                raise HTTPException(r.status_code, f"Brevo error: {r.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/newsletter/subscribe")
async def subscribe_newsletter(email: str, prenom: str = ""):
    """Ajoute un contact à la liste Brevo"""
    if not BREVO_KEY:
        return {"success": False}
    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                "https://api.brevo.com/v3/contacts",
                headers={"api-key": BREVO_KEY, "Content-Type": "application/json"},
                json={
                    "email": email,
                    "attributes": {"PRENOM": prenom},
                    "listIds": [2],
                    "updateEnabled": True,
                },
                timeout=10,
            )
        return {"success": True}
    except:
        return {"success": False}
