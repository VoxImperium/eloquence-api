# api/routers/payments.py
# Gestion des abonnements Stripe

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
import stripe
import os

router = APIRouter(prefix="/payments", tags=["payments"])

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
PRICE_ID = os.getenv("STRIPE_PRICE_ID")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


class CheckoutRequest(BaseModel):
    user_id:    str
    user_email: str
    success_url: str = "http://localhost:3000/dashboard?success=true"
    cancel_url:  str = "http://localhost:3000/pricing"


class PortalRequest(BaseModel):
    customer_id:  str
    return_url:   str = "http://localhost:3000/dashboard"


@router.post("/create-checkout")
async def create_checkout(req: CheckoutRequest):
    """Crée une session Stripe Checkout pour l'abonnement Pro."""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=req.user_email,
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            metadata={"user_id": req.user_id},
            subscription_data={"metadata": {"user_id": req.user_id}},
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except Exception as e:
        raise HTTPException(500, f"Erreur Stripe: {str(e)}")


@router.post("/portal")
async def customer_portal(req: PortalRequest):
    """Crée un lien vers le portail client Stripe."""
    try:
        session = stripe.billing_portal.Session.create(
            customer=req.customer_id,
            return_url=req.return_url,
        )
        return {"portal_url": session.url}
    except Exception as e:
        raise HTTPException(500, f"Erreur portail: {str(e)}")


@router.post("/webhook")
async def stripe_webhook(request):
    """Reçoit les événements Stripe et met à jour les plans."""
    from fastapi import Request
    payload = await request.body()

    try:
        event = stripe.Event.construct_from(
            stripe.util.convert_to_stripe_object(
                stripe.util.convert_to_dict(payload)
            ), stripe.api_key
        )
    except Exception:
        return {"status": "ignored"}

    if event.type == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        if user_id:
            supabase.table("profiles").update({"plan": "pro"}).eq("id", user_id).execute()

    elif event.type in ["customer.subscription.deleted", "customer.subscription.paused"]:
        sub = event.data.object
        user_id = sub.metadata.get("user_id")
        if user_id:
            supabase.table("profiles").update({"plan": "free"}).eq("id", user_id).execute()

    return {"status": "ok"}
