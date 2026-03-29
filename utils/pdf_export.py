"""
utils/pdf_export.py — Génération de PDF professionnel avec le style Eloquence AI.

Utilise ReportLab pour produire un document PDF élégant (dark navy / gold)
incluant : qualification juridique, essence du drame, stratégie, plaidoirie,
fondements juridiques et jurisprudences.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette Eloquence AI ──────────────────────────────────────────────────────
_NAVY = colors.HexColor("#0A0F1E")          # fond dark navy
_GOLD = colors.HexColor("#D4AF37")          # accent gold
_GOLD_LIGHT = colors.HexColor("#F0D060")    # gold clair pour texte sur fond sombre
_WHITE = colors.HexColor("#FFFFFF")
_LIGHT_GREY = colors.HexColor("#E8E8E8")    # fond de cellule clair
_DARK_GREY = colors.HexColor("#2A2F3E")     # fond d'en-tête de section

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2 * cm


# ── Styles ────────────────────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    return {
        "doc_title": ParagraphStyle(
            "doc_title",
            fontName="Times-Bold",
            fontSize=22,
            textColor=_GOLD,
            alignment=TA_CENTER,
            spaceAfter=4,
            leading=28,
        ),
        "doc_subtitle": ParagraphStyle(
            "doc_subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=_LIGHT_GREY,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontName="Times-Bold",
            fontSize=13,
            textColor=_GOLD,
            alignment=TA_LEFT,
            spaceBefore=14,
            spaceAfter=6,
            leading=18,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=_NAVY,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=15,
        ),
        "article_label": ParagraphStyle(
            "article_label",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=_DARK_GREY,
            spaceAfter=2,
        ),
        "article_text": ParagraphStyle(
            "article_text",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=_NAVY,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            spaceAfter=8,
            leading=14,
        ),
        "jurisprudence_ref": ParagraphStyle(
            "jurisprudence_ref",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=_DARK_GREY,
            spaceAfter=2,
        ),
        "jurisprudence_resume": ParagraphStyle(
            "jurisprudence_resume",
            fontName="Helvetica",
            fontSize=9,
            textColor=_NAVY,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            spaceAfter=8,
            leading=14,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=_LIGHT_GREY,
        ),
    }


# ── Header / Footer callbacks ─────────────────────────────────────────────────

def _make_header_footer(gen_date: str, total_pages_ref: list[int]):
    """
    Retourne les callbacks onPage appelés par ReportLab pour chaque page.
    total_pages_ref[0] sera mis à jour après la construction (multi-pass).
    """

    def _on_page(canvas, doc):
        canvas.saveState()

        # ── Bandeau d'en-tête ──────────────────────────────────────────────
        canvas.setFillColor(_NAVY)
        canvas.rect(0, PAGE_HEIGHT - 1.5 * cm, PAGE_WIDTH, 1.5 * cm, fill=True, stroke=False)

        canvas.setFont("Times-Bold", 12)
        canvas.setFillColor(_GOLD)
        canvas.drawString(MARGIN, PAGE_HEIGHT - cm, "Éloquence AI")

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_LIGHT_GREY)
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - cm, f"Analyse juridique — {gen_date}")

        # Trait gold sous l'en-tête
        canvas.setStrokeColor(_GOLD)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, PAGE_HEIGHT - 1.55 * cm, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 1.55 * cm)

        # ── Bandeau de pied de page ────────────────────────────────────────
        canvas.setFillColor(_NAVY)
        canvas.rect(0, 0, PAGE_WIDTH, 1.2 * cm, fill=True, stroke=False)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_LIGHT_GREY)
        canvas.drawString(MARGIN, 0.4 * cm, "Généré par Éloquence AI")

        page_num = doc.page
        canvas.drawRightString(
            PAGE_WIDTH - MARGIN,
            0.4 * cm,
            f"Page {page_num}",
        )

        canvas.restoreState()

    return _on_page


# ── Section helpers ───────────────────────────────────────────────────────────

def _section(title: str, content: str, styles: dict) -> list:
    """Retourne une liste de Flowables pour une section de texte simple."""
    elems: list[Any] = []
    elems.append(Paragraph(title, styles["section_heading"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_GOLD, spaceAfter=4))
    if content and content.strip():
        elems.append(Paragraph(_convert_markdown_to_html(_escape(content)), styles["body"]))
    else:
        elems.append(Paragraph("<i>Non renseigné</i>", styles["body"]))
    return elems


def _escape(text: str) -> str:
    """Échappe les caractères XML pour ReportLab."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _convert_markdown_to_html(text: str) -> str:
    """Convertit le formatage markdown en balises HTML compatibles ReportLab.

    Doit être appelé APRÈS _escape() afin que les caractères spéciaux XML
    soient déjà échappés avant l'injection des balises HTML.
    """
    # **texte** → <b>texte</b> (limité à une seule ligne pour éviter les balises mal formées)
    text = re.sub(r'\*\*([^\n]+?)\*\*', r'<b>\1</b>', text)
    return text


# ── Main builder ──────────────────────────────────────────────────────────────

def generate_analyse_pdf(
    qualification: str,
    essence_du_drame: str,
    tirade_oratoire: str,
    strategie: str,
    articles_cites: list[dict],
    jurisprudence: list[dict],
) -> bytes:
    """
    Génère un PDF professionnel avec le style Eloquence AI.

    Args:
        qualification:    Qualification juridique.
        essence_du_drame: Essence / contexte du drame.
        tirade_oratoire:  Plaidoirie (tirade oratoire).
        strategie:        Stratégie juridique.
        articles_cites:   Liste de dicts {"code", "numero", "texte"}.
        jurisprudence:    Liste de dicts {"formatage_officiel", "resume"}.

    Returns:
        Contenu binaire du PDF.
    """
    buf = io.BytesIO()
    gen_date = datetime.now().strftime("%d/%m/%Y")
    styles = _build_styles()
    total_pages_ref: list[int] = [0]

    # ── Document ──────────────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm,
        title="Analyse Juridique — Éloquence AI",
        author="Éloquence AI",
    )

    on_page = _make_header_footer(gen_date, total_pages_ref)

    frame = Frame(
        MARGIN,
        1.8 * cm,
        PAGE_WIDTH - 2 * MARGIN,
        PAGE_HEIGHT - 2.2 * cm - 1.8 * cm,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])

    # ── Flowables ─────────────────────────────────────────────────────────────
    story: list[Any] = []

    # Titre principal (sur fond blanc, couleur navy)
    title_data = [
        [Paragraph("ANALYSE JURIDIQUE", styles["doc_title"])],
        [Paragraph("Éloquence AI — Document confidentiel", styles["doc_subtitle"])],
        [Paragraph(f"Généré le {gen_date}", styles["doc_subtitle"])],
    ]
    title_table = Table(title_data, colWidths=[PAGE_WIDTH - 2 * MARGIN])
    title_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_NAVY]),
            ("BOX", (0, 0), (-1, -1), 2, _GOLD),
        ])
    )
    story.append(title_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── 1. Qualification juridique ────────────────────────────────────────────
    story.extend(_section("1. Qualification juridique", qualification, styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── 2. Essence du drame ───────────────────────────────────────────────────
    story.extend(_section("2. Essence du drame", essence_du_drame, styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── 3. Stratégie juridique ────────────────────────────────────────────────
    story.extend(_section("3. Stratégie juridique", strategie, styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── 4. Plaidoirie ─────────────────────────────────────────────────────────
    story.extend(_section("4. Plaidoirie (tirade oratoire)", tirade_oratoire, styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── 5. Fondements juridiques ──────────────────────────────────────────────
    story.append(Paragraph("5. Fondements juridiques", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GOLD, spaceAfter=4))

    if articles_cites:
        for i, art in enumerate(articles_cites, 1):
            code = _escape(str(art.get("code", "")))
            numero = _escape(str(art.get("numero", "")))
            texte = _escape(str(art.get("texte", "")))
            label = f"{i}. Article {numero} — {code}" if numero else f"{i}. {code}"
            story.append(Paragraph(label, styles["article_label"]))
            if texte:
                story.append(Paragraph(f"« {texte} »", styles["article_text"]))
    else:
        story.append(Paragraph("<i>Aucun article cité</i>", styles["body"]))

    story.append(Spacer(1, 0.3 * cm))

    # ── 6. Jurisprudences ─────────────────────────────────────────────────────
    story.append(Paragraph("6. Jurisprudences", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GOLD, spaceAfter=4))

    if jurisprudence:
        for i, j in enumerate(jurisprudence, 1):
            ref_raw = j.get("formatage_officiel") or j.get("numero") or f"Jurisprudence {i}"
            ref = _escape(str(ref_raw))
            resume = _escape(str(j.get("resume", "")))
            story.append(Paragraph(f"{i}. {ref}", styles["jurisprudence_ref"]))
            if resume:
                story.append(Paragraph(resume, styles["jurisprudence_resume"]))
    else:
        story.append(Paragraph("<i>Aucune jurisprudence citée</i>", styles["body"]))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    return buf.getvalue()
