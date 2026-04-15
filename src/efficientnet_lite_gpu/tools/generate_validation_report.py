#!/usr/bin/env python3
"""
PDF Validation Report Generator — MobileNetV2-0.35 Food Classification Model.

Generates a professional PDF report with:
  - Model architecture summary
  - Training results (2-stage)
  - Per-class metrics table
  - Confusion matrix image
  - Per-class F1 chart image
  - Export formats summary
  - Data integrity / leakage verification

Usage:
    cd src/efficientnet_lite_gpu
    python -m tools.generate_validation_report
"""

import datetime
import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    HRFlowable,
    KeepTogether,
)


# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE = HexColor("#0f3460")
MID_BLUE = HexColor("#16213e")
ACCENT_BLUE = HexColor("#4C72B0")
LIGHT_BG = HexColor("#f0f4fa")
WHITE = colors.white
BLACK = colors.black
LIGHT_GREY = HexColor("#e8e8e8")
GREEN_OK = HexColor("#27ae60")
RED_WARN = HexColor("#e74c3c")

# ── Page dimensions ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm


# ── Paths ─────────────────────────────────────────────────────────────────────
def _root() -> Path:
    """Return the project root (src/efficientnet_lite_gpu)."""
    # When run as  python -m tools.generate_validation_report  the CWD is the root.
    return Path.cwd()


OUTPUT_PDF = _root() / "exports" / "rapport_validation_modeles.pdf"
CONFUSION_MATRIX_PNG = (
    _root() / "train" / "results" / "evaluation_results" / "confusion_matrix.png"
)
CLASS_PERFORMANCE_PNG = (
    _root() / "train" / "results" / "evaluation_results" / "class_performance.png"
)


# ── Styles ────────────────────────────────────────────────────────────────────
def _build_styles():
    ss = getSampleStyleSheet()

    ss.add(
        ParagraphStyle(
            "ReportTitle",
            parent=ss["Title"],
            fontSize=22,
            leading=28,
            textColor=DARK_BLUE,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    ss.add(
        ParagraphStyle(
            "ReportSubtitle",
            parent=ss["Normal"],
            fontSize=11,
            leading=14,
            textColor=HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    ss.add(
        ParagraphStyle(
            "SectionHeading",
            parent=ss["Heading1"],
            fontSize=15,
            leading=20,
            textColor=DARK_BLUE,
            spaceBefore=18,
            spaceAfter=8,
            borderWidth=0,
            borderPadding=0,
        )
    )
    ss.add(
        ParagraphStyle(
            "SubSectionHeading",
            parent=ss["Heading2"],
            fontSize=12,
            leading=16,
            textColor=MID_BLUE,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    ss.add(
        ParagraphStyle(
            "BodyText2",
            parent=ss["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=BLACK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    ss.add(
        ParagraphStyle(
            "SmallNote",
            parent=ss["Normal"],
            fontSize=8,
            leading=10,
            textColor=HexColor("#888888"),
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    ss.add(
        ParagraphStyle(
            "TableHeader",
            parent=ss["Normal"],
            fontSize=9,
            leading=11,
            textColor=WHITE,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
    )
    ss.add(
        ParagraphStyle(
            "TableCell",
            parent=ss["Normal"],
            fontSize=9,
            leading=11,
            textColor=BLACK,
            alignment=TA_CENTER,
        )
    )
    ss.add(
        ParagraphStyle(
            "TableCellLeft",
            parent=ss["Normal"],
            fontSize=9,
            leading=11,
            textColor=BLACK,
            alignment=TA_LEFT,
        )
    )
    ss.add(
        ParagraphStyle(
            "Footer",
            parent=ss["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=HexColor("#aaaaaa"),
            alignment=TA_CENTER,
        )
    )
    return ss


# ── Helper: build a styled table ─────────────────────────────────────────────
def _make_table(headers, rows, col_widths=None):
    """Build a reportlab Table with standard styling."""
    ss = _build_styles()

    header_row = [Paragraph(h, ss["TableHeader"]) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(c), ss["TableCell"]) for c in row])

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, DARK_BLUE),
        # Alternating row colours
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Zebra striping
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_BG))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Helper: horizontal rule ──────────────────────────────────────────────────
def _hr():
    return HRFlowable(
        width="100%", thickness=1, color=LIGHT_GREY, spaceAfter=10, spaceBefore=4
    )


# ── Helper: embed a PNG with bounded size ─────────────────────────────────────
def _embed_image(path: Path, max_width=None, max_height=None):
    """Return a reportlab Image flowable, scaled to fit constraints."""
    if max_width is None:
        max_width = PAGE_W - 2 * MARGIN
    if max_height is None:
        max_height = 14 * cm

    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight

    # Scale to fit within bounds while preserving aspect ratio.
    ratio = min(max_width / iw, max_height / ih, 1.0)
    img.drawWidth = iw * ratio
    img.drawHeight = ih * ratio
    img.hAlign = "CENTER"
    return img


# ── Report content builders ──────────────────────────────────────────────────
def _section_title_page(ss):
    """Cover / title elements."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    elements = [
        Spacer(1, 2.5 * cm),
        Paragraph(
            "Rapport de Validation de Modele",
            ss["ReportTitle"],
        ),
        Paragraph(
            "MobileNetV2 (alpha=0.35) — Classification alimentaire 9 classes",
            ss["ReportSubtitle"],
        ),
        Paragraph(
            f"Genere le {now}",
            ss["SmallNote"],
        ),
        _hr(),
        Spacer(1, 0.5 * cm),
    ]
    return elements


def _section_model_info(ss):
    """Section 1 — Model architecture."""
    elements = [
        Paragraph("1. Informations sur le modele", ss["SectionHeading"]),
        _hr(),
    ]

    rows = [
        ["Backbone", "MobileNetV2 (alpha=0.35, ImageNet)"],
        ["Entree", "224 x 224 x 3 (RGB)"],
        ["Nombre de classes", "9 (8 alimentaires + 'Other')"],
        [
            "Classes",
            "Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Other, Pizza, Sandwich",
        ],
        ["Parametres totaux", "~426 K (1.63 MB)"],
        ["Strategie d'entrainement", "2 etapes (backbone gele + fine-tuning)"],
    ]

    headers = ["Propriete", "Valeur"]
    col_widths = [5.5 * cm, None]

    tbl = _make_table(headers, rows, col_widths)
    # Override first-column alignment to left
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_training_results(ss):
    """Section 2 — Training results (2-stage)."""
    elements = [
        Paragraph("2. Resultats de l'entrainement", ss["SectionHeading"]),
        _hr(),
    ]

    # Stage summary
    elements.append(Paragraph("2.1 Resume par etape", ss["SubSectionHeading"]))

    headers = ["Etape", "Epochs (EarlyStopping)", "Meilleure val_accuracy"]
    rows = [
        ["Stage 1 — Backbone gele", "9", "84.15 %"],
        ["Stage 2 — Fine-tuning", "6", "84.82 %"],
    ]
    tbl = _make_table(headers, rows)
    elements.append(tbl)
    elements.append(Spacer(1, 0.4 * cm))

    # Global metrics
    elements.append(Paragraph("2.2 Metriques globales sur le jeu de test", ss["SubSectionHeading"]))

    headers = ["Metrique", "Valeur"]
    rows = [
        ["Test Accuracy", "84.95 %"],
        ["Macro F1-score", "83.69 %"],
        ["Weighted F1-score", "85.08 %"],
    ]
    tbl = _make_table(headers, rows, col_widths=[7 * cm, 5 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_per_class(ss):
    """Section 3 — Per-class test results."""
    elements = [
        Paragraph("3. Performances par classe (jeu de test)", ss["SectionHeading"]),
        _hr(),
    ]

    headers = ["Classe", "Precision", "Recall", "F1-score", "Support"]
    rows = [
        ["Baked Potato", "0.89", "0.79", "0.83", "98"],
        ["Burger", "0.91", "0.72", "0.80", "94"],
        ["Crispy Chicken", "0.83", "0.87", "0.85", "95"],
        ["Donut", "0.79", "0.91", "0.85", "142"],
        ["Fries", "0.95", "0.78", "0.86", "96"],
        ["Hot Dog", "0.93", "0.79", "0.85", "94"],
        ["Other", "0.90", "0.97", "0.94", "150"],
        ["Pizza", "0.88", "0.83", "0.85", "70"],
        ["Sandwich", "0.57", "0.91", "0.70", "45"],
    ]

    col_widths = [4.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm]
    tbl = _make_table(headers, rows, col_widths)
    # Left-align class names, highlight the lowest F1
    extra_style = [
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]
    # Highlight Sandwich row (F1 = 0.70) in light red
    sandwich_row_idx = 9  # 0-indexed header + 9 rows → row 9
    extra_style.append(("BACKGROUND", (0, sandwich_row_idx), (-1, sandwich_row_idx), HexColor("#fdecea")))
    # Highlight Other row (F1 = 0.94) in light green
    other_row_idx = 7  # "Other" is 7th data row → row 7
    extra_style.append(("BACKGROUND", (0, other_row_idx), (-1, other_row_idx), HexColor("#eafaf1")))

    tbl.setStyle(TableStyle(extra_style))
    elements.append(tbl)

    elements.append(Spacer(1, 0.3 * cm))
    elements.append(
        Paragraph(
            "<i>Note : La classe Sandwich (F1=0.70) presente les performances les plus faibles — "
            "probablement en raison de la faible taille de l'echantillon (support=45) et de confusions "
            "avec Burger / Hot Dog. La classe Other (F1=0.94) est la mieux classee.</i>",
            ss["BodyText2"],
        )
    )
    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_confusion_matrix(ss):
    """Section 4 — Confusion matrix image."""
    elements = [
        Paragraph("4. Matrice de confusion", ss["SectionHeading"]),
        _hr(),
    ]

    if CONFUSION_MATRIX_PNG.exists():
        elements.append(_embed_image(CONFUSION_MATRIX_PNG, max_height=13 * cm))
        elements.append(
            Paragraph(
                f"<i>Source : {CONFUSION_MATRIX_PNG.relative_to(_root())}</i>",
                ss["SmallNote"],
            )
        )
    else:
        elements.append(
            Paragraph(
                f"<b>[Image non trouvee]</b> {CONFUSION_MATRIX_PNG}",
                ss["BodyText2"],
            )
        )

    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_class_performance_chart(ss):
    """Section 5 — Per-class F1 chart."""
    elements = [
        Paragraph("5. Graphique des performances par classe", ss["SectionHeading"]),
        _hr(),
    ]

    if CLASS_PERFORMANCE_PNG.exists():
        elements.append(_embed_image(CLASS_PERFORMANCE_PNG, max_height=11 * cm))
        elements.append(
            Paragraph(
                f"<i>Source : {CLASS_PERFORMANCE_PNG.relative_to(_root())}</i>",
                ss["SmallNote"],
            )
        )
    else:
        elements.append(
            Paragraph(
                f"<b>[Image non trouvee]</b> {CLASS_PERFORMANCE_PNG}",
                ss["BodyText2"],
            )
        )

    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_export_formats(ss):
    """Section 6 — Export formats and cross-format validation."""
    elements = [
        Paragraph("6. Formats d'export et validation croisee", ss["SectionHeading"]),
        _hr(),
    ]

    # 6.1 — File listing
    elements.append(Paragraph("6.1 Fichiers exportes", ss["SubSectionHeading"]))

    headers = ["Format", "Fichier(s)", "Taille"]
    rows = [
        ["Keras (.keras)", "BestModelEfficientNetLite.keras", "5.3 MB"],
        ["TensorFlow.js", "model.json + group1-shard1of1.bin", "1.5 MB"],
        ["TFLite (dynamic range)", "model.tflite", "0.5 MB"],
        ["TFLite (float16)", "model_float16.tflite", "0.8 MB"],
    ]

    col_widths = [3.8 * cm, 5.5 * cm, None]
    tbl = _make_table(headers, rows, col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("ALIGN", (1, 1), (1, -1), "LEFT"),
                ("ALIGN", (2, 1), (2, -1), "LEFT"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.5 * cm))

    # 6.2 — Cross-format accuracy comparison
    elements.append(Paragraph(
        "6.2 Comparaison des performances par format (jeu de test)",
        ss["SubSectionHeading"],
    ))

    # Load validation_metrics.json if available
    metrics_path = _root() / "exports" / "validation_metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            vm = json.load(f)

        headers = ["Metrique", "Keras", "TFLite", "TFLite-fp16"]
        rows = [
            [
                "Accuracy",
                f"{vm['Keras']['accuracy']:.4f}",
                f"{vm['TFLite']['accuracy']:.4f}",
                f"{vm['TFLite-fp16']['accuracy']:.4f}",
            ],
            [
                "F1 (weighted)",
                f"{vm['Keras']['f1_weighted']:.4f}",
                f"{vm['TFLite']['f1_weighted']:.4f}",
                f"{vm['TFLite-fp16']['f1_weighted']:.4f}",
            ],
            [
                "F1 (macro)",
                f"{vm['Keras']['f1_macro']:.4f}",
                f"{vm['TFLite']['f1_macro']:.4f}",
                f"{vm['TFLite-fp16']['f1_macro']:.4f}",
            ],
            [
                "Precision (weighted)",
                f"{vm['Keras']['precision_weighted']:.4f}",
                f"{vm['TFLite']['precision_weighted']:.4f}",
                f"{vm['TFLite-fp16']['precision_weighted']:.4f}",
            ],
            [
                "Recall (weighted)",
                f"{vm['Keras']['recall_weighted']:.4f}",
                f"{vm['TFLite']['recall_weighted']:.4f}",
                f"{vm['TFLite-fp16']['recall_weighted']:.4f}",
            ],
            [
                "Vitesse (img/s)",
                f"{vm['Keras']['images_per_second']:.1f}",
                f"{vm['TFLite']['images_per_second']:.1f}",
                f"{vm['TFLite-fp16']['images_per_second']:.1f}",
            ],
            [
                "Taille fichier",
                f"{vm['Keras']['file_size_mb']} MB",
                f"{vm['TFLite']['file_size_mb']} MB",
                f"{vm['TFLite-fp16']['file_size_mb']} MB",
            ],
        ]

        col_widths = [4.0 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm]
        tbl = _make_table(headers, rows, col_widths)
        tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ]
            )
        )
        elements.append(tbl)
        elements.append(Spacer(1, 0.5 * cm))

        # 6.3 — Pairwise agreement
        elements.append(Paragraph(
            "6.3 Concordance entre formats (% predictions identiques)",
            ss["SubSectionHeading"],
        ))

        headers = ["Paire", "Concordance"]
        rows = [
            ["Keras vs TFLite", vm.get("agreement_Keras_vs_TFLite", "N/A")],
            ["Keras vs TFLite-fp16", vm.get("agreement_Keras_vs_TFLite-fp16", "N/A")],
            ["TFLite vs TFLite-fp16", vm.get("agreement_TFLite_vs_TFLite-fp16", "N/A")],
        ]

        col_widths = [6 * cm, 5 * cm]
        tbl = _make_table(headers, rows, col_widths)
        tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("TEXTCOLOR", (1, 1), (1, -1), GREEN_OK),
                    ("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"),
                ]
            )
        )
        elements.append(tbl)
        elements.append(Spacer(1, 0.3 * cm))

        elements.append(
            Paragraph(
                "<i>Note : La conversion TFLite-fp16 preserve 99.69 % de concordance avec le modele Keras "
                "original (seulement 3 images sur 964 different). La quantification dynamic range (TFLite) "
                "introduit une perte d'accuracy legerement plus importante (~0.8 %) mais produit un fichier "
                "significativement plus compact (0.5 MB).</i>",
                ss["BodyText2"],
            )
        )
    else:
        elements.append(
            Paragraph(
                "<b>[Fichier non trouve]</b> exports/validation_metrics.json — "
                "Executez d'abord : python3 -m tools.validate_exports",
                ss["BodyText2"],
            )
        )

    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_data_integrity(ss):
    """Section 7 — Data integrity / leakage verification."""
    elements = [
        Paragraph("7. Integrite des donnees et verification de fuite", ss["SectionHeading"]),
        _hr(),
    ]

    # Split summary
    elements.append(Paragraph("7.1 Repartition du dataset", ss["SubSectionHeading"]))

    headers = ["Split", "Images", "Pourcentage"]
    total = 4497 + 963 + 964
    rows = [
        ["Train", "4 497", f"{4497/total*100:.1f} %"],
        ["Validation", "963", f"{963/total*100:.1f} %"],
        ["Test", "964", f"{964/total*100:.1f} %"],
        ["Total", f"{total:,}".replace(",", " "), "100.0 %"],
    ]
    col_widths = [4 * cm, 4 * cm, 4 * cm]
    tbl = _make_table(headers, rows, col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                # Bold the total row
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("LINEABOVE", (0, -1), (-1, -1), 1.2, DARK_BLUE),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.4 * cm))

    # Leakage verification
    elements.append(Paragraph("7.2 Verification de fuite de donnees", ss["SubSectionHeading"]))

    headers = ["Verification", "Methode", "Resultat"]
    rows = [
        ["Fuite par chemin", "Comparaison des chemins de fichiers entre splits", "0 fuite detectee"],
        ["Fuite par contenu", "Hachage SHA-256 de chaque image", "0 fuite detectee"],
    ]
    col_widths = [4 * cm, 6 * cm, None]
    tbl = _make_table(headers, rows, col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("ALIGN", (1, 1), (1, -1), "LEFT"),
                ("ALIGN", (2, 1), (2, -1), "LEFT"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (2, 1), (2, -1), GREEN_OK),
                ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(tbl)

    elements.append(Spacer(1, 0.3 * cm))
    elements.append(
        Paragraph(
            "<i>Split : 70 / 15 / 15 avec seed=42.  "
            "Zero fuite de donnees verifiee par chemin + hash SHA-256.</i>",
            ss["BodyText2"],
        )
    )
    elements.append(Spacer(1, 0.5 * cm))
    return elements


def _section_conclusions(ss):
    """Section 8 — Conclusions / recommendations."""
    elements = [
        Paragraph("8. Conclusions et recommandations", ss["SectionHeading"]),
        _hr(),
    ]

    bullets = [
        "Le modele MobileNetV2 (alpha=0.35) atteint une <b>accuracy de 85.58 %</b> (Keras) et un "
        "<b>weighted F1 de 85.64 %</b> sur le jeu de test, ce qui est un excellent resultat "
        "pour un modele aussi leger (~426 K parametres, 1.63 MB).",

        "La <b>conversion TFLite-fp16</b> preserve 99.69 % de concordance avec le modele Keras "
        "(accuracy 85.37 %, F1 weighted 85.43 %). La perte est negligeable (3 images sur 964).",

        "La <b>conversion TFLite dynamic range</b> introduit une perte d'accuracy de ~0.8 % "
        "(accuracy 84.75 %, F1 weighted 84.77 %) mais produit le fichier le plus compact (0.5 MB) "
        "et offre la meilleure vitesse d'inference.",

        "La classe <b>Other</b> (F1 = 0.94) est la mieux classee, "
        "beneficiant du plus grand support (150 images de test). "
        "Cela confirme l'efficacite de la strategie de rejet des images non-alimentaires.",

        "La classe <b>Sandwich</b> (F1 = 0.70) est la plus faible. "
        "Cela s'explique par un support reduit (45 images) et des confusions "
        "frequentes avec les classes Burger et Hot Dog. "
        "Une augmentation du jeu de donnees pour cette classe est recommandee.",

        "L'exportation TensorFlow.js (1.5 MB total) est adaptee pour un deploiement "
        "dans le navigateur via le service de moderation Whispr.",

        "Aucune fuite de donnees n'a ete detectee (verification par chemin + SHA-256).",
    ]

    for b in bullets:
        elements.append(
            Paragraph(
                f"\u2022 {b}",
                ss["BodyText2"],
            )
        )

    elements.append(Spacer(1, 1 * cm))
    return elements


# ── Footer callback ───────────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(HexColor("#aaaaaa"))
    text = f"Whispr Moderation Service — Rapport de validation MobileNetV2-0.35  |  Page {doc.page}"
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, text)
    canvas.restoreState()


# ── Main: assemble and build PDF ─────────────────────────────────────────────
def main():
    ss = _build_styles()

    # Ensure output directory exists.
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Rapport de Validation — MobileNetV2-0.35",
        author="Whispr Moderation Service",
    )

    elements = []

    # 0 — Title
    elements.extend(_section_title_page(ss))

    # 1 — Model info
    elements.extend(_section_model_info(ss))

    # 2 — Training results
    elements.extend(_section_training_results(ss))

    # 3 — Per-class metrics
    elements.extend(_section_per_class(ss))

    # Page break before images
    elements.append(PageBreak())

    # 4 — Confusion matrix
    elements.extend(_section_confusion_matrix(ss))

    # 5 — Class performance chart
    elements.extend(_section_class_performance_chart(ss))

    # Page break before remaining sections
    elements.append(PageBreak())

    # 6 — Export formats
    elements.extend(_section_export_formats(ss))

    # 7 — Data integrity
    elements.extend(_section_data_integrity(ss))

    # 8 — Conclusions
    elements.extend(_section_conclusions(ss))

    # Build PDF
    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)

    print(f"PDF report generated: {OUTPUT_PDF}")
    print(f"  Size: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
