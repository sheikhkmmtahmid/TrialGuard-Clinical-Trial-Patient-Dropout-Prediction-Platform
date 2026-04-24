"""
PDF report generator for TrialGuard patient risk reports.
Uses ReportLab with full TrialGuard branding.
"""
import io
import base64
import logging
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

logger = logging.getLogger('core')

CRIMSON = colors.HexColor('#003087')      # NHS Blue
GOLD = colors.HexColor('#0072CE')         # NHS Bright Blue
DARK_BG = colors.HexColor('#003087')      # NHS Blue header bg
CARD_BG = colors.HexColor('#F0F4F5')      # NHS Pale Grey
TEXT_PRIMARY = colors.HexColor('#FFFFFF') # White text on dark header
ACCENT = colors.HexColor('#00A9CE')       # NHS Aqua

TIER_COLOURS = {
    'low': colors.HexColor('#009639'),    # NHS Green
    'medium': colors.HexColor('#FFB81C'), # NHS Warm Yellow
    'high': colors.HexColor('#E65C00'),    # Vivid Orange
    'critical': colors.HexColor('#CC0000'), # Vivid Red
}

REPORTS_DIR = Path(__file__).resolve().parents[2] / 'media' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOGO_PATH = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'logo.svg'


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'TGTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=GOLD,
        spaceAfter=4,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'TGHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=GOLD,
        spaceAfter=4,
        spaceBefore=10,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'TGBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        leading=14,
    ))
    styles.add(ParagraphStyle(
        'TGSmall',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        'TGCaption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=CRIMSON,
        alignment=TA_CENTER,
    ))
    return styles


def _section_rule():
    return HRFlowable(width='100%', thickness=1, color=GOLD, spaceAfter=6, spaceBefore=2)


def _decode_image(b64_str: str):
    """Convert base64 PNG string to a ReportLab Image flowable."""
    if not b64_str:
        return None
    try:
        data = base64.b64decode(b64_str)
        buf = io.BytesIO(data)
        img = RLImage(buf, width=16 * cm, height=8 * cm, kind='proportional')
        return img
    except Exception as e:
        logger.warning("Could not decode image for report: %s", e)
        return None


def generate_patient_report(patient, prediction, coordinator_actions,
                             survival_b64: str = None,
                             shap_b64: str = None,
                             risk_timeline_b64: str = None) -> str:
    """
    Generate a branded PDF report for a patient. Returns the file path.
    """
    filename = f"patient_{patient.pk}_report_{date.today().isoformat()}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
        title=f'TrialGuard — Patient #{patient.pk} Risk Report',
        author='TrialGuard',
    )

    styles = _build_styles()
    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    header_data = [
        [
            Paragraph('<b><font color="#ffffff" size="18">Trial</font>'
                      '<font color="#41B6E6" size="18">Guard</font></b>', styles['TGBody']),
            Paragraph(
                '<font color="#41B6E6" size="9">60-Day Early Warning for Clinical Trial Retention</font>',
                styles['TGBody']
            ),
            Paragraph(
                f'<font color="#888888" size="8">Generated: {date.today().strftime("%d %b %Y")}</font>',
                ParagraphStyle('right', parent=styles['TGBody'], alignment=TA_RIGHT, fontSize=8)
            ),
        ]
    ]
    header_table = Table(header_data, colWidths=[5 * cm, 9 * cm, 4.5 * cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), DARK_BG),
        ('TEXTCOLOR', (0, 0), (-1, -1), TEXT_PRIMARY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (0, -1), 12),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # ── TITLE ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(f'Patient Risk Assessment Report — #{patient.pk}', styles['TGTitle']))
    story.append(_section_rule())

    # ── PATIENT SUMMARY ───────────────────────────────────────────────────────
    story.append(Paragraph('Patient Summary', styles['TGHeading']))

    gender_label = dict(patient.GENDER_CHOICES).get(patient.gender, patient.gender)
    severity_label = patient.condition_severity.capitalize()
    employment_label = patient.employment_status.replace('_', ' ').title()

    summary_data = [
        ['Trial', patient.trial.name, 'Phase', patient.trial.phase],
        ['Sponsor', patient.trial.sponsor, 'Therapeutic Area', patient.trial.therapeutic_area],
        ['Patient ID', f'#{patient.pk}', 'Enrollment Date', patient.enrollment_date.strftime('%d %b %Y')],
        ['Age', str(patient.age), 'Gender', gender_label],
        ['Ethnicity', patient.ethnicity.title(), 'Severity', severity_label],
        ['Distance to Site', f'{patient.distance_to_site_km:.1f} km', 'Employment', employment_label],
        ['Prior Dropout History', 'Yes' if patient.prior_dropout_history else 'No',
         'Status', 'Dropped Out' if patient.dropout_status else 'Active'],
    ]

    summary_table = Table(summary_data, colWidths=[4.5 * cm, 6.5 * cm, 4.5 * cm, 3 * cm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), CRIMSON),
        ('TEXTCOLOR', (2, 0), (2, -1), CRIMSON),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF8F0')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#FFF8F0')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0D0C0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # ── CURRENT RISK TIER ─────────────────────────────────────────────────────
    if prediction:
        story.append(Paragraph('Current Risk Assessment', styles['TGHeading']))

        tier_colour = TIER_COLOURS.get(prediction.risk_tier, CRIMSON)
        prob_pct = f'{prediction.dropout_probability:.1%}'

        risk_data = [
            ['Risk Tier', prediction.risk_tier.upper(),
             'Dropout Probability', prob_pct],
            ['Hazard Ratio', f'{prediction.hazard_ratio:.3f}' if prediction.hazard_ratio else 'N/A',
             'Survival Estimate', f'{prediction.survival_time_estimate:.0f} days' if prediction.survival_time_estimate else 'N/A'],
            ['Model Version', prediction.model_version,
             'Assessment Date', prediction.prediction_timestamp.strftime('%d %b %Y %H:%M')],
        ]

        risk_table = Table(risk_data, colWidths=[4.5 * cm, 5.5 * cm, 4.5 * cm, 4 * cm])
        risk_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 0), (1, 0), tier_colour),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 11),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF8F0')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#FFF8F0')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0D0C0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 8))

        # ── PLAIN ENGLISH EXPLANATION ─────────────────────────────────────────
        story.append(Paragraph('Risk Explanation', styles['TGHeading']))
        story.append(Paragraph(prediction.plain_english_explanation(), styles['TGBody']))
        story.append(Spacer(1, 8))

        # ── TOP 5 DRIVERS ─────────────────────────────────────────────────────
        top_features = prediction.shap_top_features()
        if top_features:
            story.append(Paragraph('Top Dropout Risk Drivers (SHAP Analysis)', styles['TGHeading']))
            driver_data = [['Rank', 'Feature', 'Direction', 'SHAP Impact']]
            for i, feat in enumerate(top_features, 1):
                direction_label = '▲ Increases Risk' if feat['direction'] == 'increases' else '▼ Decreases Risk'
                driver_data.append([
                    str(i),
                    feat['feature'].replace('_', ' ').title(),
                    direction_label,
                    f"{feat['shap_value']:+.4f}",
                ])
            driver_table = Table(driver_data, colWidths=[1.5 * cm, 6.5 * cm, 5 * cm, 4 * cm])
            driver_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), CRIMSON),
                ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_PRIMARY),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FFF8F0')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0D0C0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (3, 0), (3, -1), 'CENTER'),
            ]))
            story.append(driver_table)
            story.append(Spacer(1, 10))

    # ── CHARTS ────────────────────────────────────────────────────────────────
    for b64, caption in [
        (risk_timeline_b64, 'Figure 1: Dropout Probability Across Visits'),
        (survival_b64, 'Figure 2: Patient Survival (Retention) Curve'),
        (shap_b64, 'Figure 3: SHAP Waterfall — Top Dropout Drivers'),
    ]:
        if b64:
            img = _decode_image(b64)
            if img:
                story.append(Paragraph(caption.split(': ')[1], styles['TGHeading']))
                story.append(img)
                story.append(Paragraph(caption, styles['TGCaption']))
                story.append(Spacer(1, 8))

    # ── COORDINATOR ACTIONS ───────────────────────────────────────────────────
    if coordinator_actions:
        story.append(Paragraph('Coordinator Intervention History', styles['TGHeading']))
        action_data = [['Date', 'Action Type', 'Outcome', 'Notes']]
        for action in coordinator_actions[:20]:
            action_data.append([
                action.action_date.strftime('%d %b %Y'),
                action.action_type.replace('_', ' ').title(),
                action.outcome.replace('_', ' ').title(),
                (action.notes[:60] + '...') if len(action.notes) > 60 else action.notes,
            ])
        action_table = Table(action_data, colWidths=[3 * cm, 4.5 * cm, 4.5 * cm, 6.5 * cm])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), CRIMSON),
            ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_PRIMARY),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FFF8F0')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0D0C0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('WORDWRAP', (3, 1), (3, -1), True),
        ]))
        story.append(action_table)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(_section_rule())
    story.append(Paragraph(
        '<font color="#888888" size="8">'
        'Generated by TrialGuard — 60-Day Early Warning for Clinical Trial Retention | '
        'Built by SKMMT | skmmt.rootexception.com | '
        'Powered by XGBoost &amp; Survival Analysis | '
        f'Report Date: {date.today().strftime("%d %b %Y")}'
        '</font>',
        ParagraphStyle('footer', parent=styles['TGSmall'], alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        '<font color="#999" size="7">'
        'DISCLAIMER: This report is generated by an AI-assisted decision support tool. '
        'All clinical decisions must be made by qualified healthcare professionals. '
        'Synthetic data disclosure: training data includes SDV-generated synthetic patients.'
        '</font>',
        ParagraphStyle('disclaimer', parent=styles['TGSmall'], alignment=TA_CENTER)
    ))

    doc.build(story)
    logger.info("Report generated: %s", filepath)
    return str(filepath)
