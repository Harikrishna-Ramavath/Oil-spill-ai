"""
utils/report.py

This module implements the PDF report generation system using ReportLab.
It gathers metadata, prediction outputs, and images (original and Grad-CAM)
to output a professionally formatted PDF document for downloading.
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def load_config():
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def generate_pdf_report(prediction_record, output_path):
    """
    Generates a professional PDF report.
    
    Args:
        prediction_record: Dict containing prediction details:
            - image_name
            - prediction_label
            - confidence
            - severity
            - spill_percentage
            - original_image_path
            - heatmap_path
            - created_at
        output_path: Path where the PDF should be saved.
    """
    config = load_config()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'), # Slate 900
        alignment=0,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'), # Slate 500
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155') # Slate 700
    )

    header_bold_style = ParagraphStyle(
        'HeaderBold',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0F172A')
    )

    # Header / Title Block
    story.append(Paragraph(config["project_name"], title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Verification Agency System", subtitle_style))
    
    # Divider Line
    line_data = [['']]
    line_table = Table(line_data, colWidths=[540], rowHeights=[2])
    line_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#06B6D4')), # Cyan accent line
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 15))

    # Details Block
    story.append(Paragraph("Incident Assessment Report", h2_style))
    
    # Format date
    try:
        dt = datetime.fromisoformat(prediction_record["created_at"])
        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        formatted_date = prediction_record.get("created_at", "N/A")

    # Table of details
    details_data = [
        [Paragraph("File Name:", header_bold_style), Paragraph(prediction_record["image_name"], body_style),
         Paragraph("Detection Timestamp:", header_bold_style), Paragraph(formatted_date, body_style)],
        
        [Paragraph("AI Classification:", header_bold_style), 
         Paragraph(f"<font color='#E11D48'><b>{prediction_record['prediction_label']}</b></font>" if prediction_record['prediction_label'] == 'Oil Spill' else f"<font color='#16A34A'><b>{prediction_record['prediction_label']}</b></font>", body_style),
         Paragraph("Confidence Level:", header_bold_style), 
         Paragraph(f"{prediction_record['confidence']:.2f}%", body_style)],
        
        [Paragraph("Estimated Severity:", header_bold_style), Paragraph(prediction_record["severity"], body_style),
         Paragraph("Spill Area Percentage:", header_bold_style), Paragraph(f"{prediction_record['spill_percentage']:.2f}%", body_style)]
    ]

    details_table = Table(details_data, colWidths=[120, 150, 130, 140])
    details_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')), # Slate 50
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')), # Slate 200
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(details_table)
    story.append(Spacer(1, 20))

    # Imagery Section
    story.append(Paragraph("Imagery and AI Interpretations", h2_style))

    # Load and resize images for the PDF
    # We fit both images into 250x250 boundaries
    orig_path = prediction_record["original_image_path"]
    heat_path = prediction_record["heatmap_path"]

    img_width, img_height = 250, 250
    
    orig_flowable = RLImage(orig_path, width=img_width, height=img_height)
    heat_flowable = RLImage(heat_path, width=img_width, height=img_height)

    image_table_data = [
        [orig_flowable, heat_flowable],
        [Paragraph("<font color='#64748B'><i>Figure 1. Original Imagery</i></font>", body_style),
         Paragraph("<font color='#64748B'><i>Figure 2. Grad-CAM Attention Heatmap</i></font>", body_style)]
    ]

    image_table = Table(image_table_data, colWidths=[270, 270])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
    ]))
    
    story.append(image_table)
    story.append(Spacer(1, 15))

    # Analytical / Explanatory text
    story.append(Paragraph("Diagnostic Summary", h2_style))
    if prediction_record['prediction_label'] == 'Oil Spill':
        spill_text = (
            f"The Deep Learning model has detected pixel patterns consistent with an active oil spill in {prediction_record['image_name']}. "
            f"Grad-CAM activation mapping (Figure 2) illustrates that the neural network's decision was strongly influenced by the "
            f"dark, high-absorption slick segments, confirming target anomalies. The total affected footprint is estimated at "
            f"{prediction_record['spill_percentage']:.2f}% of the observation window, classifying this event as <b>{prediction_record['severity']} Severity</b>. "
            f"Immediate review and containment measures are recommended."
        )
    else:
        spill_text = (
            f"Analytical run completed successfully for image: {prediction_record['image_name']}. "
            "No anomalous dark bands matching the signature of oil slick characteristics were identified. "
            "Model gradients indicate normal background scattering from water, clouds, or surface glint. "
            "No action is required at this time."
        )
    
    story.append(Paragraph(spill_text, body_style))
    
    # Build Document
    doc.build(story)
    logger.info(f"Report PDF generated at: {output_path}")
