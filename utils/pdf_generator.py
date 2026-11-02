import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from config import REPORTS_DIR

def generate_pdf_report(eval_data, candidate_data):
    """
    Generates a PDF driving evaluation report for the candidate using ReportLab.
    Returns absolute path to generated PDF.
    """
    pdf_filename = f"Driving_Report_{candidate_data['candidate_id']}_{eval_data['id']}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Title & Header Styles
    title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    # 1. Header Header
    story.append(Paragraph("AUTOMATED DRIVING SKILL VALIDATION REPORT", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Regional Transport Authority • AI Driving Test System", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563EB'), spaceAfter=15))

    # 2. Candidate Information Table with Passport Photo
    photo_cell = Paragraph("<b>[ No Photo ]</b>", body_style)
    photo_path = candidate_data.get('photo_path', '')
    if photo_path and os.path.exists(photo_path):
        try:
            img = Image(photo_path, width=1.1*inch, height=1.3*inch)
            photo_cell = img
        except Exception:
            pass

    info_text = f"""
    <b>Candidate Name:</b> {candidate_data.get('name', 'N/A')}<br/>
    <b>Candidate ID:</b> {candidate_data.get('candidate_id', 'N/A')}<br/>
    <b>Aadhaar Number:</b> {candidate_data.get('aadhaar', 'N/A')}<br/>
    <b>Date of Birth:</b> {candidate_data.get('dob', 'N/A')} &nbsp;|&nbsp; <b>Gender:</b> {candidate_data.get('gender', 'N/A')}<br/>
    <b>Mobile:</b> {candidate_data.get('mobile', 'N/A')} &nbsp;|&nbsp; <b>Email:</b> {candidate_data.get('email', 'N/A')}<br/>
    <b>Address:</b> {candidate_data.get('address', 'N/A')}
    """

    info_table_data = [
        [Paragraph(info_text, body_style), photo_cell]
    ]

    info_table = Table(info_table_data, colWidths=[5.5*inch, 1.5*inch])
    info_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('ROUNDEDCORNERS', [4, 4, 4, 4])
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # 3. Overall Driving Test Result Summary Card
    res_upper = eval_data['overall_result'].upper()
    is_pass = res_upper == 'PASS'
    is_review = res_upper in ['NEEDS REVIEW', 'NEEDS_REVIEW']

    if is_pass:
        res_bg = colors.HexColor('#DCFCE7')
        res_border = colors.HexColor('#16A34A')
        res_text_color = '#15803D'
    elif is_review:
        res_bg = colors.HexColor('#FEF3C7')
        res_border = colors.HexColor('#D97706')
        res_text_color = '#B45309'
    else:
        res_bg = colors.HexColor('#FEE2E2')
        res_border = colors.HexColor('#DC2626')
        res_text_color = '#B91C1C'

    result_card_html = f"""
    <font size=12 color="{res_text_color}"><b>FINAL EVALUATION RESULT:</b></font><br/>
    <font size=24 color="{res_text_color}"><b>{eval_data['overall_result'].upper()}</b></font><br/>
    <font size=10 color="#475569">Overall Skill Score: <b>{eval_data['overall_score']}%</b> &nbsp;|&nbsp; Evaluation Date: {eval_data.get('evaluated_at', 'Today')}</font>
    """

    result_table = Table([[Paragraph(result_card_html, body_style)]], colWidths=[7.0*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), res_bg),
        ('BOX', (0,0), (-1,-1), 2, res_border),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 12)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 15))

    # 4. Computer Vision Module Detection Breakdown
    story.append(Paragraph("Driving Test Module Analysis", section_heading))

    auth_str = "PASSED" if eval_data['auth_status'] else "FAILED"
    if eval_data.get('helmet_state') in ['Needs Review', 'NEEDS_REVIEW', 'UNCLEAR', 'NOT VERIFIED', 'Not Verified']:
        helm_str = "NOT VERIFIED"
    elif eval_data.get('helmet_status') or eval_data.get('helmet_state') == 'PASS':
        helm_str = "PASS"
    else:
        helm_str = "FAIL"

    if eval_data.get('lane_state') in ['NEEDS_REVIEW', 'Needs Review', 'UNCLEAR', 'NOT VERIFIED']:
        lane_str = "NOT VERIFIED (Inconclusive Tracking)"
    elif eval_data.get('lane_status'):
        lane_str = "PASS"
    else:
        lane_str = "FAIL (Boundary Violation)"

    foot_count = eval_data.get('foot_touch_count', 0)
    if eval_data.get('foot_touch_status') and foot_count == 0:
        foot_str = "NOT DETECTED (0 Events)"
    else:
        foot_str = f"DETECTED ({foot_count} Event{'s' if foot_count > 1 else ''})"
    bal_str = "PASS (Stable)" if eval_data['balance_status'] else "FAIL (Unstable / Imbalance)"

    if eval_data.get('braking_state') in ['NOT VERIFIED', 'Unable to Verify', 'Needs Review', 'NEEDS REVIEW', 'NEEDS_REVIEW']:
        brake_str = "NEEDS REVIEW"
    elif eval_data.get('braking_state') == 'FAIL' or (eval_data.get('braking_status') == 0 and eval_data.get('braking_status') is not None):
        brake_str = "FAIL (Sudden Braking Detected)"
    else:
        brake_str = "PASS (Normal / Controlled Braking)"

    module_data = [
        [Paragraph("<b>Driving Skill Metric</b>", body_style), Paragraph("<b>Status Result</b>", body_style)],
        [Paragraph("Candidate Authentication", body_style), Paragraph(f"<b>{auth_str}</b>", body_style)],
        [Paragraph("Helmet Wearing Detection", body_style), Paragraph(f"<b>{helm_str}</b>", body_style)],
        [Paragraph("Lane Line Tracking", body_style), Paragraph(f"<b>{lane_str}</b>", body_style)],
        [Paragraph("Foot-to-Ground Contact", body_style), Paragraph(f"<b>{foot_str}</b>", body_style)],
        [Paragraph("Rider Posture & Balance", body_style), Paragraph(f"<b>{bal_str}</b>", body_style)],
        [Paragraph("Braking Control", body_style), Paragraph(f"<b>{brake_str}</b>", body_style)],
    ]

    mod_table = Table(module_data, colWidths=[4.2*inch, 2.8*inch])
    mod_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 15))

    # 6. Detected Mistakes & Faults
    story.append(Paragraph("Detected Driving Errors & Penalty Remarks", section_heading))
    mistakes = eval_data.get('mistakes', [])
    if mistakes:
        mistake_items = "<br/>".join([f"• <font color='#DC2626'><b>{m}</b></font>" for m in mistakes])
        mistakes_html = f"The automated system recorded the following violation(s) during testing:<br/><br/>{mistake_items}"
    else:
        mistakes_html = "<font color='#16A34A'><b>No driving violations detected. The candidate demonstrated safe and stable driving performance.</b></font>"

    m_table = Table([[Paragraph(mistakes_html, body_style)]], colWidths=[7.0*inch])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFBEB') if mistakes else colors.HexColor('#F0FDF4')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#F59E0B') if mistakes else colors.HexColor('#4ADE80')),
        ('PADDING', (0,0), (-1,-1), 10)
    ]))
    story.append(m_table)
    story.append(Spacer(1, 25))

    # 7. Signature Footer
    sig_data = [
        [Paragraph("<b>Automated System Inspector</b><br/>AI Video Analytics Engine", body_style),
         Paragraph("<b>Authorized Officer Signature</b><br/>Regional Transport Office (RTO)", body_style)]
    ]
    sig_table = Table(sig_data, colWidths=[3.5*inch, 3.5*inch])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(sig_table)

    doc.build(story)
    return pdf_path
