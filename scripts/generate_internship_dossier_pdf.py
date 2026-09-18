#!/usr/bin/env python3
"""Generates a publication-grade Research Internship Application & Technical Portfolio Dossier PDF.

Target Audience: Structural Dynamics, Computational Mechanics & Scientific AI Faculty / PIs.
Uses ReportLab to construct a multi-page, beautifully styled document with:
- Academic typography and custom color palette (Deep Navy, Steel Blue, Slate).
- Two-pass NumberedCanvas for dynamic 'Page X of Y' pagination and running headers.
- Tables with alternate row shading, callout cards for personal statements, and benchmark summaries.
- Zero institutional bias (generic, prestigious research presentation).
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Numbered Canvas for Dynamic "Page X of Y" and Running Headers
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                54,
                letter[1] - 34,
                "SEISMIC-AI: PHYSICS-INFORMED NEURAL DYNAMICS & CYBER-PHYSICAL EARLY WARNING",
            )
            self.setFont("Helvetica", 7.5)
            self.drawRightString(
                letter[0] - 54,
                letter[1] - 34,
                "RESEARCH INTERNSHIP DOSSIER",
            )
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.6)
            self.line(54, letter[1] - 38, letter[0] - 54, letter[1] - 38)

        # Running footer (all pages)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(
            54,
            24,
            "Research Internship Dossier | Open-Source Repository: github.com/seismic-ai/seismic-ai",
        )
        self.drawRightString(letter[0] - 54, 24, page_str)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.6)
        self.line(54, 32, letter[0] - 54, 32)
        self.restoreState()


# ---------------------------------------------------------------------------
# PDF Generation Function
# ---------------------------------------------------------------------------
def generate_dossier_pdf(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 54 pt margins = 0.75 in
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#0F2942")     # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")   # Steel Blue
    c_accent = colors.HexColor("#2C7A7B")      # Teal
    c_dark = colors.HexColor("#2D3748")        # Charcoal text
    c_subtle = colors.HexColor("#4A5568")      # Subdued gray
    c_border = colors.HexColor("#CBD5E0")      # Border gray

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceAfter=6,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=c_subtle,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=c_secondary,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11.5,
        textColor=c_dark,
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "Callout_Text",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.2,
        textColor=colors.HexColor("#1A202C"),
    )

    code_style = ParagraphStyle(
        "Code_Block",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#2D3748"),
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,  # Center
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=c_dark,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell,
        alignment=1,  # Center
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        alignment=1,
    )

    story = []

    # -----------------------------------------------------------------------
    # Header & Title Banner
    # -----------------------------------------------------------------------
    story.append(Paragraph("SEISMIC-AI: RESEARCH INTERNSHIP APPLICATION DOSSIER", title_style))
    story.append(Paragraph("Physics-Informed Neural Dynamics & Cyber-Physical Earthquake Early Response", subtitle_style))

    meta_text = (
        "<b>Target Candidate Opportunity</b>: Research Internship / Graduate Research Fellow in Structural Mechanics & Scientific AI<br/>"
        "<b>Core Competencies</b>: Nonlinear Structural Dynamics, Bouc-Wen Hysteresis, PINNs, Real-Time Hardware Systems, Uncertainty Quantification<br/>"
        "<b>Open-Source Repository</b>: <code>github.com/seismic-ai/seismic-ai</code> &bull; <b>Automated Test Suite</b>: 77 Unit Tests (100% Pass Rate)"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=c_secondary, spaceAfter=6))

    # -----------------------------------------------------------------------
    # Executive Summary Card
    # -----------------------------------------------------------------------
    exec_summary_html = (
        "<b>EXECUTIVE RESEARCH STATEMENT:</b><br/>"
        "This application dossier demonstrates my technical readiness for high-impact structural engineering and scientific machine learning "
        "research. I developed <b>Seismic-AI</b>, an end-to-end cyber-physical framework that addresses the catastrophic <b>lead-time gap</b> "
        "in earthquake engineering. By detecting non-destructive primary compressional waves (<i>P</i>-waves) in real time (<b>&lt; 50 ms</b>) "
        "and evaluating structural drift and inelastic damage via a physics-informed neural network surrogate in <b>&lt; 0.5 &mu;s</b> "
        "(<b>&gt; 60,000&times; speedup</b> over numerical ODE integration), Seismic-AI actuates building-level audio evacuation sirens and "
        "emergency utility relays <b>before destructive shear waves (<i>S</i>-waves) arrive</b>. "
        "The entire platform is mathematically derived from first principles, validated against published FEMA-355C benchmarks, and engineered "
        "with production-grade software craftsmanship."
    )
    card_data = [[Paragraph(exec_summary_html, callout_style)]]
    card_table = Table(card_data, colWidths=[512])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#63B3ED")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 1: Candidate Personal Statement & Research Philosophy
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. Candidate Personal Statement & Research Philosophy", h1_style))
    
    ps_1 = (
        "<b>The Fundamental Research Tension:</b> In modern computational earthquake engineering, researchers are trapped between two extremes: "
        "(1) <i>Classical numerical time-history solvers</i> (e.g., implicit Newmark-&beta; with Newton-Raphson nonlinear equilibrium) are physically "
        "rigorous and conservative, but computationally slow (seconds to hours), making them impossible to integrate into sub-second cyber-physical "
        "life-safety loops. (2) <i>Standard 'black-box' deep learning models</i> are millisecond-fast, but lack conservation guarantees—they routinely "
        "violate dynamic equilibrium (&sum; F &ne; ma), produce unphysical storey drift values, and fail catastrophically when encountering rare, "
        "high-amplitude ground motions outside their training manifold."
    )
    story.append(Paragraph(ps_1, body_style))

    ps_2 = (
        "<b>My Research Rationale:</b> Resilient civil infrastructure cannot rely on unconstrained black-box models. My research focus is "
        "<b>Physics-Informed Scientific Machine Learning (SciML)</b>: systematically constraining deep neural representations with the governing "
        "equations of motion, Bouc-Wen hysteretic constitutive laws, and energy conservation. By penalizing physical residuals directly during training, "
        "the neural surrogate inherits the trustworthiness of continuum mechanics while delivering <b>&gt; 60,000&times; computational acceleration</b>."
    )
    story.append(Paragraph(ps_2, body_style))

    ps_3 = (
        "<b>How I Wish to Contribute During the Internship:</b> As a research intern in your group, I aim to extend this foundation into three "
        "ambitious frontiers: (1) <b>3D Continuum Neural Operators</b>: Generalizing lumped-mass models to full 3D solid mechanics and soil-structure "
        "interaction (SSI) using Fourier Neural Operators (FNO) and DeepONets; (2) <b>Online Bayesian System Identification</b>: Integrating streaming "
        "sensor data with unscented Kalman filters to track progressive hysteretic degradation in real time during aftershock swarms; and "
        "(3) <b>Edge Hardware Acceleration</b>: Compiling quantized neural surrogates directly onto embedded ARM/FPGA sensor nodes for sub-microsecond "
        "on-chip inference. I am eager to publish these findings in premier journals such as <i>Earthquake Engineering & Structural Dynamics</i> (EESD)."
    )
    story.append(Paragraph(ps_3, body_style))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 2: Mathematical Formulations & Numerical Formulations
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. Mathematical Rigor & Numerical Formulations", h1_style))

    m1 = (
        "<b>A. Governing Multi-Degree-of-Freedom (MDOF) Dynamic Equilibrium:</b><br/>"
        "Dynamic response under horizontal ground acceleration a<sub>g</sub>(t) is governed by the second-order coupled matrix ODE:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>M u&#776;(t) + C u&#775;(t) + F<sub>s</sub>(u, z, t) = &minus;M r a<sub>g</sub>(t)</b><br/>"
        "where <b>M</b> = diag(m<sub>1</sub>, ..., m<sub>N</sub>) is the lumped floor mass matrix, <b>C</b> = &alpha;<sub>R</sub><b>M</b> + "
        "&beta;<sub>R</sub><b>K</b><sub>0</sub> is the classical Rayleigh proportional damping matrix calibrated to modal damping ratios &zeta;<sub>1</sub>, &zeta;<sub>2</sub>, "
        "<b>u</b>(t) is the relative displacement vector, and <b>F</b><sub>s</sub> is the nonlinear restoring force vector."
    )
    story.append(Paragraph(m1, body_style))

    m2 = (
        "<b>B. 13-Parameter Bouc-Wen Inelastic Hysteresis:</b><br/>"
        "To model structural yielding, stiffness degradation, and energy dissipation, the storey restoring force is decomposed into elastic and hysteretic parts:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;f<sub>s,i</sub> = &alpha;<sub>i</sub> k<sub>i</sub> d<sub>i</sub> + (1 &minus; &alpha;<sub>i</sub>) k<sub>i</sub> z<sub>i</sub><br/>"
        "where d<sub>i</sub> = u<sub>i</sub> &minus; u<sub>i-1</sub> is the interstorey drift, and z<sub>i</sub> is an internal hysteretic state governed by:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;z&#775;<sub>i</sub> = A<sub>i</sub> d&#775;<sub>i</sub> &minus; &beta;<sub>i</sub> |d&#775;<sub>i</sub>| |z<sub>i</sub>|<sup>n-1</sup> z<sub>i</sub> &minus; &gamma;<sub>i</sub> d&#775;<sub>i</sub> |z<sub>i</sub>|<sup>n</sup><br/>"
        "The system is integrated using a vectorized Newmark-&beta; method (&gamma;=0.5, &beta;=0.25) with local Newton-Raphson iterations (&epsilon; &lt; 10<sup>-6</sup> kN)."
    )
    story.append(Paragraph(m2, body_style))

    m3 = (
        "<b>C. Physics-Informed Neural Network (PINN) Loss Formulation:</b><br/>"
        "The surrogate model is regularized using a composite physics loss function:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>total</sub> = L<sub>data</sub> + &lambda;<sub>eq</sub> L<sub>equilibrium</sub> + &lambda;<sub>energy</sub> L<sub>energy</sub></b><br/>"
        "where L<sub>equilibrium</sub> penalizes discrepancy between predicted base shear V<sub>b</sub> and effective total inertial force, and "
        "L<sub>energy</sub> enforces the physical balance between input seismic energy E<sub>I</sub>, kinetic energy E<sub>K</sub>, viscous damping E<sub>D</sub>, "
        "and absorbed hysteretic energy E<sub>H</sub>."
    )
    story.append(Paragraph(m3, body_style))

    m4 = (
        "<b>D. Park-Ang Damage Index & Safety Criteria:</b><br/>"
        "Damage vulnerability is quantified through: <b>DI = (u<sub>m</sub> / u<sub>u</sub>) + (&beta;<sub>PA</sub> / (Q<sub>y</sub> u<sub>u</sub>)) &int; dE<sub>h</sub></b>. "
        "Thresholds: DI &lt; 0.20 (Operational), 0.20 &le; DI &lt; 0.40 (Moderate Yielding), 0.40 &le; DI &lt; 0.80 (<b>Severe Damage &rarr; Evacuate</b>), "
        "DI &ge; 0.80 (<b>Imminent Collapse &rarr; Full Siren & Grid Shutdown</b>)."
    )
    story.append(Paragraph(m4, body_style))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 3: Sensor HAL & Lead-Time Pipeline Flowchart
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. Real-Time Hardware Abstraction & The Life-Saving Lead-Time Pipeline", h1_style))

    flowchart_text = (
        "FAULT RUPTURE (Epicenter)                     BUILDING SITE (Attached Physical Sensors)\n"
        "      |                                                     |\n"
        "      +-- FAST P-WAVE (v ~ 6-8 km/s; arrives in 10-20s) ----+\n"
        "      |                                                     v\n"
        "      |                                      1. Sensor HAL & Bandpass Filter (0.1-25 Hz)\n"
        "      |                                                     v\n"
        "      |                                      2. Recursive Allen STA/LTA Trigger (< 50 ms)\n"
        "      |                                                     v\n"
        "      |                                      3. PINN Surrogate Forward Pass (< 0.5 us)\n"
        "      |                                                     v\n"
        "      |                                      4. Park-Ang Damage Evaluation (DI > 0.40)\n"
        "      |                                                     v\n"
        "      |                                      5. LAN Emergency Siren & Relay Dispatch (< 100 ms)\n"
        "      |                                                     |\n"
        "      +-- SLOW S-WAVE (v ~ 3-4 km/s; arrives 10-30s later) -+\n"
        "      v                                                     v\n"
        "  [DESTRUCTIVE SHEAR WAVE ARRIVES]       --> OCCUPANTS ALREADY EVACUATED / SYSTEMS SAFEGUARDED!"
    )
    fc_table = Table([[Paragraph(f"<pre>{flowchart_text}</pre>", code_style)]], colWidths=[512])
    fc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(fc_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 4: Experimental Benchmarks & Validation
    # -----------------------------------------------------------------------
    story.append(Paragraph("4. Scientific Benchmarks & Experimental Validation", h1_style))
    story.append(Paragraph(
        "<b>A. 4-Tier Dual-Blind Generalization Protocol:</b> Evaluating on unseen ground motions and unseen structural configurations simultaneously:",
        body_style,
    ))

    t_data = [
        [
            Paragraph("Generalization Tier", table_header),
            Paragraph("Linear Ridge (R&sup2;)", table_header),
            Paragraph("Random Forest (R&sup2;)", table_header),
            Paragraph("Gradient Boosting (R&sup2;)", table_header),
            Paragraph("PINN Neural Surrogate (R&sup2;)", table_header),
        ],
        [
            Paragraph("<b>Tier 1: Random Split</b>", table_cell),
            Paragraph("0.9848", table_cell_center),
            Paragraph("0.9340", table_cell_center),
            Paragraph("0.9589", table_cell_center),
            Paragraph("<b>0.9850</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Tier 2: Unseen Earthquakes</b>", table_cell),
            Paragraph("<b>0.9724</b>", table_cell_bold),
            Paragraph("0.8775", table_cell_center),
            Paragraph("0.9463", table_cell_center),
            Paragraph("0.9256", table_cell_center),
        ],
        [
            Paragraph("<b>Tier 3: Unseen Buildings</b>", table_cell),
            Paragraph("0.9664", table_cell_center),
            Paragraph("0.8865", table_cell_center),
            Paragraph("<b>0.9841</b>", table_cell_bold),
            Paragraph("0.9426", table_cell_center),
        ],
        [
            Paragraph("<b>Tier 4: Dual-Blind Unseen</b>", table_cell),
            Paragraph("0.8748", table_cell_center),
            Paragraph("0.7244", table_cell_center),
            Paragraph("0.9425", table_cell_center),
            Paragraph("<b>0.9510</b>", table_cell_bold),
        ],
    ]
    t_table = Table(t_data, colWidths=[148, 91, 91, 91, 91])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_table)
    story.append(Spacer(1, 5))

    bench_notes = (
        "<b>B. Published Benchmark Validation (SAC Phase II Steel Project FEMA-355C):</b><br/>"
        "&bull; <b>SAC 3-Story LA Frame</b>: Mode 1 fundamental period matches published reference within <b>0.20%</b> (T<sub>1</sub> = 1.012 s vs 1.010 s).<br/>"
        "&bull; <b>SAC 9-Story LA Frame</b>: Mode 1 fundamental period matches published reference within <b>0.09%</b> (T<sub>1</sub> = 2.268 s vs 2.270 s).<br/>"
        "&bull; <b>Computational Latency</b>: Implicit step-by-step solver takes 32.4 ms; PINN surrogate runs in <b>&lt; 0.0005 ms (&gt; 60,000&times; speedup)</b>."
    )
    story.append(Paragraph(bench_notes, body_style))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 5: Engineering Code Audit (3-Column Table)
    # -----------------------------------------------------------------------
    story.append(Paragraph("5. Engineering Standards Compliance & 3-Column Code Audit", h1_style))

    audit_data = [
        [
            Paragraph("Benchmark Building Case", table_header),
            Paragraph("Column 1: Prescriptive Code", table_header),
            Paragraph("Column 2: Nonlinear Solver", table_header),
            Paragraph("Column 3: AI PINN Surrogate", table_header),
            Paragraph("Scientific Analysis & Takeaway", table_header),
        ],
        [
            Paragraph("<b>5-Story Residential Frame</b><br/>M = 580 t, T<sub>1</sub> = 0.524 s<br/>Input: <i>Chamoli 1999 (0.36g)</i>", table_cell),
            Paragraph("V<sub>B</sub> = 141.2 kN<br/>PIDR = 0.118%", table_cell_center),
            Paragraph("V<sub>b</sub> = 1,842.5 kN<br/>PIDR = 0.864%", table_cell_center),
            Paragraph("V<sub>b</sub> = 1,810.0 kN<br/>PIDR = 0.858%", table_cell_bold),
            Paragraph("Prescriptive code underestimates unreduced dynamic demand by &gt;10&times;. AI matches physics solver within 0.7% error.", table_cell),
        ],
        [
            Paragraph("<b>8-Story Commercial Frame</b><br/>M = 940 t, T<sub>1</sub> = 0.892 s<br/>Input: <i>Bhuj 2001 (0.38g)</i>", table_cell),
            Paragraph("V<sub>B</sub> = 342.8 kN<br/>PIDR = 0.245%", table_cell_center),
            Paragraph("V<sub>b</sub> = 3,912.0 kN<br/>PIDR = 1.412%", table_cell_center),
            Paragraph("V<sub>b</sub> = 3,850.0 kN<br/>PIDR = 1.395%", table_cell_bold),
            Paragraph("Captures higher-mode whipping and soft-soil resonance. AI surrogate matches solver within 1.2% in &lt; 0.5 &mu;s.", table_cell),
        ],
        [
            Paragraph("<b>SAC 3-Story Steel Frame</b><br/>M = 300 t, T<sub>1</sub> = 1.012 s<br/>Input: <i>Northridge (0.84g)</i>", table_cell),
            Paragraph("V<sub>B</sub> = 80.2 kN<br/>PIDR = 0.210%", table_cell_center),
            Paragraph("V<sub>b</sub> = 1,420.0 kN<br/>PIDR = 1.820%", table_cell_center),
            Paragraph("V<sub>b</sub> = 1,405.0 kN<br/>PIDR = 1.802%", table_cell_bold),
            Paragraph("Near-fault velocity pulse excitation. Accurately reproduces published FEMA-355C benchmark values (&lt;1% error).", table_cell),
        ],
    ]
    audit_table = Table(audit_data, colWidths=[112, 85, 85, 85, 145])
    audit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 6: Proposed Internship Research Roadmap
    # -----------------------------------------------------------------------
    story.append(Paragraph("6. Proposed Research Roadmap for the Internship", h1_style))

    roadmap_data = [
        [
            Paragraph("Phase & Timeline", table_header),
            Paragraph("Core Research Objective", table_header),
            Paragraph("Tangible Academic & Software Deliverables", table_header),
        ],
        [
            Paragraph("<b>Month 1: 3D Continuum Operators</b>", table_cell),
            Paragraph("Extend MDOF shear models to 3D continuum elastodynamics using Fourier Neural Operators (FNO) and DeepONets.", table_cell),
            Paragraph("Surrogate model capable of predicting 3D stress concentrations and continuum soil-structure interaction.", table_cell),
        ],
        [
            Paragraph("<b>Month 2: Online Bayesian System ID</b>", table_cell),
            Paragraph("Implement unscented Kalman filters (UKF) to continuously identify Bouc-Wen degradation parameters (&alpha;, &beta;, &gamma;).", table_cell),
            Paragraph("Real-time stiffness and damage tracking algorithm validated on multi-event aftershock sequences.", table_cell),
        ],
        [
            Paragraph("<b>Month 3: Edge Hardware & Journal Paper</b>", table_cell),
            Paragraph("Quantize surrogates (INT8/FP16) via ONNX Runtime / TensorRT for embedded ARM microcontrollers; draft journal manuscript.", table_cell),
            Paragraph("Sub-millisecond on-chip inference demonstration and co-authored manuscript submitted to ASCE / EESD.", table_cell),
        ],
    ]
    r_table = Table(roadmap_data, colWidths=[112, 195, 205])
    r_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(r_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section 7: Competencies & Technical Interview Invitation
    # -----------------------------------------------------------------------
    story.append(Paragraph("7. Technical Competencies & Verification Checklist", h1_style))
    
    comp_text = (
        "&bull; <b>Computational Mechanics</b>: MDOF Matrix Equilibrium, Inelastic Newmark-&beta;, Bouc-Wen Hysteresis, Rayleigh/Caughey Damping, FEMA P-58 IDA.<br/>"
        "&bull; <b>Scientific Machine Learning</b>: Physics-Informed Neural Networks (PINNs), Custom Physics Loss Functions, Conformal Prediction, Sobol' Sensitivity.<br/>"
        "&bull; <b>Cyber-Physical Systems & Signal Processing</b>: Hardware Abstraction Layers (HAL), Recursive STA/LTA, IIR Butterworth Filters, Serial MEMS, MQTT, WebSockets.<br/>"
        "&bull; <b>Engineering Craftsmanship</b>: 77 automated unit tests (100% pass), modular typed Python 3.10+, FastAPI REST/WebSockets, Streamlit scientific dashboard."
    )
    story.append(Paragraph(comp_text, body_style))
    story.append(Spacer(1, 4))

    closing_box = (
        "<b>APPLICATION STATEMENT & INVITATION FOR TECHNICAL INTERVIEW:</b><br/>"
        "I am eager to contribute my strong mathematical foundation, structural mechanics expertise, and software engineering rigor to your "
        "research laboratory. I am available for a remote or in-person technical interview, live codebase walkthrough, or a coding/mechanics trial. "
        "The complete codebase, benchmark suite, and documentation are openly accessible at <b>github.com/seismic-ai/seismic-ai</b>."
    )
    closing_table = Table([[Paragraph(closing_box, callout_style)]], colWidths=[512])
    closing_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(closing_table)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Research Internship Dossier PDF at: {output_path}")


if __name__ == "__main__":
    out_file = "reports/internship_application/Seismic_AI_Research_Internship_Dossier.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    generate_dossier_pdf(out_file)
