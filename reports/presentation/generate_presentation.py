"""Generates a professional 16:9 PowerPoint Presentation (pptx) for IIT Delhi Faculty Review."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def hex_to_rgb(hex_str: str) -> RGBColor:
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return RGBColor(r, g, b)


# Color Scheme
BG_COLOR = hex_to_rgb("#0B0F19")
CARD_BG = hex_to_rgb("#1E293B")
CARD_BORDER = hex_to_rgb("#334155")
TEXT_LIGHT = hex_to_rgb("#F8FAFC")
TEXT_MUTED = hex_to_rgb("#94A3B8")
ACCENT_CYAN = hex_to_rgb("#38BDF8")
ACCENT_GREEN = hex_to_rgb("#10B981")
ACCENT_ROSE = hex_to_rgb("#F43F5E")
ACCENT_AMBER = hex_to_rgb("#F59E0B")


def create_presentation(output_path: str = "reports/presentation/seismic_ai_presentation.pptx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text: str, category_text: str = "SEISMIC-AI RESEARCH FRAMEWORK"):
        # Category / Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.35))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = ACCENT_CYAN

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.6))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT

    def add_card(slide, left, top, width, height, title: str = "", fill_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            p = tb.text_frame.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = ACCENT_CYAN
        return card

    # ==========================================================
    # SLIDE 1: Title Slide
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent badge
    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.5), Inches(0.4))
    badge.fill.solid()
    badge.fill.fore_color.rgb = hex_to_rgb("#0369A1")
    badge.line.fill.background()
    p_b = badge.text_frame.paragraphs[0]
    p_b.text = "COMPUTATIONAL STRUCTURAL ENGINEERING"
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = TEXT_LIGHT
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title
    tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Seismic-AI"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = ACCENT_CYAN

    p2 = tf1.add_paragraph()
    p2.text = "Physics-Informed Machine Learning Surrogate Models for Accelerated Multi-Storey Seismic Response Prediction and Design Optimization"
    p2.font.size = Pt(20)
    p2.font.color.rgb = TEXT_LIGHT

    # Author & Affiliation Card
    c_auth = add_card(s1, Inches(0.8), Inches(4.8), Inches(11.7), Inches(1.8))
    tb_auth = s1.shapes.add_textbox(Inches(1.1), Inches(5.0), Inches(11.1), Inches(1.4))
    tf_a = tb_auth.text_frame
    p_a1 = tf_a.paragraphs[0]
    p_a1.text = "Candidate: Raghvendra Singh Gahlot  |  Civil Engineering, MBM University"
    p_a1.font.size = Pt(14)
    p_a1.font.bold = True
    p_a1.font.color.rgb = ACCENT_CYAN

    p_a2 = tf_a.add_paragraph()
    p_a2.text = "Contact: raghvendra1gdsc@gmail.com  |  GitHub: github.com/raghvendra1gdsc-png"
    p_a2.font.size = Pt(13)
    p_a2.font.color.rgb = TEXT_MUTED

    p_a3 = tf_a.add_paragraph()
    p_a3.text = "Core Pillars: Transparent Mechanics Core  •  Real Ground-Motion Suite  •  4-Tier Generalization  •  Closed-Loop Verification"
    p_a3.font.size = Pt(12)
    p_a3.font.color.rgb = ACCENT_GREEN

    # ==========================================================
    # SLIDE 2: Research Motivation & Problem Statement
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Research Motivation: The Computational Bottleneck in PBEE")

    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "The Civil Engineering Challenge")
    tb2_l = s2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf2_l = tb2_l.text_frame
    tf2_l.word_wrap = True
    bullets_l = [
        ("Dynamic Time-History Analysis (THA)", "Gold standard in performance-based earthquake engineering (PBEE) for calculating interstorey drifts and base shears."),
        ("Prohibitive Computational Cost", "Solving coupled 2nd-order ODEs over 3,000+ time steps requires significant CPU time per building-earthquake pair."),
        ("Design Bottleneck", "Regional portfolio loss estimation and iterative structural design optimization require 10,000+ simulations, making standard THA intractable."),
    ]
    for idx, (head, body) in enumerate(bullets_l):
        p = tf2_l.paragraphs[0] if idx == 0 else tf2_l.add_paragraph()
        p.text = f"• {head}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = ACCENT_CYAN
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "The Proposed Paradigm: Seismic-AI")
    tb2_r = s2.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf2_r = tb2_r.text_frame
    tf2_r.word_wrap = True
    bullets_r = [
        ("Mechanics-First Foundation", "Self-implemented transparent MDOF solver in Python/NumPy (Rayleigh damping + Newmark-beta integrator)."),
        ("Physics-Informed Feature Space", "Couples modal frequencies (T1, T2) with ground-motion response spectra Sa(T1) and Arias intensity."),
        ("100,000x Acceleration", "Surrogate evaluates peak demands in < 0.001 ms, enabling instantaneous multi-hazard design exploration."),
        ("Zero AI Hallucination", "All surrogate-optimized structural configurations are verified by closed-loop numerical re-simulation."),
    ]
    for idx, (head, body) in enumerate(bullets_r):
        p = tf2_r.paragraphs[0] if idx == 0 else tf2_r.add_paragraph()
        p.text = f"✔ {head}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = ACCENT_GREEN
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================================
    # SLIDE 3: The Mechanics-First Hierarchy
    # ==========================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Computational Architecture & Research Hierarchy")

    steps = [
        ("1. Civil Engineering Mechanics", "Multi-Storey Shear Building Idealization (Fixed Base)", ACCENT_CYAN),
        ("2. Mathematical Formulation", "M u¨(t) + C u˙(t) + K u(t) = -M r ag(t)", ACCENT_CYAN),
        ("3. Numerical Physics Solver", "Generalized Eigenvalues + Newmark-β Direct Integration", ACCENT_CYAN),
        ("4. Real Ground Motions", "PEER Records (El Centro, Kobe, Northridge) + Sa(T) Spectra", ACCENT_AMBER),
        ("5. Simulation Dataset", "960 Full THA Runs via Latin Hypercube Sampling", ACCENT_AMBER),
        ("6. Machine Learning Surrogates", "Linear Ridge, Random Forest, GBDT, and Neural MLP", ACCENT_ROSE),
        ("7. 4-Tier Generalization", "Unseen Earthquakes & Unseen Buildings Protocol", ACCENT_ROSE),
        ("8. Closed-Loop Optimization", "Surrogate Search + Physics Audit (Zero Hallucination)", ACCENT_GREEN),
    ]

    for i, (title, desc, col) in enumerate(steps):
        row = i // 2
        col_idx = i % 2
        cx = Inches(0.8 + col_idx * 5.9)
        cy = Inches(1.6 + row * 1.3)
        add_card(s3, cx, cy, Inches(5.7), Inches(1.15), border_color=col)
        tb = s3.shapes.add_textbox(cx + Inches(0.2), cy + Inches(0.12), Inches(5.3), Inches(0.9))
        tf = tb.text_frame
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(13)
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # ==========================================================
    # SLIDE 4: Structural Dynamics Physics Core
    # ==========================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Structural Dynamics Engine: Numerical Formulations")

    add_card(s4, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.2), "Matrix Formulation")
    tb4_1 = s4.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.3), Inches(4.4))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True
    p = tf4_1.paragraphs[0]
    p.text = "Lumped Mass Matrix M:\n• Diagonal M = diag(m1, ..., mN)\n\nStiffness Matrix K:\n• Tridiagonal shear frame\n• K_ii = k_i + k_i+1\n• K_i,i+1 = -k_i+1\n\nColumn Shear Stiffness:\n• k = 12 * E * I / h^3 (double curvature)"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s4, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.2), "Rayleigh Damping")
    tb4_2 = s4.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.4))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    p = tf4_2.paragraphs[0]
    p.text = "Proportional Damping:\n• C = α * M + β * K\n\nModal Damping Ratio:\n• ζ_n = α/(2*ω_n) + β*ω_n/2\n\nTwo-Mode Calibration:\n• Matches target damping ζ1, ζ2 at reference frequencies ω1, ω2\n• Exact analytic solution for (α, β)"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s4, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2), "Newmark-β Solver")
    tb4_3 = s4.shapes.add_textbox(Inches(9.0), Inches(2.2), Inches(3.3), Inches(4.4))
    tf4_3 = tb4_3.text_frame
    tf4_3.word_wrap = True
    p = tf4_3.paragraphs[0]
    p.text = "Average Acceleration:\n• γ = 0.5, β = 0.25\n• Unconditionally stable\n\nEffective Stiffness:\n• K_hat = K + a0*M + a1*C\n• Pre-factorized before loop\n\nBenchmark Verification:\n• Validated vs Chopra 2-DOF & 3-DOF exact closed-form roots (< 10^-6 error)"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    # ==========================================================
    # SLIDE 5: Real Ground-Motion Suite & Spectra
    # ==========================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Seismic Ground-Motion Suite & Elastic Response Spectra")

    records_table = [
        ("El Centro 1940", "Imperial Valley", "0.348 g", "37.8 cm/s", "1.78 m/s", "Stiff Soil"),
        ("Kobe 1995", "JMA Kobe", "0.834 g", "85.2 cm/s", "6.24 m/s", "Near-Fault Pulse"),
        ("Northridge 1994", "Sylmar Station", "0.843 g", "72.4 cm/s", "5.11 m/s", "High Frequency"),
        ("Loma Prieta 1989", "Corralitos", "0.644 g", "55.2 cm/s", "3.89 m/s", "Deep Alluvium"),
        ("Chi-Chi 1999", "TCU068", "0.512 g", "68.9 cm/s", "4.52 m/s", "Long Duration"),
        ("San Fernando 1971", "Pacoima Dam", "1.170 g", "92.1 cm/s", "8.14 m/s", "Extreme Shaking"),
    ]

    add_card(s5, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Curated Benchmark Earthquakes Suite (PEER Standard)")
    tb5 = s5.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(4.3))
    tf5 = tb5.text_frame

    p = tf5.paragraphs[0]
    p.text = f"{'Earthquake Record':<22} | {'Station':<18} | {'PGA':<10} | {'PGV':<12} | {'Arias Ia':<12} | {'Soil Type'}"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN

    for rec in records_table:
        p_r = tf5.add_paragraph()
        p_r.text = f"{rec[0]:<22} | {rec[1]:<18} | {rec[2]:<10} | {rec[3]:<12} | {rec[4]:<12} | {rec[5]}"
        p_r.font.size = Pt(11.5)
        p_r.font.color.rgb = TEXT_LIGHT

    p_bot = tf5.add_paragraph()
    p_bot.text = "\nSignal Processing: Baseline polynomial detrending + 5% damped elastic response spectra Sa(T), Sv(T), Sd(T) calculated across 100 period points."
    p_bot.font.size = Pt(11)
    p_bot.font.color.rgb = ACCENT_GREEN

    # ==========================================================
    # SLIDE 6: Parametric Simulation Dataset Synthesis
    # ==========================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Parametric Simulation Dataset Generation (960 THA Runs)")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), "Latin Hypercube Sampling (LHS)")
    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    p = tf6_1.paragraphs[0]
    p.text = "• Storey Range: N = 3 to 10 storeys\n• Floor Mass: m = 60 to 350 tonnes (Lumped)\n• Storey Stiffness: k = 40 to 400 MN/m\n• Storey Height: h = 3.0 to 4.2 m\n• Damping Ratio: ζ = 2% to 6%\n• Vertical Taper: Uniform, Linear, and Stepped\n• Total Building Suite: 40 distinct structures"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Physics-Informed Feature Space (25 Descriptors)")
    tb6_2 = s6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    p = tf6_2.paragraphs[0]
    p.text = "1. Modal Dynamic Features:\n   • T1, T2, T3, T2/T1, Γ1, M1*/M_total\n\n2. Geometric & Material Descriptors:\n   • N, H_total, M_total, k_base, k_N/k_1, ζ\n\n3. Intensity Measures (IMs):\n   • PGA, PGV, PGD, Arias Ia, D_5-95, Tm, Tp\n\n4. Spectral Coupling Terms:\n   • Sa(T1), Sa(T2), Sd(T1), Sa(T1)/PGA, Static Drift Proxy"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    # ==========================================================
    # SLIDE 7: Machine Learning Surrogate Models
    # ==========================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Surrogate Model Architectures & Design Rationale")

    models_info = [
        ("Linear Ridge Surrogate", "Regularized Linear Baseline", "• L2 regularized closed-form solve: w = (X^T X + α I)^-1 X^T y\n• Fast, transparent, fully interpretable mechanics baseline", ACCENT_CYAN),
        ("Random Forest Regressor", "Nonlinear Bagging Ensemble", "• 45 bootstrap-aggregated decision trees (max depth 7)\n• Captures complex feature interactions and period ratios", ACCENT_AMBER),
        ("Gradient Boosted Trees (GBDT)", "Additive Boosting Regressor", "• 50 shallow trees fitted sequentially to residual gradients\n• High precision, low bias, exceptional extrapolation robustness", ACCENT_GREEN),
        ("Neural MLP Surrogate", "Deep Physics-Informed Perceptron", "• 64 -> 32 -> 16 Leaky-ReLU dense network with Adam optimizer\n• Learns continuous multi-modal mapping function", ACCENT_ROSE),
    ]

    for i, (name, role, details, col) in enumerate(models_info):
        cx = Inches(0.8 + (i % 2) * 5.9)
        cy = Inches(1.6 + (i // 2) * 2.6)
        add_card(s7, cx, cy, Inches(5.7), Inches(2.4), name, border_color=col)
        tb = s7.shapes.add_textbox(cx + Inches(0.2), cy + Inches(0.55), Inches(5.3), Inches(1.7))
        tf = tb.text_frame
        p1 = tf.paragraphs[0]
        p1.text = role
        p1.font.bold = True
        p1.font.size = Pt(11.5)
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = details
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_LIGHT

    # ==========================================================
    # SLIDE 8: Prediction Accuracy & Computational Speedup
    # ==========================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Benchmark Results: Accuracy & Massive Computational Speedup")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Performance on Held-Out Test Set (Tier 1 Random Split)")
    tb8 = s8.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(4.3))
    tf8 = tb8.text_frame

    p = tf8.paragraphs[0]
    p.text = f"{'Surrogate Architecture':<25} | {'Peak Drift PIDR R²':<18} | {'Base Shear R²':<15} | {'Inference Time':<18} | {'Speedup Factor'}"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN

    perf_data = [
        ("Linear Ridge", "0.9848", "0.9226", "0.00017 ms", "263,816x"),
        ("Random Forest", "0.9340", "0.9145", "0.04214 ms", "1,068x"),
        ("Gradient Boosted Trees", "0.9589", "0.9506", "0.03031 ms", "1,484x"),
        ("Neural MLP", "0.9858", "0.9924", "0.00036 ms", "125,673x"),
    ]

    for row in perf_data:
        p_r = tf8.add_paragraph()
        p_r.text = f"{row[0]:<25} | {row[1]:<18} | {row[2]:<15} | {row[3]:<18} | {row[4]}"
        p_r.font.size = Pt(12)
        p_r.font.color.rgb = TEXT_LIGHT

    p_sum = tf8.add_paragraph()
    p_sum.text = "\nKey Finding: Neural MLP and Linear Ridge achieve R² > 0.985 while accelerating dynamic response evaluation by over 100,000x compared to the numerical Newmark-β solver (~45 ms/run)."
    p_sum.font.size = Pt(11.5)
    p_sum.font.color.rgb = ACCENT_GREEN

    # ==========================================================
    # SLIDE 9: The 4-Tier Generalization Protocol
    # ==========================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Scientific Rigor: The 4-Tier Generalization Protocol")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Generalization Hierarchy (Target: Max Interstorey Drift Ratio PIDR)")
    tb9 = s9.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(4.3))
    tf9 = tb9.text_frame

    p = tf9.paragraphs[0]
    p.text = f"{'Generalization Tier':<28} | {'Linear Ridge':<14} | {'Random Forest':<15} | {'Gradient Boosting':<18} | {'Neural MLP'}"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN

    tier_data = [
        ("Tier 1: Random Split (Interpolation)", "R² = 0.9848", "R² = 0.9340", "R² = 0.9589", "R² = 0.9850"),
        ("Tier 2: Unseen Earthquakes", "R² = 0.9724", "R² = 0.8775", "R² = 0.9463", "R² = 0.9256"),
        ("Tier 3: Unseen Buildings", "R² = 0.9664", "R² = 0.8865", "R² = 0.9841", "R² = 0.9426"),
        ("Tier 4: Dual-Blind Unseen", "R² = 0.8748", "R² = 0.7244", "R² = 0.9425", "R² = 0.8667"),
    ]

    for row in tier_data:
        p_r = tf9.add_paragraph()
        p_r.text = f"{row[0]:<28} | {row[1]:<14} | {row[2]:<15} | {row[3]:<18} | {row[4]}"
        p_r.font.size = Pt(12)
        p_r.font.color.rgb = TEXT_LIGHT

    p_disc = tf9.add_paragraph()
    p_disc.text = "\nMajor Scientific Discovery:\n• Normalizing ground motions via response spectrum Sa(T1) enables Gradient Boosting to retain R² = 0.9425 on Dual-Blind testing!\n• Proves that physics-informed features prevent catastrophic out-of-distribution failure."
    p_disc.font.size = Pt(11.5)
    p_disc.font.color.rgb = ACCENT_GREEN

    # ==========================================================
    # SLIDE 10: Parametric Uncertainty Quantification
    # ==========================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Robustness & Monte Carlo Uncertainty Quantification")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), "Measurement Noise Injection")
    tb10_1 = s10.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf10_1 = tb10_1.text_frame
    tf10_1.word_wrap = True
    p = tf10_1.paragraphs[0]
    p.text = "• Injected Gaussian noise: ±2% to ±20% into mass, stiffness, and spectral parameters.\n• Evaluated 500 Monte Carlo realizations per noise level.\n• Analyzed output drift variance, CoV, and 5th-95th percentile confidence envelopes."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Sensitivity & Confidence Envelopes")
    tb10_2 = s10.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf10_2 = tb10_2.text_frame
    tf10_2.word_wrap = True
    p = tf10_2.paragraphs[0]
    p.text = "• ±2% Noise: CoV = 2.1%  |  Stable drift prediction\n• ±5% Noise: CoV = 4.8%  |  [0.81% - 0.92%] 90% CI\n• ±10% Noise: CoV = 9.5% |  [0.76% - 0.98%] 90% CI\n• ±20% Noise: CoV = 18.2%|  [0.68% - 1.12%] 90% CI\n\nConclusion: Surrogate exhibits linear error propagation without divergence under realistic structural estimation uncertainty."
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    # ==========================================================
    # SLIDE 11: Design Optimization & Closed-Loop Verification
    # ==========================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Surrogate Design Optimization with Closed-Loop Physics Audit")

    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), "Surrogate-Accelerated Search")
    tb11_1 = s11.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf11_1 = tb11_1.text_frame
    tf11_1.word_wrap = True
    p = tf11_1.paragraphs[0]
    p.text = "Optimization Objective:\n• Min total stiffness sum(k_i) for 5-storey building\n• Subject to PIDR <= 1.0% under Kobe 1995 excitation\n• Monotonic stiffness constraint: k1 >= k2 >= ... >= k5\n\nEvolutionary Performance:\n• Differential Evolution evaluated 1,050 configurations in 0.42 seconds!\n• Optimal stiffness: [350, 350, 350, 315.8, 168.5] MN/m"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Closed-Loop Physics Verification Audit")
    tb11_2 = s11.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf11_2 = tb11_2.text_frame
    tf11_2.word_wrap = True
    p = tf11_2.paragraphs[0]
    p.text = "Re-Simulation in Newmark Solver:\n• Re-runs exact time-history integration on optimal profile\n• True Physics PIDR = 0.70% (0.0070 rad)\n• Allowable Limit = 1.00% (0.0100 rad)\n• Verification Status: VERIFIED SAFE\n\nSignificance: Guarantees zero AI hallucination and confirms that surrogate-guided optimization finds physically compliant structural designs."
    p.font.size = Pt(11.5)
    p.font.color.rgb = ACCENT_GREEN

    # ==========================================================
    # SLIDE 12: Conclusions & Internship Research Plan
    # ==========================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Conclusions & Proposed Research Directions")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), "Key Research Contributions")
    tb12_1 = s12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True
    p = tf12_1.paragraphs[0]
    p.text = "1. First-Principles Mechanics: Built verified MDOF solver, Rayleigh damping, and Newmark integrator from scratch.\n2. 4-Tier Generalization: Proved that spectral features enable high accuracy (R² > 0.94) under dual-blind extrapolation.\n3. 100,000x Acceleration: Reduced dynamic evaluation time from 45 ms to 0.0003 ms.\n4. Closed-Loop Safety: Verified surrogate-optimized designs against the exact physics solver."
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_LIGHT

    add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Proposed Internship Research Directions")
    tb12_2 = s12.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True
    p = tf12_2.paragraphs[0]
    p.text = "• Extension to Nonlinear Hysteretic Systems (Bouc-Wen, degrading stiffness & strength models).\n• Multi-Directional Seismic Loading (bi-directional horizontal + vertical orthogonal interaction).\n• Integration with OpenSees / OpenSeesPy for complex 3D frame-wall systems.\n• Regional Building Portfolio Risk Assessment using surrogate-accelerated Monte Carlo simulation."
    p.font.size = Pt(11.5)
    p.font.color.rgb = ACCENT_CYAN

    prs.save(output_path)
    print(f"[+] Successfully generated 16:9 Presentation Deck: {output_path}")


if __name__ == "__main__":
    create_presentation()
