"""
Generates an 8-slide, professional PowerPoint presentation for the
Stock Market Risk Prediction System college project review.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Color Palette (Dark Financial / Dashboard Aesthetic)
BG_DARK = RGBColor(9, 13, 26)          # #090D1A - Deep Navy/Black
CARD_BG = RGBColor(17, 24, 39)         # #111827 - Slate Surface
CARD_ALT_BG = RGBColor(22, 31, 51)     # #161F33 - Slightly lighter card
CARD_BORDER = RGBColor(45, 58, 84)     # #2D3A54 - Slate Blue Outline

ACCENT_BLUE = RGBColor(56, 189, 248)   # #38BDF8 - Electric Cyan / Sky Blue
ACCENT_GREEN = RGBColor(16, 185, 129)  # #10B981 - Bullish / Emerald Green
ACCENT_AMBER = RGBColor(245, 158, 11)  # #F59E0B - Amber / Warning
ACCENT_RED = RGBColor(239, 68, 68)     # #EF4444 - Drawdown / Crimson
ACCENT_PURPLE = RGBColor(168, 85, 247) # #A855F7 - Evolutionary Purple
ACCENT_TEAL = RGBColor(20, 184, 166)   # #14B8A6 - Teal
TEXT_WHITE = RGBColor(248, 250, 252)   # #F8FAFC - Primary Text
TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8 - Secondary Muted
TEXT_DIM = RGBColor(100, 116, 139)     # #64748B - Dark Muted

FONT_MAIN = "Calibri"
FONT_HEADING = "Calibri"

def set_slide_background(slide):
    """Fills slide background with dark navy/black."""
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_DARK
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, eyebrow_text, title_text, subtitle_text=None, eyebrow_color=ACCENT_BLUE):
    """Adds a standard slide header with eyebrow, main title, and optional subtitle."""
    box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.733), Inches(1.3))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    # Eyebrow
    p_eye = tf.paragraphs[0]
    p_eye.text = eyebrow_text.upper()
    p_eye.font.name = FONT_HEADING
    p_eye.font.size = Pt(11)
    p_eye.font.bold = True
    p_eye.font.color.rgb = eyebrow_color
    p_eye.space_after = Pt(4)
    
    # Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    
    # Subtitle
    if subtitle_text:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_MAIN
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.space_before = Pt(4)

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER, border_width=1):
    """Draws a rounded rectangular surface card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(border_width)
    else:
        card.line.fill.background()
    return card

def add_pill(slide, left, top, width, height, text, bg_color, text_color=TEXT_WHITE, font_size=11):
    """Draws an eye-catching pill tag."""
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    pill.fill.solid()
    pill.fill.fore_color.rgb = bg_color
    pill.line.fill.background()
    tf = pill.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = text_color
    return pill

# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_slide_1(prs):
    """Slide 1 – Title Slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    # Decorative Top Accent Glow Line
    glow = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    glow.fill.solid()
    glow.fill.fore_color.rgb = ACCENT_BLUE
    glow.line.fill.background()

    # Central Hero Card Container
    hero = add_card(slide, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1), bg_color=CARD_BG, border_color=CARD_BORDER, border_width=1.5)
    
    # Pill tags on top of hero card
    add_pill(slide, Inches(2.2), Inches(1.7), Inches(2.4), Inches(0.4), "ACADEMIC CAPSTONE", RGBColor(30, 41, 59), ACCENT_BLUE, font_size=11)
    add_pill(slide, Inches(4.8), Inches(1.7), Inches(2.3), Inches(0.4), "FACULTY REVIEW", RGBColor(16, 50, 40), ACCENT_GREEN, font_size=11)
    add_pill(slide, Inches(7.3), Inches(1.7), Inches(2.7), Inches(0.4), "HYBRID AI / FINTECH", RGBColor(50, 30, 60), ACCENT_PURPLE, font_size=11)
    
    # Title & Subtitle Box
    t_box = slide.shapes.add_textbox(Inches(2.2), Inches(2.4), Inches(8.933), Inches(2.3))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Stock Market Risk Prediction System"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.space_after = Pt(10)
    
    p2 = tf.add_paragraph()
    p2.text = "A Hybrid AI-Based Approach Combining Soft Computing & Deep Learning"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(17)
    p2.font.color.rgb = ACCENT_BLUE
    p2.space_after = Pt(14)
    
    p3 = tf.add_paragraph()
    p3.text = "An intelligent, explainable framework predicting financial market stress, drawdown potential, and risk levels using BiLSTM, Neuro-Fuzzy logic, and Evolutionary algorithms."
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED

    # Bottom Metadata Strip inside Hero Card
    meta_box = slide.shapes.add_textbox(Inches(2.2), Inches(5.0), Inches(8.933), Inches(0.9))
    tf_m = meta_box.text_frame
    tf_m.word_wrap = True
    
    pm1 = tf_m.paragraphs[0]
    pm1.text = "Domain: Machine Learning & Quantitative Finance   |   Focus: Capital Protection & Risk Assessment"
    pm1.font.name = FONT_MAIN
    pm1.font.size = Pt(12)
    pm1.font.color.rgb = TEXT_DIM


def build_slide_2(prs):
    """Slide 2 – What Does Our Project Do?"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Project Objective",
        title_text="What Does Our Project Do?",
        subtitle_text="It analyses historical stock market data to measure and explain market risk — not to guess exact stock prices."
    )
    
    # Crucial Disclaimer Banner
    banner = add_card(slide, Inches(0.8), Inches(1.85), Inches(11.733), Inches(0.85), bg_color=CARD_ALT_BG, border_color=ACCENT_BLUE, border_width=1)
    b_box = slide.shapes.add_textbox(Inches(1.1), Inches(1.95), Inches(11.133), Inches(0.65))
    tf_b = b_box.text_frame
    tf_b.word_wrap = True
    pb = tf_b.paragraphs[0]
    pb.text = "Key Philosophy: Predicting exact stock prices is notoriously noisy and unreliable. Our system focuses on what matters most for capital preservation: Market Risk, Volatility Regimes, and Downside Protection."
    pb.font.name = FONT_MAIN
    pb.font.size = Pt(13)
    pb.font.color.rgb = TEXT_WHITE
    
    # Three Output Cards
    card_w = Inches(3.68)
    card_h = Inches(3.9)
    gap = Inches(0.34)
    start_x = Inches(0.8)
    start_y = Inches(2.95)
    
    # 1. Risk Score Card
    x1 = start_x
    c1 = add_card(slide, x1, start_y, card_w, card_h, border_color=ACCENT_BLUE)
    add_pill(slide, x1 + Inches(0.3), start_y + Inches(0.3), Inches(1.4), Inches(0.35), "OUTPUT 1", RGBColor(14, 50, 75), ACCENT_BLUE)
    
    tb1 = slide.shapes.add_textbox(x1 + Inches(0.3), start_y + Inches(0.85), card_w - Inches(0.6), card_h - Inches(1.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Risk Score\n(0 – 100)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(12)
    
    p = tf1.add_paragraph()
    p.text = "• Continuous risk rating from 0 (very safe) to 100 (extreme danger)."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)
    
    p = tf1.add_paragraph()
    p.text = "• Summarises multiple market threats into a single intuitive metric."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)

    p = tf1.add_paragraph()
    p.text = "• Visualised on the dashboard as a real-time risk gauge speedometer."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED

    # 2. Risk Category Card
    x2 = start_x + card_w + gap
    c2 = add_card(slide, x2, start_y, card_w, card_h, border_color=ACCENT_GREEN)
    add_pill(slide, x2 + Inches(0.3), start_y + Inches(0.3), Inches(1.4), Inches(0.35), "OUTPUT 2", RGBColor(16, 50, 40), ACCENT_GREEN)
    
    tb2 = slide.shapes.add_textbox(x2 + Inches(0.3), start_y + Inches(0.85), card_w - Inches(0.6), card_h - Inches(1.0))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    
    p = tf2.paragraphs[0]
    p.text = "Risk Category\n(4 Levels)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(12)
    
    p = tf2.add_paragraph()
    p.text = "• Four distinct investment bands:\n  - Low (0–25)\n  - Moderate (25–50)\n  - High (50–75)\n  - Extreme (75–100)"
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)
    
    p = tf2.add_paragraph()
    p.text = "• Helps investors and fund managers make immediate portfolio decisions."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED

    # 3. 10-Day Drawdown Card
    x3 = x2 + card_w + gap
    c3 = add_card(slide, x3, start_y, card_w, card_h, border_color=ACCENT_RED)
    add_pill(slide, x3 + Inches(0.3), start_y + Inches(0.3), Inches(1.4), Inches(0.35), "OUTPUT 3", RGBColor(60, 20, 25), ACCENT_RED)
    
    tb3 = slide.shapes.add_textbox(x3 + Inches(0.3), start_y + Inches(0.85), card_w - Inches(0.6), card_h - Inches(1.0))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    
    p = tf3.paragraphs[0]
    p.text = "10-Day Drawdown\n(Downside Drop)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(12)
    
    p = tf3.add_paragraph()
    p.text = "• Predicts the maximum expected percentage fall over the next 10 days."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)
    
    p = tf3.add_paragraph()
    p.text = "• Example: 'Expect a potential peak-to-trough drop of 8.5%'."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)

    p = tf3.add_paragraph()
    p.text = "• Acts as an early warning shield against sudden market crashes."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED


def build_slide_3(prs):
    """Slide 3 – How Does It Work? (End-to-End Pipeline)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Working Workflow",
        title_text="How Does It Work?",
        subtitle_text="A systematic 6-step pipeline from raw stock prices to clear risk insights."
    )
    
    # Flowchart step data
    steps = [
        ("1. Historical Data", "Daily OHLCV prices from Yahoo & Google Finance (US & Indian stocks).", ACCENT_BLUE),
        ("2. Processing", "Timezone alignment, missing-bar imputation & uniform cleaning.", ACCENT_TEAL),
        ("3. Features", "Extraction of Sharpe, Max Drawdown, Intraday IV & Historical Volatility.", ACCENT_GREEN),
        ("4. AI Modeling", "BiLSTM, Neuro-Fuzzy, GA tuning & GP symbolic discovery.", ACCENT_PURPLE),
        ("5. Risk Prediction", "Multi-head generation of Risk Score, Category & 10-day Drawdown.", ACCENT_AMBER),
        ("6. Dashboard", "Interactive visual dashboard with gauges, charts & rule explanations.", ACCENT_RED),
    ]
    
    step_w = Inches(1.78)
    step_h = Inches(3.6)
    gap = Inches(0.20)
    start_x = Inches(0.8)
    y_pos = Inches(2.2)
    
    for idx, (s_title, s_desc, color) in enumerate(steps):
        x = start_x + idx * (step_w + gap)
        # Main Step Card
        add_card(slide, x, y_pos, step_w, step_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        # Step Number Badge
        add_pill(slide, x + Inches(0.2), y_pos + Inches(0.25), step_w - Inches(0.4), Inches(0.35), f"STEP 0{idx+1}", color, TEXT_WHITE, font_size=10)
        
        # Text in step
        tb = slide.shapes.add_textbox(x + Inches(0.15), y_pos + Inches(0.75), step_w - Inches(0.3), step_h - Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = s_title.split(". ")[1]
        p.font.name = FONT_HEADING
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(10)
        
        p2 = tf.add_paragraph()
        p2.text = s_desc
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_MUTED
        
        # Connecting arrow (except last step)
        if idx < len(steps) - 1:
            arr_x = x + step_w + Inches(0.04)
            arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arr_x, y_pos + Inches(1.6), Inches(0.12), Inches(0.25))
            arr.fill.solid()
            arr.fill.fore_color.rgb = TEXT_DIM
            arr.line.fill.background()

    # Bottom summary card
    summary_card = add_card(slide, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.9), bg_color=CARD_ALT_BG, border_color=CARD_BORDER)
    s_tb = slide.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.333), Inches(0.8))
    s_tf = s_tb.text_frame
    s_tf.word_wrap = True
    p = s_tf.paragraphs[0]
    p.text = "Key Advantage: Entire pipeline runs automatically when a stock ticker is selected. Pre-trained weights ensure predictions render in under 1 second without re-training."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_BLUE


def build_slide_4(prs):
    """Slide 4 – Data Used (The 4 Financial Indicators)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Feature Engineering",
        title_text="Data Used — Four Core Financial Indicators",
        subtitle_text="We extract 4 specialized metrics from raw prices so the AI model evaluates risk from multiple angles."
    )
    
    indicators = [
        (
            "Sharpe Ratio",
            "Return Compared with Risk",
            "• Measures how much excess return a stock delivers for every unit of volatility taken.\n• High Sharpe = Efficient, healthy performance.\n• Low/Negative Sharpe = High risk, poor compensation for losses.",
            ACCENT_GREEN,
            "RISK-ADJUSTED RETURN"
        ),
        (
            "Maximum Drawdown",
            "Largest Historical Fall",
            "• Measures the worst peak-to-trough drop over the evaluation window.\n• Answers: 'What is the absolute worst loss an investor experienced?'\n• Critical indicator of extreme stress and downside vulnerability.",
            ACCENT_RED,
            "WORST-CASE DROP"
        ),
        (
            "Garman–Klass Volatility",
            "Intraday Price Movement",
            "• Uses Open, High, Low, and Close prices rather than just the Close.\n• Captures turbulent swings that happen during the trading day.\n• Up to 8x more informative than standard daily closing variance.",
            ACCENT_AMBER,
            "INTRADAY TURBULENCE"
        ),
        (
            "Historical Volatility",
            "Price Fluctuation Over Time",
            "• Annualized standard deviation of daily log price returns.\n• Measures how rapidly and unpredictably prices bounce around.\n• Distinguishes calm, stable stocks from volatile, unpredictable ones.",
            ACCENT_BLUE,
            "SWING INTENSITY"
        ),
    ]
    
    grid_w = Inches(5.7)
    grid_h = Inches(2.25)
    gap_x = Inches(0.33)
    gap_y = Inches(0.25)
    
    positions = [
        (Inches(0.8), Inches(1.9)),
        (Inches(0.8) + grid_w + gap_x, Inches(1.9)),
        (Inches(0.8), Inches(1.9) + grid_h + gap_y),
        (Inches(0.8) + grid_w + gap_x, Inches(1.9) + grid_h + gap_y),
    ]
    
    for (name, tag, details, color, badge), (x, y) in zip(indicators, positions):
        add_card(slide, x, y, grid_w, grid_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        # Badge
        add_pill(slide, x + Inches(0.25), y + Inches(0.2), Inches(2.4), Inches(0.3), badge, color, TEXT_WHITE, font_size=10)
        
        tb = slide.shapes.add_textbox(x + Inches(0.25), y + Inches(0.55), grid_w - Inches(0.5), grid_h - Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f"{name} — {tag}"
        p.font.name = FONT_HEADING
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)
        
        p2 = tf.add_paragraph()
        p2.text = details
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_MUTED

    # Bottom note
    b_card = add_card(slide, Inches(0.8), Inches(6.65), Inches(11.733), Inches(0.5), bg_color=CARD_ALT_BG, border_color=None)
    b_tb = slide.shapes.add_textbox(Inches(1.0), Inches(6.65), Inches(11.333), Inches(0.45))
    b_tf = b_tb.text_frame
    b_p = b_tf.paragraphs[0]
    b_p.text = "All 4 indicators are extracted in real time from public market data — no expensive institutional data feeds required."
    b_p.font.name = FONT_MAIN
    b_p.font.size = Pt(12)
    b_p.font.color.rgb = TEXT_MUTED


def build_slide_5(prs):
    """Slide 5 – Main Components (The 4 AI Techniques)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Artificial Intelligence Engine",
        title_text="Main Components — Four Integrated AI Techniques",
        subtitle_text="Each technique solves a specific limitation of traditional risk models."
    )
    
    components = [
        (
            "BiLSTM Neural Network",
            "Understands Patterns Over Time",
            "• Reads 20-day historical sequences in both forward and reverse directions.\n• Understands how today's price action is contextualised by past volatility buildups.\n• Outputs a rich 128-dimensional temporal market summary.",
            ACCENT_BLUE,
            "TEMPORAL MEMORY"
        ),
        (
            "TSK Neuro-Fuzzy System",
            "Transparent, Flexible Rules",
            "• Evaluates market risk using 16 human-readable IF-THEN linguistic rules.\n• Unlike black-box models, explains exactly *why* a stock is risky.\n• Uses smooth Gaussian curves to handle market uncertainty gracefully.",
            ACCENT_GREEN,
            "EXPLAINABLE RULES"
        ),
        (
            "Genetic Algorithm (GA)",
            "Optimises Fuzzy Settings",
            "• Inspired by natural selection: evolves solutions using crossover & mutation.\n• Tunes 24 Gaussian membership centers and spreads to their ideal values.\n• Prevents the fuzzy system from getting stuck in poor manual guesses.",
            ACCENT_AMBER,
            "EVOLUTIONARY TUNING"
        ),
        (
            "Genetic Programming (GP)",
            "Discovers Mathematical Formulas",
            "• Automatically discovers new algebraic formulas from raw stock features.\n• Uncovers non-linear relationships that human quantitative analysts might miss.\n• Adds an independent symbolic signal to verify neural network findings.",
            ACCENT_PURPLE,
            "SYMBOLIC DISCOVERY"
        ),
    ]
    
    card_w = Inches(2.78)
    card_h = Inches(4.7)
    gap = Inches(0.20)
    start_x = Inches(0.8)
    start_y = Inches(1.95)
    
    for idx, (title, tag, body, color, badge) in enumerate(components):
        x = start_x + idx * (card_w + gap)
        add_card(slide, x, start_y, card_w, card_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        add_pill(slide, x + Inches(0.2), start_y + Inches(0.25), card_w - Inches(0.4), Inches(0.35), badge, color, TEXT_WHITE, font_size=10)
        
        tb = slide.shapes.add_textbox(x + Inches(0.2), start_y + Inches(0.75), card_w - Inches(0.4), card_h - Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(2)
        
        p_sub = tf.add_paragraph()
        p_sub.text = tag
        p_sub.font.name = FONT_MAIN
        p_sub.font.size = Pt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = color
        p_sub.space_after = Pt(12)
        
        p_body = tf.add_paragraph()
        p_body.text = body
        p_body.font.name = FONT_MAIN
        p_body.font.size = Pt(12)
        p_body.font.color.rgb = TEXT_MUTED


def build_slide_6(prs):
    """Slide 6 – System Architecture (How All Components Work Together)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Technical Architecture",
        title_text="System Architecture — How Components Work Together",
        subtitle_text="A multi-branch fusion network combining temporal, fuzzy, dense, and symbolic features into one unified decision."
    )
    
    # 1. Left: Input Data Box
    in_card = add_card(slide, Inches(0.8), Inches(2.0), Inches(2.2), Inches(4.6), bg_color=CARD_BG, border_color=ACCENT_BLUE, border_width=1.5)
    add_pill(slide, Inches(0.95), Inches(2.2), Inches(1.9), Inches(0.32), "INPUT FEATURES", RGBColor(14, 50, 75), ACCENT_BLUE, font_size=10)
    
    tb_in = slide.shapes.add_textbox(Inches(0.95), Inches(2.7), Inches(1.9), Inches(3.7))
    tf_in = tb_in.text_frame
    tf_in.word_wrap = True
    p = tf_in.paragraphs[0]
    p.text = "Historical OHLCV\nPrices\n\n1. Sharpe Ratio\n2. Max Drawdown\n3. Garman-Klass IV\n4. Historical Vol\n\nSupplied as:\n• 20-Day Sequence\n• Static Metric Vector"
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    # 2. Middle: 4 AI Branches
    b_x = Inches(3.4)
    b_w = Inches(3.4)
    b_h = Inches(1.0)
    gap_y = Inches(0.2)
    start_y = Inches(2.0)
    
    branches = [
        ("BiLSTM Temporal Branch", "128 Dimensions • 20-day sequential memory", ACCENT_BLUE),
        ("TSK Neuro-Fuzzy Branch", "1 Dimension • 16 transparent Larsen rules", ACCENT_GREEN),
        ("Dense Feature Branch", "32 Dimensions • Static non-linear metric map", ACCENT_TEAL),
        ("GP Symbolic Branch", "1 Dimension • Evolved algebraic formula", ACCENT_PURPLE),
    ]
    
    for i, (title, desc, color) in enumerate(branches):
        y = start_y + i * (b_h + gap_y)
        add_card(slide, b_x, y, b_w, b_h, bg_color=CARD_BG, border_color=color, border_width=1.2)
        
        tb = slide.shapes.add_textbox(b_x + Inches(0.2), y + Inches(0.12), b_w - Inches(0.4), b_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(11)
        p2.font.color.rgb = color

    # Special callout for GA tuning TSK
    ga_pill = add_card(slide, Inches(3.4), Inches(6.8), Inches(3.4), Inches(0.45), bg_color=CARD_ALT_BG, border_color=ACCENT_AMBER)
    ga_tb = slide.shapes.add_textbox(Inches(3.5), Inches(6.8), Inches(3.2), Inches(0.4))
    ga_tf = ga_tb.text_frame
    ga_p = ga_tf.paragraphs[0]
    ga_p.text = "⚡ GA tunes 24 Gaussian membership settings for TSK"
    ga_p.font.name = FONT_MAIN
    ga_p.font.size = Pt(11)
    ga_p.font.color.rgb = ACCENT_AMBER

    # 3. Middle-Right: Combined Fusion Layer
    f_x = Inches(7.2)
    add_card(slide, f_x, Inches(2.0), Inches(2.4), Inches(4.6), bg_color=CARD_BG, border_color=ACCENT_AMBER, border_width=1.5)
    add_pill(slide, f_x + Inches(0.2), Inches(2.2), Inches(2.0), Inches(0.32), "NEURAL FUSION", RGBColor(50, 35, 10), ACCENT_AMBER, font_size=10)
    
    f_tb = slide.shapes.add_textbox(f_x + Inches(0.2), Inches(2.7), Inches(2.0), Inches(3.7))
    f_tf = f_tb.text_frame
    f_tf.word_wrap = True
    p = f_tf.paragraphs[0]
    p.text = "162-D Combined Vector:\n[128 + 1 + 32 + 1]\n\nFusion Layers:\n• Linear(162 → 64)\n• ReLU + Dropout(0.2)\n• Linear(64 → 32)\n• ReLU\n\nCombines temporal, fuzzy & mathematical signals."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    # 4. Right: Three Final Predictions
    o_x = Inches(10.0)
    o_w = Inches(2.533)
    o_h = Inches(1.35)
    o_gap = Inches(0.27)
    
    outputs = [
        ("Risk Score Head", "0 to 100 Rating", "Continuous numerical score of overall stress.", ACCENT_BLUE),
        ("Classification Head", "4 Risk Categories", "Categorises into Low, Moderate, High, Extreme.", ACCENT_GREEN),
        ("Drawdown Head", "10-Day Downside %", "Expected peak-to-trough price fall potential.", ACCENT_RED),
    ]
    
    for i, (head, sub, desc, color) in enumerate(outputs):
        y = start_y + i * (o_h + o_gap)
        add_card(slide, o_x, y, o_w, o_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        tb = slide.shapes.add_textbox(o_x + Inches(0.2), y + Inches(0.12), o_w - Inches(0.4), o_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = head
        p.font.name = FONT_HEADING
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = color
        
        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.name = FONT_MAIN
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED


def build_slide_7(prs):
    """Slide 7 – Results and Findings (Actual Project Numbers)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Empirical Evaluation",
        title_text="Results and Findings — Proven Project Performance",
        subtitle_text="Evaluated on 648 out-of-sample test trading days without data leakage."
    )
    
    # 4 Big Stat Metric Cards
    metrics = [
        ("6.1945", "Risk Score MAE", "Mean Absolute Error on 0–100 scale (±6 points average error)", ACCENT_BLUE),
        ("0.7477", "R² Score", "74.8% of variance in stock risk explained by our hybrid model", ACCENT_GREEN),
        ("68.98%", "Accuracy", "Correct categorization into Low, Moderate, High, or Extreme risk", ACCENT_PURPLE),
        ("648", "Test Observations", "Strictly chronological held-out test sessions (zero data leakage)", ACCENT_AMBER),
    ]
    
    m_w = Inches(2.78)
    m_h = Inches(1.8)
    m_gap = Inches(0.20)
    start_x = Inches(0.8)
    start_y = Inches(1.9)
    
    for idx, (val, title, desc, color) in enumerate(metrics):
        x = start_x + idx * (m_w + m_gap)
        add_card(slide, x, start_y, m_w, m_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        tb = slide.shapes.add_textbox(x + Inches(0.2), start_y + Inches(0.15), m_w - Inches(0.4), m_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_val = tf.paragraphs[0]
        p_val.text = val
        p_val.font.name = FONT_HEADING
        p_val.font.size = Pt(36)
        p_val.font.bold = True
        p_val.font.color.rgb = color
        p_val.space_after = Pt(2)
        
        p_title = tf.add_paragraph()
        p_title.text = title
        p_title.font.name = FONT_MAIN
        p_title.font.size = Pt(13)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE
        p_title.space_after = Pt(4)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.name = FONT_MAIN
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = TEXT_MUTED

    # Qualitative Findings Cards (Two Columns below metrics)
    b_y = Inches(3.95)
    b_h = Inches(2.8)
    b_w = Inches(5.76)
    
    # Left Box: Key Strengths
    left_card = add_card(slide, start_x, b_y, b_w, b_h, bg_color=CARD_BG, border_color=ACCENT_GREEN)
    add_pill(slide, start_x + Inches(0.3), b_y + Inches(0.25), Inches(2.6), Inches(0.35), "KEY STRENGTHS", RGBColor(16, 50, 40), ACCENT_GREEN)
    
    tb_l = slide.shapes.add_textbox(start_x + Inches(0.3), b_y + Inches(0.75), b_w - Inches(0.6), b_h - Inches(0.9))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "• Strongest Component: BiLSTM"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(2)
    
    p = tf_l.add_paragraph()
    p.text = "  Ablation tests proved removing BiLSTM caused the biggest drop in accuracy (~12% loss). Temporal sequence memory is essential."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)
    
    p = tf_l.add_paragraph()
    p.text = "• Neuro-Fuzzy Transparency"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(2)
    
    p = tf_l.add_paragraph()
    p.text = "  TSK system achieved complete interpretability with zero loss in predictive accuracy, proving that AI does not have to be an opaque black box."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    # Right Box: Honest Limitations & Future Improvements
    right_x = start_x + b_w + Inches(0.2)
    right_card = add_card(slide, right_x, b_y, b_w, b_h, bg_color=CARD_BG, border_color=ACCENT_AMBER)
    add_pill(slide, right_x + Inches(0.3), b_y + Inches(0.25), Inches(2.8), Inches(0.35), "AREAS FOR IMPROVEMENT", RGBColor(50, 35, 10), ACCENT_AMBER)
    
    tb_r = slide.shapes.add_textbox(right_x + Inches(0.3), b_y + Inches(0.75), b_w - Inches(0.6), b_h - Inches(0.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "• GP Contribution Was Mixed"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(2)
    
    p = tf_r.add_paragraph()
    p.text = "  Discovered symbolic formulas contributed valuable non-linear signals, but required careful z-score scaling to prevent constant DC bias."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(8)
    
    p = tf_r.add_paragraph()
    p.text = "• Drawdown Forecasting Needs Refinement"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(2)
    
    p = tf_r.add_paragraph()
    p.text = "  Predicting exact 10-day forward maximum drawdown is inherently challenging due to extreme tail-risk skewness. Quantile regression is planned next."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED


def build_slide_8(prs):
    """Slide 8 – Conclusion & Final Summary"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    
    add_header(
        slide,
        eyebrow_text="Project Wrap-Up",
        title_text="Conclusion — Intelligent & Explainable Risk Assessment",
        subtitle_text="A unified system delivering actionable market risk intelligence for investors and fund managers."
    )
    
    # Linear Process Banner: DATA -> ANALYSIS -> PREDICTION -> RISK INSIGHTS
    p_card = add_card(slide, Inches(0.8), Inches(1.85), Inches(11.733), Inches(1.1), bg_color=CARD_ALT_BG, border_color=CARD_BORDER)
    
    blocks = ["1. HISTORICAL DATA", "2. MULTI-AI ANALYSIS", "3. HYBRID PREDICTION", "4. ACTIONABLE INSIGHTS"]
    colors = [ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_AMBER]
    b_width = Inches(2.4)
    b_start_x = Inches(1.1)
    
    for i, (b_text, b_col) in enumerate(zip(blocks, colors)):
        x = b_start_x + i * Inches(2.8)
        pill = add_pill(slide, x, Inches(2.15), b_width, Inches(0.5), b_text, b_col, TEXT_WHITE, font_size=12)
        if i < len(blocks) - 1:
            arr_x = x + b_width + Inches(0.12)
            arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arr_x, Inches(2.28), Inches(0.16), Inches(0.24))
            arr.fill.solid()
            arr.fill.fore_color.rgb = TEXT_DIM
            arr.line.fill.background()

    # Three Main Takeaways Columns
    c_w = Inches(3.68)
    c_h = Inches(3.65)
    c_gap = Inches(0.34)
    start_x = Inches(0.8)
    start_y = Inches(3.2)
    
    takeaways = [
        (
            "Explainable AI",
            "Demystifies the Black Box",
            "• Unlike standard neural networks that give unexplained numbers, our TSK fuzzy system shows which rules fired and why.\n• Essential for regulatory compliance, risk committees, and financial auditing.\n• Builds trust with non-technical investors.",
            ACCENT_GREEN,
            "AUDITABLE"
        ),
        (
            "Multi-Market Ready",
            "US & Indian NSE Benchmarks",
            "• Supports major US equities (Apple, Microsoft, Tesla, Nvidia) and Indian NSE stocks (Reliance, TCS, HDFC Bank, SBI).\n• Real-time currency switching (₹ vs $) and live market quote validation.\n• Fully responsive web dashboard.",
            ACCENT_BLUE,
            "PRACTICAL"
        ),
        (
            "Scientifically Validated",
            "Rigorous Testing & Architecture",
            "• 287 passing automated unit and integration tests.\n• Verified through 6-way ablation experiments, proving each component's distinct value.\n• Clean separation of training, validation, and held-out test splits.",
            ACCENT_PURPLE,
            "ROBUST"
        ),
    ]
    
    for i, (title, sub, body, color, badge) in enumerate(takeaways):
        x = start_x + i * (c_w + c_gap)
        add_card(slide, x, start_y, c_w, c_h, bg_color=CARD_BG, border_color=color, border_width=1.5)
        
        add_pill(slide, x + Inches(0.25), start_y + Inches(0.25), Inches(2.0), Inches(0.32), badge, color, TEXT_WHITE, font_size=10)
        
        tb = slide.shapes.add_textbox(x + Inches(0.25), start_y + Inches(0.7), c_w - Inches(0.5), c_h - Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(2)
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = FONT_MAIN
        p_sub.font.size = Pt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = color
        p_sub.space_after = Pt(10)
        
        p_body = tf.add_paragraph()
        p_body.text = body
        p_body.font.name = FONT_MAIN
        p_body.font.size = Pt(12)
        p_body.font.color.rgb = TEXT_MUTED


# ==============================================================================
# MAIN GENERATOR
# ==============================================================================

def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    print("Building Slide 1: Title...")
    build_slide_1(prs)
    
    print("Building Slide 2: What Does Our Project Do?...")
    build_slide_2(prs)
    
    print("Building Slide 3: How Does It Work?...")
    build_slide_3(prs)
    
    print("Building Slide 4: Data Used...")
    build_slide_4(prs)
    
    print("Building Slide 5: Main Components...")
    build_slide_5(prs)
    
    print("Building Slide 6: System Architecture...")
    build_slide_6(prs)
    
    print("Building Slide 7: Results and Findings...")
    build_slide_7(prs)
    
    print("Building Slide 8: Conclusion...")
    build_slide_8(prs)
    
    output_path = Path("/Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/Stock_Market_Risk_Prediction_Presentation.pptx")
    prs.save(str(output_path))
    print(f"\nPresentation successfully generated at:\n{output_path}")

if __name__ == "__main__":
    main()
