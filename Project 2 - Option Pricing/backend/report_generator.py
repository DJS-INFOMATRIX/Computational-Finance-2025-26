from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import datetime

class ReportGenerator:
    def __init__(self):
        pass

    def generate_strategy_report(self, strategy_name: str, legs: list, greeks: dict, simulation_results: list = None):
        """
        Generates a PDF report for the strategy.
        Returns bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph(f"Glass Box Option Strategist Report: {strategy_name}", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Date
        date_text = Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_text)
        elements.append(Spacer(1, 24))

        # Strategy Composition
        elements.append(Paragraph("Strategy Composition", styles['Heading2']))
        data = [["Side", "Type", "Strike (K)", "Contracts"]]
        for leg in legs:
            data.append([leg['side'].upper(), leg['type'].upper(), f"${leg['K']}", str(leg['quantity'])])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

        # Risk Profile
        elements.append(Paragraph("Risk Profile (Greeks)", styles['Heading2']))
        greeks_data = [
            ["Metric", "Value", "Explanation"],
            ["Price", f"${greeks.get('price', 0):.2f}", "Net Cost/Credit"],
            ["Delta", f"{greeks.get('delta', 0):.2f}", "Directional Risk"],
            ["Theta", f"{greeks.get('theta', 0):.2f}", "Daily Time Decay"],
            ["Vega", f"{greeks.get('vega', 0):.2f}", "Volatility Sensitivity"],
        ]
        t2 = Table(greeks_data, colWidths=[60, 80, 200])
        t2.setStyle(TableStyle([
             ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
             ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
             ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 24))

        # Transparent Verification (Trust Section)
        elements.append(Paragraph("Trust & Verification", styles['Heading2']))
        
        # 1. Formula Check
        formula_text = """
        <b>Black-Scholes Formula Used:</b><br/>
        C = S N(d1) - K e^(-rT) N(d2)<br/>
        P = K e^(-rT) N(-d2) - S N(-d1)<br/>
        Where d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma sqrt(T))<br/>
        """
        elements.append(Paragraph(formula_text, styles['Normal']))
        elements.append(Spacer(1, 12))

        # 2. Benchmark Table (Simulated)
        bench_data = [
             ["Model", "Price", "Difference"],
             ["Glass Box BS", f"${greeks.get('price', 0):.2f}", "-"],
             ["Yahoo Finance (Est)", "N/A", "Data Unavailable"] # Placeholder logic
        ]
        t3 = Table(bench_data)
        t3.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
             ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ]))
        elements.append(t3)
        elements.append(Spacer(1, 24))

        # Simulation Summary
        if simulation_results:
            elements.append(Paragraph("Historical Validation (Last 30 Days)", styles['Heading2']))
            final_pnl = simulation_results[-1]['pnl']
            min_pnl = min(item['pnl'] for item in simulation_results)
            max_pnl = max(item['pnl'] for item in simulation_results)
            
            sim_text = f"""
            Backtest Ticker: Provided Ticker<br/>
            Final PnL: ${final_pnl:.2f}<br/>
            Best Day: ${max_pnl:.2f}<br/>
            Worst Day: ${min_pnl:.2f}
            """
            elements.append(Paragraph(sim_text, styles['Normal']))
            elements.append(Spacer(1, 24))

        # Disclaimer
        elements.append(Paragraph("DISCLAIMER", styles['Heading2']))
        disclaimer = """
        This report is for educational purposes only. Option trading involves significant risk and is not suitable for all investors. 
        Glass Box Option Strategist is a simulation tool and does not provide financial advice. 
        Calculations are theoretical based on Black-Scholes model.
        """
        elements.append(Paragraph(disclaimer, styles['Normal']))

        # Build
        doc.build(elements)
        buffer.seek(0)
        return buffer.read()
