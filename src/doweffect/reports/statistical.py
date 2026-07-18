import pandas as pd
import os

def generate_statistical_report(results_df: pd.DataFrame, output_path: str = "outputs/reports/statistical_report.html"):
    """
    Generates a simple HTML report from the hypothesis test results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    html_content = f"""
    <html>
    <head>
        <title>Phase 2: Statistical Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Phase 2: Statistical Analysis Report</h1>
        <p>This report contains the output of the pure econometric falsification tests.</p>
        
        <h2>Falsification Results (Decision Tree)</h2>
        {results_df.to_html(index=False)}
        
        <p><em>Note: If the final verdict is not 'Econometrically Robust Anomaly', the hypothesis is rejected.</em></p>
    </body>
    </html>
    """
    
    with open(output_path, "w") as f:
        f.write(html_content)
