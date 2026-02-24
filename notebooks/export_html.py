"""Export the analysis notebook to HTML for sharing."""

import subprocess
import sys

def export():
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'html',
        '--no-input',  # hide code cells in output
        'notebooks/kan_analysis.ipynb',
        '--output-dir', 'docs/',
        '--output', 'kan_analysis_report.html',
    ], check=True)
    print("Exported to docs/kan_analysis_report.html")

if __name__ == '__main__':
    export()
