# Eason v2 shell -> v3.1 sanitized data hub full fix

Use this package to keep the old repository/link shell while replacing the old v2 code with the v3.1 sanitized data hub.

Replace these files in the GitHub repository:

- config.py
- requirements.txt
- scripts/build_report.py
- scripts/build_latest_summary.py
- .github/workflows/main.yml
- docs/index.html

Important checks:

- scripts/build_report.py must import only `TICKERS, START_DATE` from config.
- It must NOT import `PORTFOLIO`.
- The workflow must set `PYTHONPATH: ${{ github.workspace }}` under the Build market report step.
- The repo must have the GitHub Actions secret `TIINGO_API_KEY`.

After upload/commit, run:
Actions -> Eason Quant Daily -> Run workflow.
