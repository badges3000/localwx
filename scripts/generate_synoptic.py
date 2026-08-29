name: DWD ICON Synoptic Charts Pipeline

on:
  schedule:
    # 4x täglich nach Fertigstellung der DWD ICON-EU Hauptläufe (00, 06, 12, 18 UTC)
    - cron: '45 3,9,15,21 * * *'
  workflow_dispatch:

concurrency:
  group: synoptic-pipeline
  cancel-in-progress: true

jobs:
  generate-synoptic-charts:
    runs-on: ubuntu-latest
    timeout-minutes: 25

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install System Libraries
        run: |
          sudo apt-get update -o Acquire::ForceIPv4=true -y
          sudo apt-get install -y --no-install-recommends libgeos-dev libproj-dev libeccodes0 bzip2
          sudo ln -sf /usr/lib/x86_64-linux-gnu/libeccodes.so.0 /usr/lib/x86_64-linux-gnu/libeccodes.so 2>/dev/null || true

      - name: Install Python Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pillow scipy matplotlib cartopy eccodes

      - name: Run DWD Synoptic Charts Generator & Uploader
        env:
          FTP_SERVER: ${{ secrets.FTP_SERVER }}
          FTP_USERNAME: ${{ secrets.FTP_USERNAME }}
          FTP_PASSWORD: ${{ secrets.FTP_PASSWORD }}
        run: |
          echo "📂 Prüfe Arbeitsverzeichnis..."
          pwd
          ls -la
          
          if [ -f "scripts/generate_synoptic.py" ]; then
            echo "🚀 Starte scripts/generate_synoptic.py..."
            python scripts/generate_synoptic.py
          elif [ -f "weather-app/scripts/generate_synoptic.py" ]; then
            echo "🚀 Starte weather-app/scripts/generate_synoptic.py..."
            python weather-app/scripts/generate_synoptic.py
          elif [ -f "generate_synoptic.py" ]; then
            echo "🚀 Starte generate_synoptic.py..."
            python generate_synoptic.py
          else
            echo "❌ FEHLER: generate_synoptic.py wurde nicht im Repository gefunden!"
            echo "Dateistruktur im Runner:"
            find . -maxdepth 3 -name "*.py"
            exit 1
          fi
