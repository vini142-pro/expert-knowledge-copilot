# Industrial Knowledge Copilot — ET AI Hackathon 2026 (Problem #8)

## Setup (2 minutes)

```bash
cd knowledge-copilot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Usage

1. Paste your OpenAI API key in the sidebar
2. Upload 1-3 sample PDFs (safety SOP, maintenance manual, inspection report — even dummy/sample ones are fine for demo)
3. Click "Build Knowledge Base"
4. Ask questions in the chat box — answers come with source citations (document + page number)


## Where to get sample docs fast

- Search "sample safety SOP PDF" / "equipment maintenance manual PDF" on Google
- Or use any public factory safety guideline PDF (OSHA publishes free ones)
- Even 2-3 page dummy PDFs work — judges care about the pipeline working, not real industrial data
