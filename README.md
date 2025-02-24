# Website Critic

An AI-powered tool that analyzes websites and provides UX/UI improvement suggestions.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/website-critic.git
cd website-critic
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

## Usage

### 1. Scrape and Analyze Websites

```bash
python -m src.scrape
```

This will:

- Capture screenshots of target websites
- Segment and analyze each section
- Store analyses in vector database

### 2. Interactive Analysis

```bash
python -m src.chat
```

Commands:

- Type `report` to generate comprehensive analysis
- Type `quit` to exit
- Ask any question about the analyzed websites

## Project Structure

```
website_critic/
├── src/
│   ├── scrape.py      # Website data collection
│   ├── chat.py        # Interactive analysis
│   └── analysis/      # Analysis modules
├── reports/           # Generated reports
└── raw_segments/      # Processed segments
```
