# app.py
from flask import Flask, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from word2number import w2n
from num2words import num2words
import json
from datetime import datetime
import re
import os
app = Flask(__name__)


# --------------------- CONFIG ---------------------
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
EXCEL_FILE = "Fully_Cleaned_Titles_Final.xlsx"
SIMILARITY_DUPLICATE_THRESHOLD = 70.0
TOP_N = 5

# --------------------- LOAD MODEL ---------------------
print("Loading sentence-transformers model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# --------------------- NORMALIZATION FUNCTION ---------------------
def normalize_title(title):
    if not isinstance(title, str):
        title = str(title)
    # basic cleaning
    title = title.strip()
    # replace multiple spaces
    title = re.sub(r'\s+', ' ', title)
    words = title.lower().split()
    normalized = []
    for word in words:
        # remove punctuation around word
        w = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', word)
        if not w:
            continue
        try:
            # try convert word spelled-out numbers to digits (e.g., "two" -> "2")
            normalized.append(str(w2n.word_to_num(w)))
        except Exception:
            try:
                # if token is digits already, convert to words (to unify representation)
                if w.isdigit():
                    normalized.append(num2words(int(w)).replace(" ", ""))
                else:
                    normalized.append(w)
            except Exception:
                normalized.append(w)
    return ' '.join(normalized)

# --------------------- TITLE ANALYSIS ---------------------
def analyze_title_characteristics(title):
    """Analyze various characteristics of the title"""
    if not isinstance(title, str):
        title = str(title)
    analysis = {
        'length': len(title),
        'word_count': len(title.split()),
        'has_numbers': bool(re.search(r'\d', title)),
        'has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', title)),
        'starts_with_the': title.lower().startswith('the '),
        'contains_news': 'news' in title.lower(),
        'contains_times': 'times' in title.lower(),
        'contains_daily': 'daily' in title.lower(),
        'capitalized_words': sum(1 for word in title.split() if word and word[0].isupper())
    }
    return analysis

# --------------------- LOAD DATA & PRECOMPUTE EMBEDDINGS ---------------------
def load_titles():
    if not os.path.exists(EXCEL_FILE):
        print(f"Warning: Excel file '{EXCEL_FILE}' not found. Starting with empty DB.")
        empty_df = pd.DataFrame(columns=["Title_Name_in_Lower", "normalized_title"])
        return empty_df, np.array([])
    try:
        df = pd.read_excel(EXCEL_FILE)
        # ensure the expected column exists
        if "Title_Name_in_Lower" not in df.columns:
            # try to create from some possible alternatives
            if "Title" in df.columns:
                df["Title_Name_in_Lower"] = df["Title"].astype(str).str.lower()
            else:
                df["Title_Name_in_Lower"] = df.iloc[:, 0].astype(str).str.lower()
        df["normalized_title"] = df["Title_Name_in_Lower"].astype(str).apply(normalize_title)
        print("Computing embeddings for", len(df), "titles. This may take a moment...")
        embeddings = model.encode(df["normalized_title"].tolist(), show_progress_bar=False)
        embeddings = np.array(embeddings)
        print("Embeddings ready.")
        return df, embeddings
    except Exception as e:
        print("Error loading Excel or computing embeddings:", e)
        return pd.DataFrame(columns=["Title_Name_in_Lower", "normalized_title"]), np.array([])

df, df_embeddings = load_titles()

# --------------------- CATEGORIZE SIMILARITIES ---------------------
def categorize_similarities(similarities):
    """
    similarities: array of scores in percent (0-100)
    """
    categories = {
        "Exact/Near Match (90-100%)": 0,
        "Highly Similar (80-89%)": 0,
        "Moderately Similar (60-79%)": 0,
        "Somewhat Similar (40-59%)": 0,
        "Low Similarity (20-39%)": 0,
        "Not Similar (<20%)": 0
    }
    detailed_matches = {k: [] for k in categories.keys()}

    for i, score in enumerate(similarities):
        # guard index
        title = df.iloc[i]["Title_Name_in_Lower"] if i < len(df) else ""
        match_info = {"title": title, "score": float(round(score, 2))}
        if score >= 90:
            categories["Exact/Near Match (90-100%)"] += 1
            detailed_matches["Exact/Near Match (90-100%)"].append(match_info)
        elif score >= 80:
            categories["Highly Similar (80-89%)"] += 1
            detailed_matches["Highly Similar (80-89%)"].append(match_info)
        elif score >= 60:
            categories["Moderately Similar (60-79%)"] += 1
            detailed_matches["Moderately Similar (60-79%)"].append(match_info)
        elif score >= 40:
            categories["Somewhat Similar (40-59%)"] += 1
            detailed_matches["Somewhat Similar (40-59%)"].append(match_info)
        elif score >= 20:
            categories["Low Similarity (20-39%)"] += 1
            detailed_matches["Low Similarity (20-39%)"].append(match_info)
        else:
            categories["Not Similar (<20%)"] += 1
            detailed_matches["Not Similar (<20%)"].append(match_info)
    return categories, detailed_matches

# --------------------- GENERATE SUGGESTIONS ---------------------
def generate_title_suggestions(input_title, top_matches):
    base = input_title.strip()

    raw_candidates = []

    descriptors = ["Herald", "Gazette", "Chronicle", "Post", "Journal", "Observer"]
    time_words = ["Morning", "Evening", "Weekly"]
    semantic_variants = [
        "Voice of " + base,
        base.replace("Daily", "").strip() + " Bulletin",
        "Independent " + base,
        base.replace("Times", "").strip() + " Express",
        base + " Network",
        "The " + base
    ]

    for d in descriptors:
        raw_candidates.append(f"{base} {d}")

    for t in time_words:
        raw_candidates.append(f"{t} {base}")

    raw_candidates.extend(semantic_variants)

    # 🔥 FILTER USING YOUR OWN MODEL
    unique_suggestions = []
    for title in raw_candidates:
        if is_title_unique_candidate(title):
            unique_suggestions.append(title)
        if len(unique_suggestions) == 3:
            break

    return unique_suggestions


# --------------------- SIMILARITY CHECK ---------------------
def find_similar_titles(input_title, top_n=TOP_N):
    # if no database loaded
    if df.empty or df_embeddings.size == 0:
        return [], True, {}, {}, [], {}

    norm_input = normalize_title(input_title)
    # compute embedding for input
    input_embedding = model.encode([norm_input])[0].reshape(1, -1)
    # compute cosine similarity: returns values in [-1, 1]. Convert to percent [0,100]
    similarities = cosine_similarity(input_embedding, df_embeddings)[0]
    similarities_percent = (similarities * 100).astype(float)

    # pick top N indices
    top_indices = similarities_percent.argsort()[::-1][:top_n]
    top_matches = [
        {"title": df.iloc[i]["Title_Name_in_Lower"], "score": float(round(similarities_percent[i], 2))}
        for i in top_indices
    ]

    # Title is considered unique only if ALL top matches are strictly below duplicate threshold
    is_unique = all(match["score"] < SIMILARITY_DUPLICATE_THRESHOLD for match in top_matches)

    categories, detailed_matches = categorize_similarities(similarities_percent)
    suggestions = generate_title_suggestions(input_title, top_matches) if not is_unique else []
    analysis = analyze_title_characteristics(input_title)

    return top_matches, is_unique, categories, detailed_matches, suggestions, analysis

def is_title_unique_candidate(title):
    if df.empty or df_embeddings.size == 0:
        return True

    norm = normalize_title(title)
    emb = model.encode([norm])[0].reshape(1, -1)
    sims = cosine_similarity(emb, df_embeddings)[0] * 100
    return np.max(sims) < SIMILARITY_DUPLICATE_THRESHOLD


# --------------------- DATABASE STATS ---------------------
def get_database_stats():
    if df.empty:
        return {'total_titles': 0, 'avg_length': 0, 'avg_words': 0}
    return {
        'total_titles': len(df),
        'avg_length': round(df['Title_Name_in_Lower'].str.len().mean(), 1),
        'avg_words': round(df['Title_Name_in_Lower'].str.split().str.len().mean(), 1)
    }

# --------------------- GENERATE HTML RESULTS (DARK THEME) ---------------------
def generate_results_html(result):
    if not result:
        return ""

    status_icon = '✓' if result['is_unique'] else '⚠'
    status_class = 'unique' if result['is_unique'] else 'duplicate'
    status_text = 'Unique Title - Added to Database' if result['is_unique'] else 'Similar Titles Found - Review Suggestions'

    # Title analysis
    analysis = result['analysis']
    analysis_html = f'''
    <div class="title-analysis">
        <h3>📊 Title Characteristics</h3>
        <div class="analysis-grid">
            <div class="analysis-item">
                <div class="analysis-value">{analysis['length']}</div>
                <div class="analysis-label">Characters</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">{analysis['word_count']}</div>
                <div class="analysis-label">Words</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">{'✓' if analysis['starts_with_the'] else '✗'}</div>
                <div class="analysis-label">Starts with "The"</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">{analysis['capitalized_words']}</div>
                <div class="analysis-label">Capitalized Words</div>
            </div>
        </div>
    </div>
    '''

    # Suggestions
    suggestions_html = ""
    if result['suggestions']:
        suggestion_items = ""
        for sugg in result['suggestions']:
            safe_sugg = sugg.replace("'", "\\'")
            suggestion_items += f'<div class="suggestion-item" onclick="copySuggestion(\'{safe_sugg}\')">{sugg} <span class="copy-hint">📋</span></div>'
        suggestions_html = f'''
        <div class="suggestions-box">
            <h3>💡 Alternative Title Suggestions</h3>
            <p>Based on similarity analysis, consider these alternatives:</p>
            {suggestion_items}
        </div>
        '''

    # Top matches
    matches_html = ""
    for match in result['top_matches']:
        matches_html += f'''
        <div class="match-card">
            <div class="match-title">{match['title']}</div>
            <div class="match-score">{match['score']}%</div>
        </div>
        '''

    # Category breakdown
    categories_html = ""
    idx = 0
    for category, matches in result['detailed_matches'].items():
        if matches:
            idx += 1
            mini_matches = ""
            for match in matches[:20]:
                mini_matches += f'''
                <div class="mini-match">
                    <span>{match['title']}</span>
                    <span class="mini-score">{match['score']}%</span>
                </div>
                '''
            more_html = ""
            if len(matches) > 20:
                more_html = f'''
                <div class="more-note">... and {len(matches) - 20} more titles</div>
                '''
            categories_html += f'''
            <div class="category-section">
                <div class="category-header" onclick="toggleCategory('category-{idx}')">
                    <span>{category}</span>
                    <span class="category-count">{len(matches)}</span>
                </div>
                <div id="category-{idx}" class="category-content collapsed">
                    {mini_matches}
                    {more_html}
                </div>
            </div>
            '''

    categories_json = json.dumps(result['categories'])
    top_matches_json = json.dumps(result['top_matches'])

    results_html = f'''
    <div class="results">
        <div class="status-card">
            <div class="section-title">Analysis Results</div>

            <div style="margin-bottom: 20px;">
                <strong>Input Title:</strong>
                <span class="input-title">{result['input_title']}</span>
            </div>

            <div class="status-badge {status_class}">
                <span class="status-icon">{status_icon}</span>
                <span class="status-text">{status_text}</span>
            </div>

            {analysis_html}
        </div>

        {suggestions_html}

        <div class="charts-grid">
            <div class="chart-container">
                <h3>📊 Similarity Distribution</h3>
                <div class="chart-wrapper">
                    <canvas id="similarityChart"></canvas>
                </div>
            </div>

            <div class="chart-container">
                <h3>📈 Score Distribution</h3>
                <div class="chart-wrapper">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
        </div>

        <div class="matches-section">
            <div class="section-title">Top {TOP_N} Most Similar Titles</div>
            {matches_html}
        </div>

        <div class="matches-section">
            <div class="section-title">Detailed Similarity Breakdown</div>
            {categories_html}
        </div>
    </div>

    <script>
        const categories = {categories_json};
        const ctx = document.getElementById('similarityChart') && document.getElementById('similarityChart').getContext('2d');
        if (ctx) {{
            new Chart(ctx, {{
                type: 'pie',
                data: {{
                    labels: Object.keys(categories),
                    datasets: [{{
                        data: Object.values(categories),
                        backgroundColor: ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#10b981'],
                        borderWidth: 2,
                        borderColor: '#0b1220'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1' }} }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const value = context.parsed;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = total ? ((value / total) * 100).toFixed(1) : 0;
                                    return context.label + ': ' + value + ' (' + percentage + '%)';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}

        const topMatches = {top_matches_json};
        const barCtx = document.getElementById('barChart') && document.getElementById('barChart').getContext('2d');
        if (barCtx) {{
            new Chart(barCtx, {{
                type: 'bar',
                data: {{
                    labels: topMatches.map(m => m.title.length > 30 ? m.title.substring(0, 30) + '...' : m.title),
                    datasets: [{{
                        label: 'Similarity Score (%)',
                        data: topMatches.map(m => m.score),
                        backgroundColor: ['rgba(59,130,246,0.9)','rgba(99,102,241,0.9)','rgba(16,185,129,0.9)','rgba(234,179,8,0.9)','rgba(239,68,68,0.9)'],
                        borderWidth: 1,
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }},
                                color: '#cbd5e1'
                            }}
                        }},
                        x: {{
                            ticks: {{ color: '#cbd5e1' }}
                        }}
                    }}
                }}
            }});
        }}

        function toggleCategory(id) {{
            const el = document.getElementById(id);
            if (el) el.classList.toggle('collapsed');
        }}

        function copySuggestion(text) {{
            navigator.clipboard.writeText(text).then(() => {{
                alert('Suggestion copied: ' + text);
            }});
        }}

        document.getElementById('loading') && (document.getElementById('loading').style.display = 'none');
    </script>
    '''
    return results_html

# --------------------- HTML TEMPLATE (DARK MODE) ---------------------
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Advanced Title Intelligence — Dark</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<style>
    :root{
        --bg:#0b1220;
        --card:#0f1724;
        --muted:#94a3b8;
        --accent-1: linear-gradient(135deg,#3b82f6 0%,#6366f1 100%);
        --accent-2: linear-gradient(135deg,#10b981 0%,#059669 100%);
        --glass: rgba(255,255,255,0.03);
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{
        background: radial-gradient(circle at 10% 10%, #071025 0%, #060812 40%, #06090f 100%), var(--bg);
        color:#e6eef8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding:30px;
        -webkit-font-smoothing:antialiased;
    }
    .container{
        max-width:1200px;
        margin:0 auto;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.015));
        border-radius:18px;
        overflow:hidden;
        box-shadow: 0 30px 80px rgba(2,6,23,0.7);
    }
    .header{
        padding:28px 36px;
        background: linear-gradient(135deg,#0f1724 0%, #0b1220 100%);
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .header h1{font-size:1.9rem; color:#f1f5f9}
    .header p{color:var(--muted); margin-top:6px}
    .stats-bar{display:flex;gap:12px;padding:18px 28px;background:transparent;flex-wrap:wrap}
    .stat-item{flex:1; min-width:140px;background:var(--card);padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.03)}
    .stat-value{font-size:1.4rem;font-weight:700;color:#c7d2fe}
    .stat-label{color:var(--muted);font-size:0.88rem;margin-top:6px}
    .content{padding:28px}
    .search-box{background:var(--card); padding:22px;border-radius:14px;border:1px solid rgba(255,255,255,0.03)}
    label{display:block;font-weight:700;margin-bottom:8px;color:#e6eef8}
    input[type="text"]{
        width:100%; padding:14px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.04);
        background: rgba(255,255,255,0.01); color:#e6eef8; font-size:1rem;
    }
    input[type="text"]:focus{outline:none; box-shadow:0 6px 30px rgba(59,130,246,0.08); border-color:#3b82f6}
    .button-group{display:flex;gap:12px;margin-top:14px}
    button{
        padding:12px 20px;border-radius:10px;border:none;cursor:pointer;font-weight:700;
        background: linear-gradient(135deg,#3b82f6 0%,#6366f1 100%); color:white;
        box-shadow: 0 8px 30px rgba(59,130,246,0.12);
    }
    .secondary-btn{background:linear-gradient(135deg,#334155 0%,#1f2937 100%)}
    .loading{display:none;padding:20px;text-align:center}
    .spinner{width:46px;height:46px;border-radius:50%;border:4px solid rgba(255,255,255,0.06);border-top-color:#3b82f6; margin:10px auto; animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .status-card{background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); padding:18px;border-radius:12px;border:1px solid rgba(255,255,255,0.03); margin-bottom:18px}
    .status-badge{display:inline-flex;align-items:center;gap:10px;padding:8px 16px;border-radius:20px;font-weight:700}
    .unique{background:var(--accent-2); color:#061013}
    .duplicate{background:linear-gradient(135deg,#ef4444 0%,#dc2626 100%); color:white}
    .title-analysis{margin-top:14px}
    .analysis-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:12px}
    .analysis-item{background:var(--glass);padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.02);text-align:center}
    .analysis-value{font-size:1.1rem;color:#c7d2fe;font-weight:700}
    .analysis-label{color:var(--muted);font-size:0.85rem}
    .suggestions-box{background:#071129;padding:14px;border-radius:12px;margin:16px 0;border:1px solid rgba(59,130,246,0.06)}
    .suggestion-item{background:rgba(255,255,255,0.02);padding:10px;border-radius:8px;margin:8px 0;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
    .suggestion-item:hover{transform:translateX(6px)}
    .charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;margin:18px 0}
    .chart-container{background:var(--card);padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.03)}
    .chart-wrapper{height:300px;position:relative}
    .matches-section{margin-top:12px}
    .match-card{background:rgba(255,255,255,0.01);padding:12px;border-radius:10px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;border:1px solid rgba(255,255,255,0.02)}
    .match-title{font-weight:700;color:#e6eef8}
    .match-score{background:linear-gradient(135deg,#3b82f6 0%,#6366f1 100%);padding:8px 12px;border-radius:20px;font-weight:800}
    .category-section{background:rgba(255,255,255,0.01);padding:12px;border-radius:10px;border:1px solid rgba(255,255,255,0.02);margin-top:10px}
    .category-header{display:flex;justify-content:space-between;align-items:center;cursor:pointer;color:#c7d2fe;font-weight:700}
    .category-count{background:rgba(255,255,255,0.03);padding:6px 12px;border-radius:20px}
    .category-content{margin-top:10px;max-height:260px;overflow:auto}
    .category-content.collapsed{display:none}
    .mini-match{display:flex;justify-content:space-between;padding:8px;border-radius:8px;margin-bottom:8px;background:rgba(255,255,255,0.01)}
    .mini-score{font-weight:800;color:#c7d2fe}
    .section-title{font-size:1.2rem;color:#c7d2fe;margin-bottom:10px}
    .more-note{padding:10px;text-align:center;color:var(--muted);font-style:italic}
    .input-title{font-weight:700;color:#e6eef8;margin-left:8px}
    @media (max-width:720px){.stats-bar{flex-direction:column}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📰 Advanced Title Intelligence — Dark</h1>
        <p>AI-Powered Newspaper Title Verification & Semantic Similarity Checker</p>
    </div>

    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value">%%TOTAL_TITLES%%</div>
            <div class="stat-label">Total Titles</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">%%AVG_LENGTH%%</div>
            <div class="stat-label">Avg Characters</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">%%AVG_WORDS%%</div>
            <div class="stat-label">Avg Words</div>
        </div>
    </div>

    <div class="content">
        <div class="search-box">
            <form method="POST" id="titleForm">
                <label for="title">🔍 Enter Newspaper Title</label>
                <input type="text" id="title" name="title" placeholder="e.g., The Daily Tribune" required>
                <div class="button-group">
                    <button type="submit">🚀 Analyze Title</button>
                    <button type="button" class="secondary-btn" onclick="document.getElementById('titleForm').reset();">🔄 Clear</button>
                </div>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="color:#c7d2fe;font-weight:700">Analyzing title...</p>
        </div>

        %%RESULTS%%
    </div>
</div>

<script>
document.getElementById('titleForm').addEventListener('submit', function() {
    document.getElementById('loading').style.display = 'block';
});
</script>
</body>
</html>
'''

# --------------------- ROUTE ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global df, df_embeddings

    result = None
    stats = get_database_stats()

    if request.method == "POST":
        input_title = request.form.get("title", "").strip()
        if input_title:
            top_matches, is_unique, categories, detailed_matches, suggestions, analysis = find_similar_titles(input_title)

            result = {
                "input_title": input_title,
                "top_matches": top_matches,
                "is_unique": is_unique,
                "categories": categories,
                "detailed_matches": detailed_matches,
                "suggestions": suggestions,
                "analysis": analysis
            }

            # Save if unique
            if is_unique:
                try:
                    norm_input = normalize_title(input_title)
                    new_row = {"Title_Name_in_Lower": input_title.lower(), "normalized_title": norm_input}

                    if os.path.exists(EXCEL_FILE):
                        existing_df = pd.read_excel(EXCEL_FILE)
                        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        updated_df = pd.DataFrame([new_row])

                    updated_df.to_excel(EXCEL_FILE, index=False)
                    # reload df and embeddings
                    df, df_embeddings = load_titles()
                except Exception as e:
                    print("Error saving title:", e)

    results_html = generate_results_html(result)
    html = HTML_TEMPLATE.replace('%%RESULTS%%', results_html)
    html = html.replace('%%TOTAL_TITLES%%', str(stats['total_titles']))
    html = html.replace('%%AVG_LENGTH%%', str(stats['avg_length']))
    html = html.replace('%%AVG_WORDS%%', str(stats['avg_words']))

    return html

if __name__ == "__main__":
    app.run(debug=True)

