# Shopping agent

Run these PowerShell commands from the repository root using the existing environment.

## Configuration

Set these values in the repository-root `.env` (never commit real keys):

```dotenv
GROQ_API_KEY=your_groq_api_key
SHOPPING_MODEL=qwen/qwen3.8-27b
SHOPPING_VISION_MODEL=qwen/qwen3.8-27b
SHOPPING_MAX_TOKENS=512
SHOPPING_OUTPUT_TOKEN_LIMIT=1000
LANGSMITH_TRACING=false
LANGCHAIN_TRACING_V2=false
```

Qwen 3.8 27B supports both tool calling and images. The reasoning options in
`shopping_agent.py` are Qwen-specific; review those options when selecting another model.
SHOPPING_OUTPUT_TOKEN_LIMIT caps the configured response budget even when
SHOPPING_MAX_TOKENS is set higher. Match this ceiling to your account limit.
The token cap limits each response; it does not cap accumulated conversation input
or bypass Groq account rate limits. The 512-token default stays below the
1,000 output-token limit reported by Groq for this account. If Groq returns a
429 after several calls, wait for the quota to reset before retrying. Restart
Streamlit after changing `.env`; an existing shell environment variable takes
precedence over `.env`. Local execution still calls Groq.

## Run

The existing `store.db` contains the demo catalogue. If the database is missing,
initialize it with:

```powershell
.\.venv\Scripts\python.exe .\10_project_shopping_agent\setup_db.py
```

Running setup again replaces the seeded product rows and resets the demo reviews.

Test text search first:

```powershell
.\.venv\Scripts\python.exe .\10_project_shopping_agent\shopping_agent.py
```

Start the chat and image-upload UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\10_project_shopping_agent\app.py
```

Try `organic honey under $20 with a rating of at least 4.5`, then upload
`resources/honey.png`. Confirm a selected product only when you want a demo order
written to `store.db`; checkout does not contact a payment or delivery service.

## Where to change things

- `.env`: API key, model IDs, response token cap, and tracing settings.
- `shopping_agent.py`: agent instructions and search, rating, checkout, and image tools.
- `app.py`: Streamlit interface and session chat history.
- `setup_db.py`: demo products and reviews.
- `reviews_api.py`: database rating aggregation; no separate API key or server is needed.

## Validation

Python syntax, offline model construction, database search/rating reads, and chat-state
initialization order were checked. Live Groq requests, browser interaction, and checkout
were not exercised.
API failures are displayed in the UI while preserving chat history. Image-search
triggers are consumed before execution to avoid automatic repeats on rerun.
If checkout fails after writing an order, inspect the orders table before retrying;
checkout is a demo and does not implement payment or order idempotency.

Image uploads are decoded with Pillow, normalized to RGB JPEG, and analyzed
directly before catalogue search, avoiding model-generated Windows paths. Chat
shows the original upload name. Temporary upload files are removed after analysis.
Offline validation covers the supplied honey PNG, normalized vision payload,
Python syntax, and the response-token ceiling using a mocked vision model.
Live Groq responses and browser interaction still require verification.

Image matching now validates a structured product/non-product result. Non-products,
uncertain/blank queries, and searches with no catalogue matches return a message
without suggesting products. Uploads are analyzed independently of chat history;
matching product names, IDs, prices, and ratings are rendered from database reads.
Regression checks passed with mocked non-product, empty query, no-match, honey,
and invalid-JSON results. Actual elephant recognition by Groq and browser behavior
were not verified; image classification remains model-dependent.

Loose/unpackaged food (such as oats in a bowl) is explicitly accepted by the
image-classification prompt; labels and retail packaging are not required.
Unknown organic status leaves both organic and non-organic matches eligible.
Underscore-separated search terms are normalized to spaces. Offline regression
checks cover oats, rolled_oats, whitespace, elephant rejection, and blank queries.
A live oats check was attempted but failed with APIConnectionError in this environment.
