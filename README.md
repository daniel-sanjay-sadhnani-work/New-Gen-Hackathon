# New-Gen-Hackathon
# Telegram Jobs RAG Bot

End-to-end system that reads job posts from multiple Telegram channels, caches them, builds a semantic index (RAG), and exposes a Telegram bot for conversational search with structured filtering.

## Contents
- `reader.py`: Telegram channel ingestion, validation, export to `jobs_cache.json`, and a live listener
- `channel_discovery.py`: Lists accessible channels and tests channel access
- `RAG_Chatbot.py`: RAG pipeline (embeddings + FAISS) over cached jobs with job filtering utilities
- `main.py`: Telegram bot (python-telegram-bot) wrapping the RAG chatbot and refresh command
- `jobs_cache.json`: Local cache of recent channel messages (jobs) used by RAG

## Architecture and Data Flow
1. Ingestion (`reader.py`)
   - Validates and reads messages from configured Telegram channels using Telethon.
   - Exports recent text messages to `jobs_cache.json` via `export_recent_jobs_to_json[_async]`.
2. Indexing (`RAG_Chatbot.py`)
   - Loads `jobs_cache.json`, splits text into chunks, embeds with `SentenceTransformer('all-MiniLM-L6-v2')`, and stores vectors in FAISS.
   - Provides semantic search and structured filtering over jobs.
3. Bot (`main.py`)
   - Telegram bot with commands: `/start`, `/help`, `/refresh`, `/quit` and free-text Q&A powered by the RAG pipeline.
4. Discovery (`channel_discovery.py`)
   - Utility to list channels you have access to and test access before adding to your config.

## Setup
1. Get Telegram API credentials
   - Visit https://my.telegram.org → Log in → API Development Tools → Create app → copy `api_id` and `api_hash`.
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Configure credentials
   - Open `reader.py` and `channel_discovery.py` and set `api_id` and `api_hash` to your own.
   - The first run will create `multi_channel_session.session` for login; keep it private.
4. Configure channels
   - Base channels live in `reader.py` under `channels = [...]`.
   - You can add more without editing code:
     - Environment variable `EXTRA_CHANNELS`: comma-separated usernames/IDs (with or without `@`).
     - Environment variable `CHANNELS_FILE`: path to a JSON file containing an array of channel usernames/IDs.
   - `reader.py` merges these via `load_configured_channels()` and uses them everywhere (validation, export, listener).

## Usage
### 1) Discover channels
```bash
python channel_discovery.py
```
Shows channels you can access and tests sample channels.

### 2) Export recent jobs (one-time)
Windows PowerShell example:
```powershell
$env:EXTRA_CHANNELS = "channel1,@channel2,-1001234567890"
# Optional: using a JSON file instead of EXTRA_CHANNELS
# $env:CHANNELS_FILE = "C:\\path\\to\\channels.json"
$env:EXPORT_ONCE = "1"
python .\reader.py
```
This generates/overwrites `jobs_cache.json` and prints a summary.

### 3) Run the bot
```bash
python main.py
```
In Telegram:
- Send `/refresh` to pull latest jobs from channels (async export) and rebuild the vector index.
- Ask natural-language questions or use structured queries (e.g., "$15 per hour in Tampines", "company: Mcdonalds").

## RAG Chatbot capabilities
- Semantic search over recent posts with top-k retrieval.
- Structured filters extracted from user text: hourly/monthly pay, hours per week/month, locations, companies, keywords.
- Exact company-name matching when the query lacks structured tokens.
- Small-talk handling for greetings/thanks.

## Configuration Reference
- `EXTRA_CHANNELS`: Comma-separated additional channels, e.g. `"foo,@bar,-100123"`.
- `CHANNELS_FILE`: Path to JSON array file of channels.
- `EXPORT_ONCE`: When set to `"1"`, `reader.py` performs a one-time export and exits.
- `jobs_cache.json`: Path used by default for cache (overridable when calling export functions).

## Operational Notes
- Channel updates: Removing channels from `channels` in `reader.py` immediately removes them from the base list. Ensure they are not present in `EXTRA_CHANNELS`/`CHANNELS_FILE` if you want them fully excluded.
- Rate limits: Telethon handles `FloodWaitError` warnings; reduce limits or channels if frequently rate-limited.
- Cache size: `export_recent_jobs_to_json[_async]` deduplicates by `(channel, message_id)` against the previous cache to report newly added count.

## Troubleshooting
- ChatIdInvalidError / ChannelPrivateError
  - Use discovery to validate usernames; ensure you joined the channel; private channels require invites.
- FloodWaitError (rate limiting)
  - Wait per error message; lower per-channel limits; reduce the number of channels.
- Authentication issues
  - Verify `api_id`/`api_hash` match your app; delete `multi_channel_session.session` and re-auth; ensure your phone is verified.
- No new jobs after /refresh
  - The cache was overwritten with the same window; confirm channels actually posted new content, or increase `limit_per_channel`.

## Security
- Do not commit real `api_id`, `api_hash`, Telegram bot token, or the session file.
- Treat `jobs_cache.json` as potentially sensitive content depending on channel policies.

## Project Structure
```
.
├─ reader.py                # Telethon ingestion, validation, export, listener
├─ main.py                  # Telegram bot integrating the RAG chatbot
├─ RAG_Chatbot.py           # RAG pipeline, filters, statistics
├─ channel_discovery.py     # Channel listing and access testing
├─ jobs_cache.json          # Exported messages (generated)
├─ requirements.txt         # Dependencies
└─ README.md                # This document
```

## Examples
### Fetch recent and exit
```powershell
$env:EXPORT_ONCE = "1"
python .\reader.py
```

### Bot commands
- /start: Welcome
- /help: Commands list
- /refresh: Re-export jobs and rebuild index
- /quit: Farewell

## Notes on Patterns and Extensibility
- Channel ingestion is centralized via `load_configured_channels()`; match this pattern when extending sources.
- Keep functions small, modular, and testable. Avoid broad try/except blocks; handle specific cases.
- If adding non-Telegram sources, normalize into the same JSON schema used in `jobs_cache.json` for reuse by the RAG pipeline.
