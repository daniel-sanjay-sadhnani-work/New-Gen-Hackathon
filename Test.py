import getpass
import os
import json
import re

# Set the Google API key properly
GOOGLE_API_KEY = "AIzaSyAdLKbA1gRzle8-niDS_pO3qW6eLATYqU0"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    from langchain.chat_models import init_chat_model
    # Local import for exporting jobs
    from reader import export_recent_jobs_to_json
    
    # Initialize the chat model
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    print("✅ Google AI model initialized successfully!")
    
    # Initialize conversation history
    messages = []
    jobs_cache_path = "jobs_cache.json"

    def load_jobs_cache() -> list:
        if os.path.exists(jobs_cache_path):
            try:
                with open(jobs_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def filter_jobs_by_keywords(jobs: list, keywords: list) -> list:
        if not jobs:
            return []
        if not keywords:
            return jobs
        pattern = re.compile("|".join([re.escape(k) for k in keywords]), re.IGNORECASE)
        filtered = []
        for job in jobs:
            text = job.get("text", "")
            if pattern.search(text):
                filtered.append(job)
        return filtered

    def _to_float(num_str: str) -> float:
        return float(num_str.replace(",", "").strip())

    def parse_user_filters(query: str) -> dict:
        q = query.strip()
        result = {
            "keywords": [],
            "pay_hourly_min": None,
            "pay_monthly_min": None,
            "hours_week_target": None,
            "hours_month_target": None,
            "locations": [],
            "companies": [],
        }

        # Pay with explicit hourly indicator
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*hour|/\s*h|/\s*hr|/\s*hour|hourly)", q, re.IGNORECASE):
            result["pay_hourly_min"] = max(result["pay_hourly_min"] or 0.0, _to_float(m.group(1)))

        # Pay with explicit monthly indicator
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*month|/\s*m(?:o|th)|/\s*month|monthly)", q, re.IGNORECASE):
            result["pay_monthly_min"] = max(result["pay_monthly_min"] or 0.0, _to_float(m.group(1)))

        # Generic $amount without unit (heuristic)
        generic_amounts = [
            _to_float(m.group(1))
            for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b", q)
        ]
        for amt in generic_amounts:
            if amt <= 100 and (result["pay_hourly_min"] is None or amt > result["pay_hourly_min"]):
                result["pay_hourly_min"] = amt
            if amt > 100 and (result["pay_monthly_min"] is None or amt > result["pay_monthly_min"]):
                result["pay_monthly_min"] = amt

        # Hours targets
        for m in re.finditer(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:hours|hour|hrs|h)\s*(?:per\s*(week|wk|w)|/\s*(week|wk|w)|per\s*(month|mth|mo)|/\s*(month|mth|mo))", q, re.IGNORECASE):
            value = float(m.group(1))
            unit_groups = m.groups()[1:]
            unit_text = " ".join([u for u in unit_groups if u])
            if re.search(r"(week|wk|w)", unit_text, re.IGNORECASE):
                result["hours_week_target"] = value
            elif re.search(r"(month|mth|mo)", unit_text, re.IGNORECASE):
                result["hours_month_target"] = value

        # Location and company labels (explicit)
        for m in re.finditer(r"location\s*[:=]\s*([\w\s,&\-]+)", q, re.IGNORECASE):
            loc = m.group(1).strip()
            if loc:
                result["locations"].append(loc)
        for m in re.finditer(r"company\s*[:=]\s*([\w\s,&\-]+)", q, re.IGNORECASE):
            comp = m.group(1).strip()
            if comp:
                result["companies"].append(comp)

        # Location via 'in <place>' and company via 'at <name>'
        for m in re.finditer(r"\bin\s+([A-Za-z][\w\s&\-]{1,50})", q, re.IGNORECASE):
            loc = m.group(1).strip()
            # Trim trailing stop words that often follow locations
            loc = re.sub(r"\s+(per|for|with|and|or)$", "", loc, flags=re.IGNORECASE)
            if loc:
                result["locations"].append(loc)
        for m in re.finditer(r"\bat\s+([A-Za-z][\w\s&\-]{1,50})", q, re.IGNORECASE):
            comp = m.group(1).strip()
            comp = re.sub(r"\s+(in|at|for|with|and|or)$", "", comp, flags=re.IGNORECASE)
            if comp:
                result["companies"].append(comp)

        # Keywords are remaining free text (basic approach)
        tmp = q
        tmp = re.sub(r"\$\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:per\s*(hour|month)|/\s*(h|hr|hour|mo|mth|month)|hourly|monthly)?", " ", tmp, flags=re.IGNORECASE)
        tmp = re.sub(r"[0-9]{1,3}(?:\.[0-9]+)?\s*(?:hours|hour|hrs|h)\s*(?:per\s*(week|wk|w|month|mth|mo)|/\s*(week|wk|w|month|mth|mo))", " ", tmp, flags=re.IGNORECASE)
        tmp = re.sub(r"(location|company)\s*[:=]\s*[\w\s,&\-]+", " ", tmp, flags=re.IGNORECASE)
        tmp = re.sub(r"\bin\s+[A-Za-z][\w\s&\-]{1,50}", " ", tmp, flags=re.IGNORECASE)
        tmp = re.sub(r"\bat\s+[A-Za-z][\w\s&\-]{1,50}", " ", tmp, flags=re.IGNORECASE)
        words = [w for w in re.split(r"\s+", tmp) if w]
        result["keywords"] = words
        return result

    def extract_pay_hours_from_text(text: str) -> dict:
        data = {
            "hourly": [],
            "monthly": [],
            "hours_week": [],
            "hours_month": [],
        }
        # Hourly pay
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*hour|/\s*h|/\s*hr|/\s*hour|hourly)", text, re.IGNORECASE):
            data["hourly"].append(_to_float(m.group(1)))
        # Monthly pay
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*month|/\s*m(?:o|th)|/\s*month|monthly)", text, re.IGNORECASE):
            data["monthly"].append(_to_float(m.group(1)))
        # Generic amounts: classify heuristically
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b", text):
            amt = _to_float(m.group(1))
            if amt <= 100:
                data["hourly"].append(amt)
            elif amt > 100:
                data["monthly"].append(amt)
        # Hours
        for m in re.finditer(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:hours|hour|hrs|h)\s*(?:per\s*(week|wk|w)|/\s*(week|wk|w)|per\s*(month|mth|mo)|/\s*(month|mth|mo))", text, re.IGNORECASE):
            value = float(m.group(1))
            unit_text = " ".join([g for g in m.groups()[1:] if g])
            if re.search(r"(week|wk|w)", unit_text, re.IGNORECASE):
                data["hours_week"].append(value)
            elif re.search(r"(month|mth|mo)", unit_text, re.IGNORECASE):
                data["hours_month"].append(value)
        return data

    def filter_jobs_by_company_exact(jobs: list, company_name: str) -> list:
        phrase = company_name.strip()
        if not phrase:
            return []
        # Exact phrase with non-word boundaries on both sides (case-insensitive)
        pattern = re.compile(r"(?<!\\w)" + re.escape(phrase) + r"(?!\\w)", re.IGNORECASE)
        matched = []
        for job in jobs:
            text = job.get("text", "")
            if text and pattern.search(text):
                matched.append(job)
        return matched

    def job_matches_filters(job: dict, filters: dict) -> bool:
        text = job.get("text", "")
        if not text:
            return False

        # Keywords: any match
        keywords = [k for k in filters.get("keywords", []) if k]
        if keywords:
            if not re.search("|".join([re.escape(k) for k in keywords]), text, re.IGNORECASE):
                return False

        # Location/company simple includes
        for loc in filters.get("locations", []):
            if loc and loc.lower() not in text.lower():
                return False
        for comp in filters.get("companies", []):
            if comp and comp.lower() not in text.lower():
                return False

        details = extract_pay_hours_from_text(text)

        # Pay filters
        hourly_min = filters.get("pay_hourly_min")
        if hourly_min is not None:
            if not any(v >= hourly_min for v in details.get("hourly", [])):
                return False
        monthly_min = filters.get("pay_monthly_min")
        if monthly_min is not None:
            if not any(v >= monthly_min for v in details.get("monthly", [])):
                return False

        # Hours targets with tolerance
        tol_week = 5.0
        tol_month = 20.0
        week_target = filters.get("hours_week_target")
        if week_target is not None:
            if not any(abs(v - week_target) <= tol_week for v in details.get("hours_week", [])):
                return False
        month_target = filters.get("hours_month_target")
        if month_target is not None:
            if not any(abs(v - month_target) <= tol_month for v in details.get("hours_month", [])):
                return False

        return True

    def filter_jobs_structured(jobs: list, query: str) -> list:
        filters = parse_user_filters(query)
        return [job for job in jobs if job_matches_filters(job, filters)]
    
    print("\n🤖 Welcome to the AI Chatbot! (type 'quit' to exit)")
    print("=" * 50)
    print("Type 'refresh' to fetch latest jobs from Telegram channels")
    print("Type 'jobs <query>' to list jobs, e.g.:")
    print("  - jobs <role> eg. jobs administration")
    print("  - jobs <location> eg. jobs Tampines")
    print("  - jobs <pay> eg. jobs $15 per hour")
    print("  - jobs <company> eg. jobs company: Mcdonalds")
    print("  - jobs <hours> eg. jobs 15 hrs per week")
    
    # Interactive chatbot loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() == 'quit':
            print("👋 Goodbye! Thanks for chatting!")
            break

        # Refresh jobs cache on demand
        if user_input.lower() == 'refresh':
            print("🔄 Fetching latest jobs from Telegram channels...")
            summary = export_recent_jobs_to_json(output_path=jobs_cache_path, limit_per_channel=100)
            print(f"Done. Cached {summary.get('count', 0)} messages to {summary.get('path')}")
            continue

        # List jobs with keywords / filters / company exact
        if user_input.lower().startswith('jobs'):
            query = user_input[4:].strip()
            if not query:
                print("Please provide a query after 'jobs' (e.g., 'jobs Mcdonalds')")
                continue
            jobs = load_jobs_cache()
            if not jobs:
                print("No cached jobs found. Type 'refresh' first to fetch the latest posts.")
                continue

            # Decide if this is an exact company query (no structured tokens present)
            structured_tokens = [
                r"\$", r"\bper\b", r"/\s*(h|hr|hour|mo|mth|month|wk|week)",
                r"\bhours?\b", r"\bhrs?\b", r"\bmonthly\b", r"\bweek\b",
                r"\bin\b", r"location\s*:", r"company\s*:"
            ]
            is_company_exact = not any(re.search(pat, query, re.IGNORECASE) for pat in structured_tokens)

            if is_company_exact:
                matched = filter_jobs_by_company_exact(jobs, query)
                if not matched:
                    print("No matching jobs found for that exact company name.")
                    continue
                print(f"\n📋 Found {len(matched)} posts for job '{query.strip()}':\n")
                for idx, job in enumerate(matched[:20], start=1):
                    snippet = job.get('text', '')
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    print(f"{idx}. [{job.get('channel')}] {snippet}")
                if len(matched) > 20:
                    print(f"... and {len(matched) - 20} more. Refine further.")
                continue

            matched = filter_jobs_structured(jobs, query)
            if not matched:
                print("No matching jobs found for your query.")
                continue
            # Show a concise list
            print(f"\n📋 Found {len(matched)} matching posts:\n")
            for idx, job in enumerate(matched[:20], start=1):  # limit output
                snippet = job.get('text', '')
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                print(f"{idx}. [{job.get('channel')}] {snippet}")
            if len(matched) > 20:
                print(f"... and {len(matched) - 20} more. Refine your keywords to narrow down.")
            continue
        
        # Skip empty inputs
        if not user_input:
            print("Please enter a message to continue the conversation.")
            continue
        
        try:
            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = model.invoke(user_input)
            ai_response = response.content
            
            # Add AI response to conversation history
            messages.append({"role": "assistant", "content": ai_response})
            
            # Display AI response
            print(f"🤖 AI: {ai_response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Please try again or type 'quit' to exit.")
            
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Install required packages: pip install langchain google-generativeai")
    
except Exception as e:
    print(f"❌ Error initializing Google AI: {e}")
    print("💡 Check your API key and internet connection")


    