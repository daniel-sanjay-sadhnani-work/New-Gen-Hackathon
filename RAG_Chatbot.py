import os
import json
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

# Set the Google API key properly
GOOGLE_API_KEY = "AIzaSyAdLKbA1gRzle8-niDS_pO3qW6eLATYqU0"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    from langchain.chat_models import init_chat_model
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import faiss
    from sentence_transformers import SentenceTransformer
    # Local import for exporting jobs
    from reader import export_recent_jobs_to_json, export_recent_jobs_to_json_async
    
    print("✅ All dependencies loaded successfully!")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Install required packages: pip install langchain langchain-community google-generativeai sentence-transformers faiss-cpu")
    exit(1)

class RAGJobChatbot:
    """
    RAG-powered chatbot for job search using Telegram messages as knowledge base.
    Implements vector embeddings, semantic search, context-aware responses, and structured job filtering.
    """
    
    def __init__(self, jobs_cache_path: str = "jobs_cache.json"):
        self.jobs_cache_path = jobs_cache_path
        self.vector_store = None
        self.embeddings_model = None
        self.chat_model = None
        self.documents = []
        self.conversation_history = []
        self.raw_jobs = []  # Store raw jobs for filtering
        self.messages = []  # For simple chat functionality
        
        # Initialize models
        self._initialize_models()
        
        # Load and process documents
        self._load_and_process_documents()
        
    def _initialize_models(self):
        """Initialize the embedding model and chat model."""
        try:
            # Initialize embedding model for semantic search
            print("🔄 Initializing embedding model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded!")
            
            # Initialize chat model
            print("🔄 Initializing chat model...")
            self.chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            print("✅ Chat model initialized!")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
    
    def _maybe_small_talk(self, text: str) -> Optional[str]:
        """Return a small-talk response for common phrases, else None.

        Handles simple greetings, well-being checks, and gratitude.
        """
        if not text:
            return None
        message = text.strip().lower()
        # Greetings
        if message in {"hi", "hello", "hey", "yo", "hiya"} or message.startswith(("hi ", "hello ", "hey ")):
            return "Hello! How can I help you find jobs today?"
        # How are you
        if "how are you" in message or message == "how are you?":
            return "I'm doing great and ready to help you with job searches. How are you?"
        # Thanks
        if message in {"thank you", "thanks", "thx", "ty"} or message.startswith(("thanks ", "thank you ")):
            return "You're welcome! If you need anything else, let me know!"
        return None
    
    def _load_and_process_documents(self):
        """Load telegram messages and process them for RAG."""
        print("🔄 Loading and processing documents...")
        
        # Load jobs from cache
        self.raw_jobs = self._load_jobs_cache()
        if not self.raw_jobs:
            print("⚠️ No jobs found. Run 'refresh' to fetch latest data.")
            return
        
        # Process documents
        self.documents = self._process_jobs_to_documents(self.raw_jobs)
        print(f"✅ Processed {len(self.documents)} document chunks")
        
        # Create vector store
        self._create_vector_store()
        print("✅ Vector store created!")
    
    def _load_jobs_cache(self) -> List[Dict]:
        """Load jobs from the cache file."""
        if os.path.exists(self.jobs_cache_path):
            try:
                with open(self.jobs_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading jobs cache: {e}")
                return []
        return []
    
    def _process_jobs_to_documents(self, jobs: List[Dict]) -> List[Dict]:
        """
        Process job postings into searchable document chunks.
        Each chunk contains relevant job information for semantic search.
        """
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        for i, job in enumerate(jobs):
            text = job.get("text", "")
            if not text.strip():
                continue
            
            # Create metadata for the job
            metadata = {
                "job_id": i,
                "channel": job.get("channel", "Unknown"),
                "date": job.get("date", ""),
                "sender_id": job.get("sender_id", ""),
                "source": "telegram"
            }
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                documents.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": j,
                        "full_text": text[:200] + "..." if len(text) > 200 else text
                    }
                })
        
        return documents
    
    def _create_vector_store(self):
        """Create FAISS vector store from documents."""
        if not self.documents:
            print("⚠️ No documents to vectorize")
            return
        
        # Extract text content
        texts = [doc["content"] for doc in self.documents]
        
        # Create embeddings
        print("🔄 Creating embeddings...")
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype('float32'))
        
        print(f"✅ Vector store created with {len(texts)} documents")
    
    def _search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        Returns top-k most relevant document chunks.
        """
        if not self.vector_store or not self.documents:
            return []
        
        # Encode query
        query_embedding = self.embeddings_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_store.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.documents))
        )
        
        # Return relevant documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    **self.documents[idx],
                    "similarity_score": float(score)
                })
        
        return results
    
    def _create_rag_prompt(self, user_query: str, relevant_docs: List[Dict]) -> str:
        """
        Create a RAG prompt with retrieved context and conversation history.
        """
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs[:3], 1):  # Use top 3 most relevant
            context_parts.append(f"Job Posting {i}:\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Build conversation history
        history = ""
        if self.conversation_history:
            history_parts = []
            for msg in self.conversation_history[-4:]:  # Last 4 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history = "\n".join(history_parts) + "\n\n"
        
        # Create the RAG prompt
        prompt = f"""You are a helpful job search assistant with access to recent job postings from Telegram channels. 
Use the provided job postings to answer questions accurately and help users find relevant opportunities.
You can also engage in small talk (like greetings or casual conversation) in a friendly and professional way.

{history}Context from recent job postings:
{context}

User Question: {user_query}

Instructions:
1. If the user is greeting or making small talk (e.g., "hi", "how are you?"), respond naturally and politely before returning to job-related help.
2. Use the provided job postings to answer the user's question if it's job-related.
3. If the job postings contain relevant information, **display the relevant Telegram messages (from {context}) you used as evidence**.
4. If no relevant information is found, say so clearly.
5. Be helpful and provide actionable advice.
6. Keep responses concise but informative.
7. If asked about specific jobs, provide details like salary, location, requirements, and contact info.
8. Always include the source job postings you retrieved so the user can see the original Telegram messages.

Assistant:"""

        
        return prompt
    
    def _extract_job_details(self, text: str) -> Dict:
        """Extract structured job details from text."""
        details = {
            "salary": [],
            "location": "",
            "hours": "",
            "requirements": [],
            "contact": "",
            "company": ""
        }
        
        # Extract salary information
        salary_patterns = [
            r'\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*hour|/\s*h|/\s*hr|hourly)',
            r'\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*month|/\s*month|monthly)',
            r'\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*day|/\s*day)'
        ]
        
        for pattern in salary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["salary"].extend(matches)
        
        # Extract location
        location_patterns = [
            r'📍\s*([^\n]+)',
            r'Location[:\s]*([^\n]+)',
            r'in\s+([A-Za-z\s]+(?:area|district|road|street))'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                details["location"] = match.group(1).strip()
                break
        
        # Extract contact information
        contact_patterns = [
            r'Telegram[:\s]*@([^\s\n]+)',
            r'WhatsApp[:\s]*([^\s\n]+)',
            r'Contact[:\s]*([^\n]+)'
        ]
        
        for pattern in contact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                details["contact"] = match.group(1).strip()
                break
        
        return details

    # ===== JOB FILTERING METHODS FROM Test.py =====
    
    def _to_float(self, num_str: str) -> float:
        """Convert string to float, handling commas."""
        return float(num_str.replace(",", "").strip())

    def parse_user_filters(self, query: str) -> dict:
        """Parse user query to extract structured filters."""
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
            result["pay_hourly_min"] = max(result["pay_hourly_min"] or 0.0, self._to_float(m.group(1)))

        # Pay with explicit monthly indicator
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*month|/\s*m(?:o|th)|/\s*month|monthly)", q, re.IGNORECASE):
            result["pay_monthly_min"] = max(result["pay_monthly_min"] or 0.0, self._to_float(m.group(1)))

        # Generic $amount without unit (heuristic)
        generic_amounts = [
            self._to_float(m.group(1))
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

    def extract_pay_hours_from_text(self, text: str) -> dict:
        """Extract pay and hours information from job text."""
        data = {
            "hourly": [],
            "monthly": [],
            "hours_week": [],
            "hours_month": [],
        }
        # Hourly pay
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*hour|/\s*h|/\s*hr|/\s*hour|hourly)", text, re.IGNORECASE):
            data["hourly"].append(self._to_float(m.group(1)))
        # Monthly pay
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s*month|/\s*m(?:o|th)|/\s*month|monthly)", text, re.IGNORECASE):
            data["monthly"].append(self._to_float(m.group(1)))
        # Generic amounts: classify heuristically
        for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b", text):
            amt = self._to_float(m.group(1))
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

    def filter_jobs_by_company_exact(self, jobs: list, company_name: str) -> list:
        """Filter jobs by exact company name match."""
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

    def job_matches_filters(self, job: dict, filters: dict) -> bool:
        """Check if a job matches the given filters."""
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

        details = self.extract_pay_hours_from_text(text)

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

    def filter_jobs_structured(self, jobs: list, query: str) -> list:
        """Filter jobs using structured parsing of user query."""
        filters = self.parse_user_filters(query)
        return [job for job in jobs if self.job_matches_filters(job, filters)]

    def filter_jobs_by_keywords(self, jobs: list, keywords: list) -> list:
        """Filter jobs by keyword matching."""
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

    def search_jobs(self, query: str) -> List[Dict]:
        """
        Search jobs using both structured filtering and keyword matching.
        Returns filtered job results.
        """
        if not self.raw_jobs:
            return []
        
        # Check if this is an exact company query (no structured tokens present)
        structured_tokens = [
            r"\$", r"\bper\b", r"/\s*(h|hr|hour|mo|mth|month|wk|week)",
            r"\bhours?\b", r"\bhrs?\b", r"\bmonthly\b", r"\bweek\b",
            r"\bin\b", r"location\s*:", r"company\s*:"
        ]
        is_company_exact = not any(re.search(pat, query, re.IGNORECASE) for pat in structured_tokens)

        if is_company_exact:
            return self.filter_jobs_by_company_exact(self.raw_jobs, query)
        else:
            return self.filter_jobs_structured(self.raw_jobs, query)

    def simple_chat(self, user_input: str) -> str:
        """
        Simple chat method without RAG - direct AI response.
        This replicates the Test.py chat functionality.
        """
        try:
            # Add user message to conversation history
            self.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = self.chat_model.invoke(user_input)
            ai_response = response.content
            
            # Add AI response to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def chat(self, user_input: str) -> str:
        """
        Main chat method that implements RAG pipeline.
        1. Search for relevant documents
        2. Create context-aware prompt
        3. Generate response using LLM
        """
        if not user_input.strip():
            return "Please provide a question or request."
        
        # Small talk fast-path
        small_talk = self._maybe_small_talk(user_input)
        if small_talk is not None:
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": small_talk})
            return small_talk
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Step 1: Search for relevant documents
            relevant_docs = self._search_similar_documents(user_input, top_k=5)
            
            if not relevant_docs:
                # No relevant documents found
                response = "I don't have any relevant job postings that match your query. Try asking about specific job types, locations, or salary ranges that might be available in the recent postings."
            else:
                # Step 2: Create RAG prompt
                rag_prompt = self._create_rag_prompt(user_input, relevant_docs)
                
                # Step 3: Generate response
                response = self.chat_model.invoke(rag_prompt)
                response_text = response.content
                
                # Step 4: Add response to history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return response_text
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def refresh_data(self) -> Dict:
        """Refresh the job data from Telegram channels."""
        print("🔄 Fetching latest jobs from Telegram channels...")
        summary = export_recent_jobs_to_json(output_path=self.jobs_cache_path, limit_per_channel=100)
        
        if summary.get('count', 0) > 0:
            # Reload and reprocess documents
            self._load_and_process_documents()
            print(f"✅ Refreshed data: {summary.get('count', 0)} new messages")
        else:
            print("⚠️ No new data fetched")
        
        return summary

    async def refresh_data_async(self) -> Dict:
        """Async refresh to be used inside running event loops (e.g., Telegram bot)."""
        print("🔄 Fetching latest jobs from Telegram channels (async)...")
        summary = await export_recent_jobs_to_json_async(output_path=self.jobs_cache_path, limit_per_channel=100)
        if summary.get('count', 0) > 0:
            self._load_and_process_documents()
            print(f"✅ Refreshed data: {summary.get('count', 0)} new messages")
        else:
            print("⚠️ No new data fetched")
        return summary
    
    def get_job_statistics(self) -> Dict:
        """Get statistics about the loaded job data."""
        if not self.documents:
            return {"total_jobs": 0, "total_chunks": 0}
        
        unique_jobs = len(set(doc["metadata"]["job_id"] for doc in self.documents))
        return {
            "total_jobs": unique_jobs,
            "total_chunks": len(self.documents),
            "channels": list(set(doc["metadata"]["channel"] for doc in self.documents))
        }

def main():
    """Main interactive chat interface."""
    print("🚀 Initializing RAG Job Chatbot...")
    
    try:
        # Initialize the RAG chatbot
        chatbot = RAGJobChatbot()
        
        # Display statistics
        stats = chatbot.get_job_statistics()
        print(f"\n📊 Loaded {stats['total_jobs']} jobs across {len(stats['channels'])} channels")
        print(f"📝 Created {stats['total_chunks']} searchable document chunks")
        
        print("\n🤖 Welcome to the AI Chatbot! (type 'quit' to exit)")
        print("=" * 50)
        print("Type 'refresh' to fetch latest jobs from Telegram channels")
        print("Type 'jobs <query>' to list jobs, e.g.:")
        print("  - jobs <role> eg. jobs administration")
        print("  - jobs <location> eg. jobs Tampines")
        print("  - jobs <pay> eg. jobs $15 per hour")
        print("  - jobs <company> eg. jobs company: Mcdonalds")
        print("  - jobs <hours> eg. jobs 15 hrs per week")
        print("\n💡 Ask me about:")
        print("  - Job opportunities in specific areas")
        print("  - Salary ranges for different positions")
        print("  - Part-time vs full-time opportunities")
        print("  - Specific companies or job types")
        print("  - Contact information for job postings")
        print("  - Requirements for specific positions")
        print("\nCommands:")
        print("  - 'refresh' - Update job data from Telegram")
        print("  - 'stats' - Show job statistics")
        print("  - 'quit' - Exit the chatbot")
        
        # Interactive chat loop
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
                summary = chatbot.refresh_data()
                print(f"Done. Cached {summary.get('count', 0)} messages to {summary.get('path')}")
                continue

            # List jobs with keywords / filters / company exact
            if user_input.lower().startswith('jobs'):
                query = user_input[4:].strip()
                if not query:
                    print("Please provide a query after 'jobs' (e.g., 'jobs Mcdonalds')")
                    continue
                
                if not chatbot.raw_jobs:
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
                    matched = chatbot.filter_jobs_by_company_exact(chatbot.raw_jobs, query)
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

                matched = chatbot.filter_jobs_structured(chatbot.raw_jobs, query)
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
            
            elif user_input.lower() == 'stats':
                stats = chatbot.get_job_statistics()
                print(f"\n📊 Job Statistics:")
                print(f"  Total Jobs: {stats['total_jobs']}")
                print(f"  Document Chunks: {stats['total_chunks']}")
                print(f"  Channels: {', '.join(stats['channels'])}")
                continue
            
            # Skip empty inputs
            if not user_input:
                print("Please enter a message to continue the conversation.")
                continue
            
            # Process user input through RAG pipeline for conversational responses
            response = chatbot.chat(user_input)
            print(f"🤖 AI: {response}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check your setup and try again.")

if __name__ == "__main__":
    main()