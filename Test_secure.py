import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY not found in environment variables")
    print("💡 Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
    print("💡 Or set the environment variable: set GOOGLE_API_KEY=your_api_key_here")
    exit(1)

# Set the environment variable for the Google AI library
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    from langchain.chat_models import init_chat_model
    
    # Initialize the chat model
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    print("✅ Google AI model initialized successfully!")
    
    # Test the model with a simple query
    response = model.invoke("Hello! Can you confirm you're working?")
    print(f"🤖 AI Response: {response}")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Install required packages: pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Error initializing Google AI: {e}")
    print("💡 Check your API key and internet connection") 