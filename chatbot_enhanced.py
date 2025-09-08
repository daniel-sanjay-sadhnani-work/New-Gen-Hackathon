import os
import sys

# Set the Google API key properly
GOOGLE_API_KEY = "AIzaSyAdLKbA1gRzle8-niDS_pO3qW6eLATYqU0"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def run_enhanced_chatbot():
    """Runs an enhanced chatbot with conversation memory and better error handling."""
    
    try:
        from langchain.chat_models import init_chat_model
        
        # Initialize the chat model
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        print("✅ Google AI model initialized successfully!")
        
        # Initialize conversation history for context
        conversation_history = []
        
        print("\n🤖 Welcome to the Enhanced AI Chatbot!")
        print("=" * 60)
        print("Features:")
        print("  - Maintains conversation context")
        print("  - Type 'quit' to exit")
        print("  - Type 'clear' to reset conversation")
        print("  - Type 'help' for assistance")
        print("=" * 60)
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                    
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue
                    
                elif user_input.lower() == 'help':
                    print("\n💡 Available commands:")
                    print("  - 'quit': Exit the chatbot")
                    print("  - 'clear': Clear conversation history")
                    print("  - 'help': Show this help message")
                    print("  - Just type normally to chat with the AI!")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    print("Please enter a message to continue the conversation.")
                    continue
                
                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Create context-aware prompt
                if len(conversation_history) > 2:
                    # Use recent conversation context
                    context_messages = conversation_history[-6:]  # Last 3 exchanges
                    context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_messages[:-1]])
                    full_prompt = f"Previous conversation:\n{context_prompt}\n\nUser: {user_input}\nAssistant:"
                else:
                    full_prompt = user_input
                
                # Get AI response
                response = model.invoke(full_prompt)
                ai_response = response.content
                
                # Add AI response to history
                conversation_history.append({"role": "assistant", "content": ai_response})
                
                # Display AI response
                print(f"🤖 AI: {ai_response}")
                
                # Keep conversation history manageable (max 20 messages)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Please try again or type 'quit' to exit.")
                
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install required packages: pip install langchain google-generativeai")
        
    except Exception as e:
        print(f"❌ Error initializing Google AI: {e}")
        print("💡 Check your API key and internet connection")

if __name__ == "__main__":
    run_enhanced_chatbot() 