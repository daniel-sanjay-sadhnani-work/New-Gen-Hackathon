# Setup Guide

## Google AI Authentication Fix

The error you encountered is due to improper Google AI API key configuration. Here's how to fix it:

### Option 1: Quick Fix (Use Test.py)
The `Test.py` file has been updated with proper authentication. Simply run:
```bash
python Test.py
```

### Option 2: Secure Setup (Recommended)
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a .env file:**
   Create a file named `.env` in your project root with:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Get your Google AI API key:**
   - Go to https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy it to your .env file

4. **Run the secure version:**
   ```bash
   python Test_secure.py
   ```

### Option 3: Environment Variable
Set the environment variable directly:
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"

# Windows Command Prompt
set GOOGLE_API_KEY=your_api_key_here

# Then run
python Test.py
```

## Telegram Setup
Your Telegram credentials are already configured in the Python files. If you need to update them:

1. Go to https://my.telegram.org
2. Create a new application
3. Update `api_id` and `api_hash` in `reader.py` and `channel_discovery.py`

## Testing Your Setup
1. **Test Google AI:**
   ```bash
   python Test.py
   ```

2. **Test Telegram:**
   ```bash
   python channel_discovery.py
   ```

3. **Run the main reader:**
   ```bash
   python reader.py
   ```

## Troubleshooting
- **"DefaultCredentialsError"**: Your Google API key is not set correctly
- **"ImportError"**: Run `pip install -r requirements.txt`
- **"ChatIdInvalidError"**: Check your Telegram credentials 