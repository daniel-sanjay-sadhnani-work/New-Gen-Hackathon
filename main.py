from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from RAG_Chatbot import RAGJobChatbot
from reader import export_recent_jobs_to_json_async

TOKEN: Final = '7510783924:AAH-gfo7_9-ybzTHuddbofpJiaHZW7cb7xY'
BOT_USERNAME: Final = '@job_Searcher_bbot'


#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am a job searcher bot. How can I help you today?')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Looks like you need some help!\n\nHere are the commands you can use:\n\n/start - Start the bot\n/help - Get help\n/refresh - Refresh the jobs\n/quit - Quit the bot')

async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('🔄 Fetching latest jobs from Telegram channels...')
    summary = await export_recent_jobs_to_json_async(output_path=chatbot.jobs_cache_path, limit_per_channel=100)
    count = summary.get('count', 0)
    added = summary.get('added', 0)
    path = summary.get('path', 'jobs_cache.json')
    if added > 0:
        chatbot._load_and_process_documents()
        await update.message.reply_text(f'✅ {added} new jobs found!')
    else:
        await update.message.reply_text('⚠️ No new jobs found, Please try again later.')
    #await update.message.reply_text(f'✅ {count}')

async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('👋 Goodbye!')

# Initialize RAG Chatbot
chatbot = RAGJobChatbot()


#Messages (RAG-powered)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = chatbot.chat(new_text)
        else:
            return
    else:
        response: str = chatbot.chat(text)

    print(f'Bot: "{response}"')
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('starting bot...')
    app = Application.builder().token(TOKEN).build()

    #Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('refresh', refresh_command))
    app.add_handler(CommandHandler('quit', quit_command))

    #Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    #Errors
    app.add_error_handler(error)

    #Polls the bot
    print('polling...')
    app.run_polling(poll_interval=5)