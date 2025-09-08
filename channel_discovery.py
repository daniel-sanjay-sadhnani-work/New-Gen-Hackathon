from telethon import TelegramClient
from telethon.errors import ChatIdInvalidError, ChannelPrivateError
import asyncio

# Replace these with your own values from https://my.telegram.org
api_id = 22321264
api_hash = 'c5d10a5619b36967053caecef328155b'

client = TelegramClient('multi_channel_session', api_id, api_hash)

async def discover_channels():
    """Discover channels that the user is a member of"""
    print("🔍 Discovering accessible channels...")
    
    try:
        # Get dialogs (chats, channels, groups)
        async for dialog in client.iter_dialogs():
            if dialog.is_channel:  # Only show channels, not private chats
                print(f"📺 Channel: {dialog.title}")
                print(f"   ID: {dialog.id}")
                print(f"   Username: @{dialog.entity.username}" if dialog.entity.username else "   Username: None")
                print(f"   Members: {dialog.entity.participants_count}" if hasattr(dialog.entity, 'participants_count') else "   Members: Unknown")
                print()

    except Exception as e:
        print(f"❌ Error discovering channels: {e}")

async def test_channel_access(channel_identifier):
    """Test if a specific channel is accessible"""
    print(f"🧪 Testing access to: {channel_identifier}")
    
    try:
        entity = await client.get_entity(channel_identifier)
        print(f"✅ Accessible!")
        print(f"   Title: {getattr(entity, 'title', 'Unknown')}")
        print(f"   ID: {entity.id}")
        print(f"   Username: @{getattr(entity, 'username', 'None')}")
        
        # Try to get some basic info
        try:
            # Get participant count if available
            if hasattr(entity, 'participants_count'):
                print(f"   Members: {entity.participants_count}")
        except:
            pass
            
        return True
        
    except (ChatIdInvalidError, ChannelPrivateError) as e:
        print(f"❌ Cannot access: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    print("🚀 Telegram Channel Discovery Tool")
    print("=" * 40)
    
    # First, discover all accessible channels
    await discover_channels()
    
    print("\n" + "=" * 40)
    print("🧪 Test specific channels:")
    
    # Test some common channels
    test_channels = [
        'telegram',  # Official Telegram channel
        'durov',     # Pavel Durov's channel
        'telegramtips',  # Telegram tips channel
    ]
    
    for channel in test_channels:
        await test_channel_access(channel)
        print()
    
    print("💡 Tips:")
    print("   - Use channel usernames (without @) in your reader.py")
    print("   - Make sure you're a member of channels you want to read")
    print("   - Some channels may be private and require invitation")

if __name__ == '__main__':
    try:
        with client:
            client.loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 