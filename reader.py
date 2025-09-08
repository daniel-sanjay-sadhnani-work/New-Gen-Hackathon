from telethon import TelegramClient, events
from telethon.errors import ChatIdInvalidError, ChannelPrivateError, FloodWaitError
import asyncio
import json
import os
from typing import List, Dict, Optional

# Replace these with your own values from https://my.telegram.org
api_id = 22321264        # <-- your API ID here (integer)
api_hash = 'c5d10a5619b36967053caecef328155b' # <-- your API hash here (string)

# List of channel usernames or IDs (e.g., 'mychannel', -1001234567890)
# Using usernames is more reliable than IDs
channels = [
    'SearchForJob', 
    'sgparttimers',
    'JobHitchpt',
    'SgPartTimeAgency',
    'Singapore_Jobs_SG',
    'nextjobs',
    'Sgparttimepro',
    'tuition' # Example: official Telegram channel
    # Add more channels as needed
    # You can use usernames (without @) or channel IDs
]

def load_configured_channels() -> List:
    """Return the merged list of channels including extras from env/file.

    Supports two optional sources in addition to the hardcoded `channels` list:
      - ENV EXTRA_CHANNELS: comma-separated usernames or IDs (with or without @)
      - ENV CHANNELS_FILE: path to a JSON file containing an array of usernames/IDs
    """
    merged: List = list(channels)

    # From env var EXTRA_CHANNELS
    extra_env = os.environ.get("EXTRA_CHANNELS", "")
    if extra_env.strip():
        for raw in extra_env.split(","):
            ch = raw.strip().lstrip("@")
            if ch and ch not in merged:
                merged.append(ch)

    # From JSON file pointed by CHANNELS_FILE
    channels_file = os.environ.get("CHANNELS_FILE")
    if channels_file and os.path.isfile(channels_file):
        try:
            with open(channels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for raw in data:
                    if isinstance(raw, str):
                        ch = raw.strip().lstrip("@")
                        if ch and ch not in merged:
                            merged.append(ch)
        except Exception as e:
            print(f"⚠️ Failed to load channels from {channels_file}: {e}")

    return merged

client = TelegramClient('multi_channel_session', api_id, api_hash)

async def get_channel_info(channel_identifier):
    """Get information about a channel to verify it's accessible"""
    try:
        entity = await client.get_entity(channel_identifier)
        return {
            'id': entity.id,
            'title': getattr(entity, 'title', 'Unknown'),
            'username': getattr(entity, 'username', None),
            'access_hash': getattr(entity, 'access_hash', None)
        }
    except (ChatIdInvalidError, ChannelPrivateError) as e:
        print(f"❌ Cannot access channel {channel_identifier}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error getting info for {channel_identifier}: {e}")
        return None

async def fetch_channel_messages(channel_identifier, limit=10):
    """Fetch messages from a channel with error handling"""
    try:
        print(f"\n📡 Fetching last {limit} messages from {channel_identifier}...")
        message_count = 0
        async for message in client.iter_messages(channel_identifier, limit=limit):
            if message.text:  # Only print text messages
                print(f"[{message.date}] {message.sender_id}: {message.text[:100]}...")
                message_count += 1
        
        if message_count == 0:
            print(f"   No text messages found in {channel_identifier}")
        else:
            print(f"   ✅ Fetched {message_count} messages from {channel_identifier}")
            
    except (ChatIdInvalidError, ChannelPrivateError) as e:
        print(f"   ❌ Cannot access channel {channel_identifier}: {e}")
    except FloodWaitError as e:
        print(f"   ⏳ Rate limited. Wait {e.seconds} seconds before retrying.")
    except Exception as e:
        print(f"   ❌ Error fetching messages from {channel_identifier}: {e}")

# ------------------
# New utilities for exporting recent messages to a JSON cache
# ------------------

async def _collect_recent_text_messages(channel_identifier, limit: int) -> List[Dict]:
    """Collect recent text messages for one channel and return a list of dicts."""
    collected: List[Dict] = []
    async for message in client.iter_messages(channel_identifier, limit=limit):
        if not getattr(message, 'text', None):
            continue
        collected.append({
            "channel": str(channel_identifier),
            "message_id": getattr(message, 'id', None),
            "date": message.date.isoformat() if getattr(message, 'date', None) else None,
            "sender_id": getattr(message, 'sender_id', None),
            "text": message.text
        })
    return collected

async def _collect_jobs_from_channels(channels_list: Optional[List], limit_per_channel: int) -> List[Dict]:
    """Validate channels and collect recent text messages from each."""
    print("🚀 Collecting recent messages for export...")
    valid_channels: List = []
    for channel in channels_list or load_configured_channels():
        info = await get_channel_info(channel)
        if info:
            valid_channels.append(channel)
        else:
            print(f"❌ Skipping inaccessible channel: {channel}")

    if not valid_channels:
        print("❌ No valid channels to collect from.")
        return []

    all_messages: List[Dict] = []
    for channel in valid_channels:
        try:
            channel_msgs = await _collect_recent_text_messages(channel, limit=limit_per_channel)
            # Attach resolved channel name where possible
            if channel_msgs:
                all_messages.extend(channel_msgs)
        except Exception as e:
            print(f"   ❌ Error collecting from {channel}: {e}")

    print(f"✅ Collected {len(all_messages)} messages in total")
    return all_messages

def export_recent_jobs_to_json(output_path: str = "jobs_cache.json", limit_per_channel: int = 100, channels_list: Optional[List] = None) -> Dict:
    """Export recent messages from configured channels into a JSON cache.

    Returns a summary dict with path and count. This helper can be imported from other modules.
    """
    try:
        # Load previous cache to compute newly added messages
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_cache: List[Dict] = json.load(f)
        except Exception:
            old_cache = []

        old_keys = set()
        for j in old_cache:
            ch = j.get("channel")
            mid = j.get("message_id")
            if ch is not None and mid is not None:
                old_keys.add((str(ch), mid))

        with client:
            messages: List[Dict] = client.loop.run_until_complete(
                _collect_jobs_from_channels(channels_list or load_configured_channels(), limit_per_channel)
            )

        # Compute how many in the fresh window are new compared to previous cache
        added = 0
        for m in messages:
            key = (str(m.get("channel")), m.get("message_id"))
            if key[0] is not None and key[1] is not None and key not in old_keys:
                added += 1

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(messages)} messages to {output_path}")
        return {"path": output_path, "count": len(messages), "added": added}
    except Exception as e:
        print(f"❌ Failed to export messages: {e}")
        return {"path": output_path, "count": 0, "added": 0, "error": str(e)}

async def export_recent_jobs_to_json_async(output_path: str = "jobs_cache.json", limit_per_channel: int = 100, channels_list: Optional[List] = None) -> Dict:
    """Async version of export: safe to call inside a running event loop."""
    try:
        # Load previous cache to compute newly added messages
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_cache: List[Dict] = json.load(f)
        except Exception:
            old_cache = []

        old_keys = set()
        for j in old_cache:
            ch = j.get("channel")
            mid = j.get("message_id")
            if ch is not None and mid is not None:
                old_keys.add((str(ch), mid))

        async with client:
            messages: List[Dict] = await _collect_jobs_from_channels(channels_list or load_configured_channels(), limit_per_channel)

        # Compute how many in the fresh window are new compared to previous cache
        added = 0
        for m in messages:
            key = (str(m.get("channel")), m.get("message_id"))
            if key[0] is not None and key[1] is not None and key not in old_keys:
                added += 1

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(messages)} messages to {output_path}")
        return {"path": output_path, "count": len(messages), "added": added}
    except Exception as e:
        print(f"❌ Failed to export messages: {e}")
        return {"path": output_path, "count": 0, "added": 0, "error": str(e)}

async def main():
    print("🚀 Starting Telegram Channel Reader...")
    
    # Validate channels first
    candidate_channels = load_configured_channels()
    valid_channels = []
    for channel in candidate_channels:
        info = await get_channel_info(channel)
        if info:
            print(f"✅ Channel accessible: {info['title']} (@{info['username'] or 'no username'})")
            valid_channels.append(channel)
        else:
            print(f"❌ Channel not accessible: {channel}")
    
    if not valid_channels:
        print("\n❌ No valid channels found. Please check your channel list.")
        print("💡 Try using channel usernames (without @) instead of IDs.")
        print("💡 Make sure you're a member of the channels you want to read.")
        return
    
    print(f"\n📋 Valid channels: {valid_channels}")
    
    # Fetch last 10 messages from each valid channel
    for channel in valid_channels:
        await fetch_channel_messages(channel, limit=10)
    
    print("\n🎧 Listening for new messages in all channels...")
    print("Press Ctrl+C to stop listening")

    # Set up a handler for new messages in all channels
    @client.on(events.NewMessage(chats=valid_channels))
    async def handler(event):
        try:
            chat_title = getattr(event.chat, 'title', f'Chat {event.chat_id}')
            sender_id = event.message.sender_id
            message_text = event.message.text or "[No text content]"
            
            print(f"\n📨 New message in {chat_title}:")
            print(f"   From: {sender_id}")
            print(f"   Time: {event.message.date}")
            print(f"   Text: {message_text}")
            
        except Exception as e:
            print(f"❌ Error handling new message: {e}")

    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        print("\n👋 Stopping listener...")

if __name__ == '__main__':
    try:
        # If the user wants a one-time export (non-interactive), set env EXPORT_ONCE=1
        if os.environ.get("EXPORT_ONCE") == "1":
            summary = export_recent_jobs_to_json()
            print(f"Export summary: {summary}")
        else:
            with client:
                client.loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")