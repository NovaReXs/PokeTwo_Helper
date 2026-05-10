"""
Pokétwo Auto-Catcher Bot
========================
Monitors Discord for Pokétwo spawns, identifies the Pokémon using a local
HuggingFace ViT model (skshmjn/Pokemon-classifier-gen9-1025) — covers all
1025 Pokémon from Gen 1–9. No API key needed.

WARNING: Selfbots violate Discord ToS. Use on an alt account at your own risk.

Setup:
    pip install discord.py-self aiohttp transformers torch torchvision pillow

Config:
    Set DISCORD_TOKEN below (or via env var)
"""

import asyncio
import io
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import aiohttp
import discord
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Mengambil token dari file .env
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_USER_TOKEN_HERE":
    print("[!] ERROR: Set your DISCORD_TOKEN in the .env file.")
    exit(1)

# Pokétwo official bot user ID
POKETWO_ID = 716390085896962058

# Bot Groups and State
groups = {
    "1": {"channels": [1502127674850545774], "enabled": True, "mode": "catch"},
    "2": {"channels": [1498511187845845077], "enabled": True, "mode": "helper"},
    "3": {"channels": [], "enabled": True, "mode": "catch"},
}

# Delay range before catching (in seconds) — mimics human reaction time
CATCH_DELAY_MIN = 1.0
CATCH_DELAY_MAX = 2.0

# Set to True to log identified Pokémon without actually sending the catch command
DRY_RUN = False

# ID user atau role yang di-ping saat Pokémon langka muncul.
# Contoh User: "<@123456789012345678>" | Contoh Role: "<@&123456789012345678>"
# Kosongkan "" jika tidak ingin nge-ping. (ID Anda di bawah)
RARE_PING_ID = "<@&1502342878708633611>"

# ─── RARE POKÉMON LIST ───────────────────────────────────────────────────────
RARE_POKEMON = {
    # Gen 1
    "articuno", "zapdos", "moltres", "mewtwo", "mew",
    "dratini", "dragonair", "dragonite",
    # Gen 2
    "raikou", "entei", "suicune", "lugia", "ho-oh", "celebi",
    "larvitar", "pupitar", "tyranitar",
    # Gen 3
    "regirock", "regice", "registeel", "latias", "latios", "kyogre", "groudon", "rayquaza", "jirachi", "deoxys",
    "bagon", "shelgon", "salamence", "beldum", "metang", "metagross",
    # Gen 4
    "uxie", "mesprit", "azelf", "dialga", "palkia", "heatran", "regigigas", "giratina", "cresselia", "phione", "manaphy", "darkrai", "shaymin", "arceus",
    "gible", "gabite", "garchomp",
    # Gen 5
    "victini", "cobalion", "terrakion", "virizion", "tornadus", "thundurus", "reshiram", "zekrom", "landorus", "kyurem", "keldeo", "meloetta", "genesect",
    "deino", "zweilous", "hydreigon",
    # Gen 6
    "xerneas", "yveltal", "zygarde", "diancie", "hoopa", "volcanion",
    "goomy", "sliggoo", "goodra",
    # Gen 7
    "type-null", "silvally", "tapu-koko", "tapu-lele", "tapu-bulu", "tapu-fini", "cosmog", "cosmoem", "solgaleo", "lunala", "necrozma", "magearna", "marshadow", "zeraora", "meltan", "melmetal",
    "nihilego", "buzzwole", "pheromosa", "xurkitree", "celesteela", "kartana", "guzzlord", "poipole", "naganadel", "stakataka", "blacephalon",
    "jangmo-o", "hakamo-o", "kommo-o",
    # Gen 8
    "zacian", "zamazenta", "eternatus", "kubfu", "urshifu", "zarude", "regieleki", "regidrago", "glastrier", "spectrier", "calyrex", "enamorus",
    "dreepy", "drakloak", "dragapult",
    # Gen 9
    "ting-lu", "chien-pao", "wo-chien", "chi-yu", "koraidon", "miraidon", "walking-wake", "iron-leaves", "okidogi", "munkidori", "fezandipiti", "ogerpon", "gouging-fire", "raging-bolt", "iron-boulder", "iron-crown", "terapagos", "pecharunt",
    "frigibax", "arctibax", "baxcalibur"
}

# Helper mode: bot identifies the Pokémon but does NOT catch it.
# Instead it posts the name in the channel so your friend can catch it.

# ─── SETUP ───────────────────────────────────────────────────────────────────

MODEL_ID = "skshmjn/Pokemon-classifier-gen9-1025"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Loading Pokémon classifier model ({MODEL_ID}) on {DEVICE}...")
_processor = ViTImageProcessor.from_pretrained(MODEL_ID)
_model = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
_model.eval()
print("[OK] Model loaded!")

client = discord.Client()

# Track recently caught to avoid duplicates
recently_processed: set[int] = set()

# ─── POKEMON IDENTIFIER ──────────────────────────────────────────────────────

def crop_center_square(img: Image.Image) -> Image.Image:
    """Crop the image to a center square to maintain aspect ratio."""
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return img.crop((left, top, right, bottom))

def identify_pokemon_from_image(img: Image.Image) -> tuple[str, float]:
    """Run local ViT classifier and return (pokemon_name, confidence)."""
    # Crop the image to a perfect square first
    # This prevents the model from squashing the wide Pokétwo images and hallucinating!
    img = crop_center_square(img)

    inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    top_id = probs.argmax(-1).item()
    confidence = probs[0, top_id].item()
    name = _model.config.id2label[top_id]  # e.g. "pikachu"
    # Capitalize properly (handles "mr-mime" → "Mr-Mime" etc.)
    name = "-".join(w.capitalize() for w in name.split("-"))
    return name, confidence


async def identify_pokemon_from_url(image_url: str) -> str | None:
    """Download image from URL and identify the Pokémon using local model."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    print(f"[!] Failed to download image: HTTP {resp.status}")
                    return None
                image_data = await resp.read()

        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run inference in a thread so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        name, confidence = await loop.run_in_executor(
            None, identify_pokemon_from_image, img
        )

        print(f"[*] Classifier result: {name} (confidence: {confidence:.1%})")
        return name

    except Exception as e:
        print(f"[!] Model identification error: {e}")
        return None


# ─── DISCORD EVENTS ──────────────────────────────────────────────────────────

@client.event
async def on_ready():
    print(f"[OK] Logged in as: {client.user} ({client.user.id})")
    for g, data in groups.items():
        if data["channels"]:
            print(f"[OK] Channels {g}: {data['channels']} (Mode: {data['mode'].upper()})")
    print(f"[OK] Commands: !stop | !start | !status | !catch[1-3] | !helper[1-3] | !watch[1-3] | !addwatch[1-3] | !delwatch[1-3]")
    print("-" * 50)


@client.event
async def on_message(message: discord.Message):

    # ── Owner control commands (only YOU can use these) ──────────────────────
    if message.author.id == client.user.id:
        cmd = message.content.strip().lower()

        if cmd == "!stop":
            for g in groups: groups[g]["enabled"] = False
            print("[*] ALL PAUSED via command.")
            return
        elif cmd == "!start":
            for g in groups: groups[g]["enabled"] = True
            print("[*] ALL RESUMED via command.")
            return
        elif cmd == "!helper":
            for g in groups: groups[g]["mode"] = "helper"
            print("[*] Switched ALL to HELPER MODE (announce in channel).")
            return
        elif cmd == "!catch":
            for g in groups: groups[g]["mode"] = "catch"
            print("[*] Switched ALL to AUTO-CATCH MODE.")
            return
        elif cmd == "!status":
            for g, data in groups.items():
                state = "RUNNING" if data["enabled"] else "PAUSED"
                print(f"[*] Status | Group {g}: {state} ({data['mode'].upper()}) - Channels: {data['channels']}")
            return

        # Dynamic commands for specific groups (e.g., !stop1, !watch2, !addwatch3)
        match = re.match(r"^!(stop|start|helper|catch|watch|addwatch|delwatch)(\d+)$", cmd)
        if match:
            action, group = match.groups()
            if group not in groups:
                groups[group] = {"channels": [], "enabled": True, "mode": "catch"}
            
            if action == "stop":
                groups[group]["enabled"] = False
                print(f"[*] Group {group} PAUSED.")
            elif action == "start":
                groups[group]["enabled"] = True
                print(f"[*] Group {group} RESUMED.")
            elif action == "helper":
                groups[group]["mode"] = "helper"
                print(f"[*] Group {group} set to HELPER mode.")
            elif action == "catch":
                groups[group]["mode"] = "catch"
                print(f"[*] Group {group} set to AUTO-CATCH mode.")
            elif action == "watch":
                groups[group]["channels"] = [message.channel.id]
                ch_name = getattr(message.channel, 'name', 'Direct Message')
                print(f"[*] Set group {group} to watch channel #{ch_name} ({message.channel.id}).")
            elif action == "addwatch":
                if message.channel.id not in groups[group]["channels"]:
                    groups[group]["channels"].append(message.channel.id)
                ch_name = getattr(message.channel, 'name', 'Direct Message')
                print(f"[*] Added channel #{ch_name} ({message.channel.id}) to group {group}.")
            elif action == "delwatch":
                if message.channel.id in groups[group]["channels"]:
                    groups[group]["channels"].remove(message.channel.id)
                ch_name = getattr(message.channel, 'name', 'Direct Message')
                print(f"[*] Removed channel #{ch_name} ({message.channel.id}) from group {group}.")
            return

    # Only respond to Pokétwo bot for catching
    if message.author.id != POKETWO_ID:
        return

    # ── ANTI-BOT / CAPTCHA DETECTION ──────────────────────────────────────────
    # Pokétwo sends this message when it suspects you are a bot.
    # We must stop instantly to avoid getting banned.
    msg_content = message.content.lower()
    if "tell us you're human" in msg_content or "verify.poketwo.net/captcha" in msg_content:
        # Check if the captcha is actually for THIS account (ID is in the link or we are mentioned)
        if str(client.user.id) in message.content or client.user in message.mentions:
            for g in groups.values():
                if g["mode"] == "catch":
                    g["enabled"] = False
                
            print("\n\a" + "!" * 70)  # \a plays a beep sound in terminal
            print("!!! WARNING: POKÉTWO CAPTCHA DETECTED !!!".center(70))
            print("!!! AUTO-CATCHER CHANNELS HAVE BEEN PAUSED TO PROTECT YOUR ACCOUNT !!!".center(70))
            print("!" * 70)
            print(f"\nCaptcha Link: {message.content}\n")
            print("-> Please click the link and solve the Captcha manually.")
            print("-> Once solved, type '!start' in your bot console to resume catching.\n")
            print("!" * 70 + "\n")
            
            try:
                await message.channel.send("<@716390085896962058> incense pause")
                print("[*] Sent 'incense pause' to prevent incense waste.")
            except Exception as e:
                print(f"[!] Failed to send incense pause: {e}")
                
            return
        else:
            print(f"[*] Ignored CAPTCHA warning meant for another user.")

    # Filter channels and get mode/state
    current_group = None
    for g, data in groups.items():
        if message.channel.id in data["channels"]:
            current_group = g
            break
            
    if not current_group:
        return
        
    current_mode = groups[current_group]["mode"]
    is_enabled = groups[current_group]["enabled"]

    # Avoid processing the same message twice
    if message.id in recently_processed:
        return

    # Detect wild Pokémon spawn — Pokétwo uses embeds
    is_spawn = False
    image_url = None

    for embed in message.embeds:
        title = (embed.title or "").lower()
        description = (embed.description or "").lower()

        if "wild pokémon has appeared" in title or "wild pokémon has appeared" in description:
            is_spawn = True
            if embed.image:
                image_url = embed.image.url
            elif embed.thumbnail:
                image_url = embed.thumbnail.url
            break

    # Also check message content (some versions post in content)
    if not is_spawn and "a wild pokémon has appeared" in message.content.lower():
        is_spawn = True
        if message.attachments:
            image_url = message.attachments[0].url

    if not is_spawn or not image_url:
        return

    # Check if catching is enabled for this channel
    if not is_enabled:
        print("[*] Spawn detected but channel is paused — skipping.")
        return

    recently_processed.add(message.id)
    # Keep set small
    if len(recently_processed) > 200:
        recently_processed.clear()

    print(f"\n[!] Wild Pokémon spawned in #{message.channel} (guild: {message.guild})")
    print(f"    Image: {image_url}")

    # Identify Pokémon
    pokemon_name = await identify_pokemon_from_url(image_url)

    if not pokemon_name:
        print("[!] Could not identify Pokémon — skipping.")
        return

    print(f"[*] Identified: {pokemon_name}")

    if pokemon_name.lower() in RARE_POKEMON:
        print(f"\n\a[!!!] RARE POKÉMON DETECTED: {pokemon_name.upper()} [!!!]\n")
        if RARE_PING_ID:
            try:
                # Menangani jika user memasukkan banyak ID (tuple/list) atau hanya string
                ping_str = " ".join(RARE_PING_ID) if isinstance(RARE_PING_ID, (list, tuple)) else RARE_PING_ID
                await message.channel.send(f"🚨 {ping_str} Pokémon langka muncul: **{pokemon_name.upper()}**!")
            except discord.Forbidden:
                pass
            except Exception as e:
                print(f"[!] Gagal mengirim ping: {e}")

    # Mode is already determined above

    # ── Helper mode: announce instantly, no delay ─────────────────────────────
    if current_mode == "helper":
        await message.channel.send(f"`<@716390085896962058> c {pokemon_name}`")
        print(f"[OK] Helper: posted {pokemon_name} in channel instantly")
        return

    # Human-like delay (auto-catch mode only)
    delay = random.uniform(CATCH_DELAY_MIN, CATCH_DELAY_MAX)
    print(f"[*] Waiting {delay:.1f}s before catching...")
    await asyncio.sleep(delay)

    # Re-check in case !stop was sent during the delay
    is_enabled_now = groups[current_group]["enabled"]
    if not is_enabled_now:
        print("[*] Catch aborted — paused during processing.")
        return

    catch_command = f"<@716390085896962058> c {pokemon_name}"

    # ── Auto-catch mode ───────────────────────────────────────────────────────
    if DRY_RUN:
        print(f"[DRY RUN] Would send: {catch_command}")
    else:
        try:
            await message.channel.send(catch_command)
            print(f"[OK] Sent: {catch_command}")
        except discord.Forbidden:
            print(f"[!] No permission to send in #{message.channel}")
        except Exception as e:
            print(f"[!] Failed to send catch command: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Starting Pokétwo Auto-Catcher...")
    client.run(DISCORD_TOKEN)
