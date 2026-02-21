import os
import asyncio
import logging
import time
from collections import defaultdict
from html import escape
from dotenv import load_dotenv

from openai import AsyncOpenAI
from tavily import TavilyClient

from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# LOGGING 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#  ENV
load_dotenv("xx.env")

REQUIRED_KEYS = ["OPENAI_KEY", "TELEGRAM_TOKEN", "TAVILY_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise EnvironmentError(f"Missing env variables: {', '.join(missing)}")

OPENAI_KEY = os.getenv("OPENAI_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

#  CLIENTS 
openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
tavily_client = TavilyClient(api_key=TAVILY_KEY)

GPT_MODEL = "gpt-4o-mini"
MAX_INPUT_LEN = 500
RATE_LIMIT_SECONDS = 5
MAX_EVIDENCE_LEN = 6000

user_last_request = defaultdict(float)
fact_cache = {}

#  SYSTEM PROMPT 
SYSTEM_PROMPT = """
You are a professional fact-checking AI.

Analyze the claim using the provided evidence.
Respond strictly in this format:

<b>Verdict:</b> True / False / Misleading / Partially True / Insufficient Evidence
<b>Confidence:</b> 0-100%
<b>Explanation:</b> Short reasoning based only on the evidence.
Do not add extra commentary.
"""

#  WEB SEARCH 
def search_web(query: str):
    try:
        logger.info(f"Searching Tavily for: {query}")

        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False,  
        )

        parts = []
        urls = []

        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}")

        for r in response.get("results", []):
            url = r.get("url")
            content = r.get("content")

            if url:
                urls.append(url)

            if content:
                parts.append(content.strip())

        evidence = "\n\n---\n\n".join(parts)

        if len(evidence) > MAX_EVIDENCE_LEN:
            evidence = evidence[:MAX_EVIDENCE_LEN] + "\n... [truncated]"

        return evidence, urls

    except Exception:
        logger.exception("Tavily search failed")
        return "", []

# FACT CHECK 
async def fact_check(claim: str, user_id: int):

    # Cache check
    if claim in fact_cache:
        return fact_cache[claim]

    evidence, urls = search_web(claim)

    if not evidence.strip():
        return (
            "<b>Verdict:</b> Insufficient Evidence\n"
            "<b>Confidence:</b> 0%\n"
            "<b>Explanation:</b> No reliable sources found."
        )

    try:
        response = await openai_client.responses.create(
            model=GPT_MODEL,
            temperature=0,
            max_output_tokens=600,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Claim:\n{claim}\n\nEvidence:\n{evidence}",
                },
            ],
            metadata={"user_id": str(user_id)},
        )

        verdict_text = response.output_text.strip()

    except Exception:
        logger.exception("OpenAI request failed")
        return "AI processing failed. Please try again later."

    # Add sources
    if urls:
        verdict_text += "\n\n<b>Sources:</b>\n"
        verdict_text += "\n".join(f"• {escape(u)}" for u in urls)

    # Cache result
    fact_cache[claim] = verdict_text

    return verdict_text

# TELEGRAM HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Welcome to Fact-Checking Bot</b>\n\n"
        "Send any claim and I will verify it using live web search.",
        parse_mode=ParseMode.HTML,
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    claim = (update.message.text or "").strip()
    user_id = user.id

    if not claim:
        await update.message.reply_text("Please send a valid claim.")
        return

    if len(claim) > MAX_INPUT_LEN:
        await update.message.reply_text(
            f"Message too long (max {MAX_INPUT_LEN} characters)."
        )
        return

    now = time.time()
    if now - user_last_request[user_id] < RATE_LIMIT_SECONDS:
        await update.message.reply_text(
            "Please wait before sending another request."
        )
        return

    user_last_request[user_id] = now

    await update.message.chat.send_action(ChatAction.TYPING)
    thinking = await update.message.reply_text("News-checking... Please wait.")

    try:
        result = await asyncio.wait_for(
            fact_check(claim, user_id),
            timeout=40,
        )

        await thinking.edit_text(
            result,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    except asyncio.TimeoutError:
        await thinking.edit_text(" Request timed out. Please try again.")
    except Exception:
        logger.exception("Unexpected error")
        await thinking.edit_text(" Error processing request.")

#  MAIN 
def main():
    logger.info("Starting Telegram Bot...")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()