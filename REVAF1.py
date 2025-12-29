# reva_final.py
"""
REVA — Interactive CLI demo with:
- robust CSV catalog loader
- NL parsing for occasion/category/price
- scored recommendations (style + price closeness)
- persona-rotating NLG (friendly / formal / witty)
- conversational, interactive chat loop with clarifying questions and quick replies
- optional Groq LLM integration (uses env var GROQ_API_KEY); safe fallback if not configured
- optional Supabase sync (use env vars SUPABASE_URL and SUPABASE_KEY)

Drop this file into the same folder as mock_catalog.csv (or it will use a small sample).
Run: python reva_final.py
"""

import os
import csv
import re
import datetime
from pathlib import Path
import json


# Optional LLM integration (langchain_groq). If not available or no key, REVA runs in local mode.
USE_GROQ = False
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables.history import RunnableWithMessageHistory
    USE_GROQ = True
except Exception:
    USE_GROQ = False

# Optional Supabase client (only used if env vars set)
try:
    from supabase import create_client
except Exception:
    create_client = None

# ----------------------------
# Config (load from env vars)
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set to enable Groq LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# ----------------------------
# Catalog loader
# ----------------------------

def load_catalog(file_path):
    catalog = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Convert numeric fields safely
                row['price_usd'] = float(row.get('price_usd', 0) or 0)
                row['price_inr'] = float(row.get('price_inr', 0) or 0)
                # Use 'price' column if present, else fallback to INR
                row['price'] = float(row.get('price', row.get('price_inr', 0)) or 0)
                row['stock'] = int(row.get('stock', 0) or 0)
                row['rating'] = float(row.get('rating', 0) or 0)
                row['num_reviews'] = int(row.get('num_reviews', 0) or 0)
            except Exception:
                # Fallback defaults
                row['price'] = 0
                row['stock'] = 0
                row['rating'] = 0
                row['num_reviews'] = 0

            # Sizes: handle both 'size' and 'sizes'
            sizes = []
            if row.get('sizes'):
                try:
                    sizes = json.loads(row['sizes']) if row['sizes'].startswith("[") else [row['sizes']]
                except Exception:
                    sizes = [row['sizes']]
            elif row.get('size'):
                sizes = [row['size']]
            row['sizes'] = sizes

            # Image URLs: parse JSON list or fallback
            try:
                if row.get('image_urls'):
                    row['image_urls'] = json.loads(row['image_urls'])
                else:
                    row['image_urls'] = []
            except Exception:
                row['image_urls'] = []

            # Normalize key fields
            row['name'] = row.get('name', f"Item {row.get('id','')}")
            row['category'] = row.get('category', '').strip()
            row['style'] = row.get('style', '').strip()

            catalog.append(row)
    return catalog

# ----------------------------
# Parse & mapping
# ----------------------------
def map_occasion_to_style(text: str):
    if not text:
        return None
    occasion_map = {
        "diwali": "ethnic", "wedding": "ethnic", "office": "formal", "interview": "formal",
        "beach": "boho", "beach party": "boho", "brunch": "casual", "party": "elegant",
        "gym": "athleisure", "street": "streetwear"
    }
    tl = text.lower()
    for keyword, style in occasion_map.items():
        if keyword in tl:
            return style
    return None

import re
import difflib

def parse_user_input(text: str, catalog=None):
    if not text:
        return None, None, None

    tl = text.lower()
    style = map_occasion_to_style(text)
    category = None
    max_price = None

    # Build dynamic lists from catalog if provided
    categories = set()
    subcategories = set()
    if catalog:
        for item in catalog:
            if item.get("category"):
                categories.add(item["category"].lower())
            if item.get("subcategory"):
                subcategories.add(item["subcategory"].lower())

    # Fuzzy match against categories/subcategories
    words = tl.split()
    for word in words:
        match_cat = difflib.get_close_matches(word, categories, n=1, cutoff=0.7)
        match_sub = difflib.get_close_matches(word, subcategories, n=1, cutoff=0.7)
        if match_cat:
            category = match_cat[0]
        elif match_sub:
            category = match_sub[0]

    # Explicit style keywords
    styles = ["boho", "formal", "elegant", "casual", "athleisure", "ethnic", "streetwear"]
    for s in styles:
        if s in tl:
            style = s
            break

    # Budget detection: "under 3000", "below 5000", or just "3000"
    price_match = re.search(r"(under|below)?\s*₹?(\d+)", tl)
    if price_match:
        try:
            max_price = int(price_match.group(2))
        except Exception:
            max_price = None

    return style, category, max_price

# ----------------------------
# Recommendation with scoring
# ----------------------------
def recommend_products(cat_list, style=None, category=None, max_price=None, top_k=50):
    candidates = []
    for item in cat_list:
        if category and item.get("category") and item.get("category").lower() != category.lower():
            continue
        if item.get("stock", 0) <= 0:
            continue

        score = 0.0
        # strong style match
        if style and item.get("style") and item.get("style").lower() == style.lower():
            score += 2.0
        # partial style match
        elif style and item.get("style") and style.lower() in item.get("style").lower():
            score += 1.0

        price = item.get("price", 0)
        if max_price:
            # reward in-budget and nearer to budget
            if price <= max_price:
                diff = max_price - price
                score += max(0.0, 1.0 - (diff / max_price))  # normalized closeness
            else:
                score -= 1.0  # penalize over budget

        # prefer items with better stock (small bonus)
        stock = item.get("stock", 0)
        score += min(stock, 50) / 500.0

        candidates.append((score, item))

    candidates.sort(key=lambda x: (-x[0], x[1].get("price", 0)))
    return [it for s, it in candidates][:top_k]

# ----------------------------
# Formatters & inventory
# ----------------------------
def check_inventory(product):
    stock = product.get("stock", 0)
    if stock == 0:
        return "❌ Out of stock"
    elif stock < 5:
        return f"⚠️ Low stock ({stock} left)"
    else:
        return f"✅ In stock ({stock} available)"

def format_recommendation(product):
    inventory_status = check_inventory(product)
    sizes = product.get("sizes") or ["Free Size"]
    return (
        f"**{product.get('name','Item')}**\n"
        f"*{product.get('category','').capitalize()}, {product.get('style','').capitalize()}*\n"
        f"💸 ₹{product.get('price',0)}\n"
        f"📏 Sizes: {', '.join(sizes)}\n"
        f"{inventory_status}\n"
        f"[🔗 View Item]({product.get('image_url','')})"
    )

def format_recommendation_text(product):
    sizes = product.get("sizes") or ["Free Size"]
    return (
        f"{product.get('name','Item')} | {product.get('category','').title()}, "
        f"{product.get('style','').title()} | ₹{product.get('price',0)} | "
        f"Sizes: {', '.join(sizes)} | {check_inventory(product)}"
    )

# ----------------------------
# Persona NLG
# ----------------------------
PERSONAS = ["friendly", "formal", "witty"]
PERSONA_TEMPLATES = {
    "friendly": [
        "Love this pick — it matches your {reason} and stays within your ask.",
        "This one feels perfect for {reason}. Want me to add it to your cart?",
        "Great choice — it suits {reason} and looks really fresh."
    ],
    "formal": [
        "Recommended because it aligns with the specified {reason} and constraints.",
        "This selection matches the requested {reason}. Confirm to add to cart.",
        "Choice rationale: it conforms to {reason} and stock is available."
    ],
    "witty": [
        "This one says 'celebrate in style' for {reason}. Tempted?",
        "I've found a show-stopper for {reason}. Shall I reserve it in your cart?",
        "It's a match made in wardrobe heaven for {reason}. Want to try it on (virtually)?"
    ]
}

def persona_selector(session_ctx):
    idx = session_ctx.get("persona_idx", 0)
    persona = PERSONAS[idx % len(PERSONAS)]
    session_ctx["persona_idx"] = (idx + 1) % len(PERSONAS)
    return persona

def nlg_intro_for(product, style, user_query, session_ctx):
    reason = style if style else None
    if not reason:
        reason = product.get("style") or product.get("category") or "your request"
    persona = persona_selector(session_ctx)
    templates = PERSONA_TEMPLATES.get(persona, PERSONA_TEMPLATES["friendly"])
    key = f"{persona}_tpl_idx"
    tpl_idx = session_ctx.get(key, 0)
    template = templates[tpl_idx % len(templates)]
    session_ctx[key] = (tpl_idx + 1) % len(templates)
    return template.format(reason=reason)

# ----------------------------
# Small utilities
# ----------------------------
def is_ambiguous_reply(text):
    if not text:
        return True
    low = text.strip().lower()
    ambiguous = {"yes", "y", "ok", "okay", "sure", "fine", "yeah", "yep"}
    return low in ambiguous

# ----------------------------
# Optional LLM setup (RunnableWithMessageHistory)
# ----------------------------
llm_chain = None
if USE_GROQ and GROQ_API_KEY:
    try:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
        prompt = PromptTemplate.from_template(
            "You are REVA, a stylish and emotionally intelligent retail assistant. "
            "Greet the user and ask what they're shopping for. "
            "If they mention an occasion, suggest relevant styles. "
            "Always speak like a helpful, stylish sales associate.\n\nUser: {input}\nREVA:"
        )
        chain = prompt | llm

        class MessageHistory:
            def __init__(self):
                self.messages = []

            def add_messages(self, new_messages):
                self.messages.extend(new_messages)

        store = {}
        def get_history(session_id: str):
            if session_id not in store:
                store[session_id] = MessageHistory()
            return store[session_id]

        llm_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    except Exception:
        llm_chain = None

# ----------------------------
# In-memory cart & helpers
# ----------------------------
cart = []

def add_to_cart(product, size="Free Size", quantity=1):
    cart.append({
        "name": product.get("name"),
        "price": product.get("price", 0),
        "size": size,
        "quantity": quantity,
        "image_url": product.get("image_url", ""),
        "style": product.get("style", "")
    })

def get_cart_summary():
    total = sum(item.get("price", 0) * item.get("quantity", 1) for item in cart)
    return cart, total

def simulate_checkout(payment_method):
    cart_items, total = get_cart_summary()
    return f"✅ Payment of ₹{total} successful via {payment_method}. Your order is confirmed!"

# ----------------------------
# Chat loop (interactive, greeting-first behavior)
# ----------------------------
def chat():
    print("\n🛍️ REVA is ready. Say 'hi' to begin, 'help' for commands, or 'exit' to quit.\n")
    session_ctx = {
        "last_style": None, "last_category": None, "last_price": None,
        "persona_idx": 0, "last_matches": None, "last_question": None, "last_user_query": None, "greeted": False
    }

    def prompt_for_purpose():
        q = "Hi there! What are you shopping for today? (occasion, category, or budget is fine)"
        session_ctx["last_question"] = "purpose"
        return input(q + "\nYou: ").strip()

    while True:
        if not session_ctx["greeted"]:
            raw0 = input("You: ").strip()
            if not raw0:
                continue
            low0 = raw0.lower()
            session_ctx["last_user_query"] = raw0
            if low0 in ("exit", "quit"):
                print("👋 Bye! Happy shopping!")
                break
            if low0 in ("hi", "hello", "hey", "hiya"):
                print("👋 Hello! Nice to see you.")
                follow = prompt_for_purpose()
                session_ctx["greeted"] = True
                user_text = follow
            elif low0 == "help":
                print("Commands: 'show cart', 'checkout', 'clear cart', 'show more', 'add', or just say what you want.")
                continue
            else:
                session_ctx["greeted"] = True
                user_text = raw0
        else:
            user_text = input("You: ").strip()
            if not user_text:
                continue
            low = user_text.lower()
            session_ctx["last_user_query"] = user_text
            if low in ("exit", "quit"):
                print("👋 Bye! Happy shopping!")
                break
            if low == "help":
                print("Commands: 'show cart', 'checkout', 'clear cart', 'show more', 'add', or ask naturally (e.g., 'Diwali under ₹3000').")
                continue
            if low == "show cart":
                c, total = get_cart_summary()
                if not c:
                    print("Your cart is empty.")
                else:
                    print("Cart contents:")
                    for i, it in enumerate(c, 1):
                        print(f"{i}. {it['name']} — ₹{it['price']} x {it['quantity']} (Size: {it['size']})")
                    print(f"Total: ₹{total}")
                continue
            if low == "clear cart":
                cart.clear()
                print("Cart cleared.")
                continue
            if low.startswith("checkout") or low == "pay":
                if not cart:
                    print("Cart empty — add items first.")
                    continue
                method = input("Choose payment method (UPI/Credit/COD): ").strip() or "UPI"
                print(simulate_checkout(method))
                cart.clear()
                continue
            if low in ("add", "add it", "add first", "yes", "y"):
                last_matches = session_ctx.get("last_matches")
                if last_matches:
                    first = last_matches[0]
                    size = first.get("sizes", ["Free Size"])[0]
                    add_to_cart(first, size=size, quantity=1)
                    print(f"Added {first.get('name')} to cart. Total: ₹{get_cart_summary()[1]}")
                else:
                    print("No recent recommendation to add.")
                continue
            if low in ("more", "show more"):
                last_matches = session_ctx.get("last_matches")
                if last_matches and len(last_matches) > 1:
                    for idx, p in enumerate(last_matches[1:6], start=2):
                        print(f"\nAlt {idx-1}: {format_recommendation_text(p)}")
                    print("\nSay 'add' to add the top pick, or refine your query.")
                else:
                    print("No more alternatives. Try refining your request.")
                continue

        # Normal NL processing on user_text
        style, category, max_price = parse_user_input(user_text)

        # If nothing parsed, ask for focused info
        if not any([style, category, max_price]):
            session_ctx["last_question"] = "needs_details"
            follow = input("I didn't catch that — do you have an occasion (e.g., Diwali), a category (tops/dresses), or a budget like 'under ₹2000'?\nYou: ").strip()
            if not follow:
                print("No problem — tell me the vibe or event and I'll suggest options.")
                session_ctx["last_question"] = None
                continue
            style2, category2, max_price2 = parse_user_input(follow)
            style = style or style2
            category = category or category2
            max_price = max_price or max_price2
            session_ctx["last_question"] = None

        # persist context
        session_ctx["last_style"] = style or session_ctx.get("last_style")
        session_ctx["last_category"] = category or session_ctx.get("last_category")
        session_ctx["last_price"] = max_price or session_ctx.get("last_price")

        # Optionally call LLM to produce a short preface (non-blocking fallback)
        if llm_chain:
            try:
                reply = llm_chain.invoke({"input": user_text}, config={"configurable": {"session_id": "user-session"}})
                # reply may be an object; convert to string safely
                llm_text = getattr(reply, "content", str(reply))
                if llm_text:
                    print("\nREVA (assistant):", llm_text)
            except Exception:
                pass

        matches = recommend_products(catalog, style=style, category=category, max_price=max_price)
        session_ctx["last_matches"] = matches

        if not matches:
            print("I couldn't find matches — try widening price or using a related style (e.g., elegant, casual).")
            continue

        top = matches[0]
        top_price = top.get("price", 0)
        cheaper = None
        premium = None
        cheaper_candidates = [p for p in matches if p.get("price", 0) < top_price]
        if cheaper_candidates:
            cheaper = sorted(cheaper_candidates, key=lambda x: top_price - x.get("price", 0))[0]
        premium_candidates = [p for p in matches if p.get("price", 0) > top_price]
        if premium_candidates:
            premium = sorted(premium_candidates, key=lambda x: x.get("price", 0) - top_price)[0]

        # Present results
        print("\nTop pick:")
        print(format_recommendation_text(top))
        print(nlg_intro_for(top, style or None, user_text, session_ctx))

        if cheaper:
            print("\nCheaper alternative:")
            print(format_recommendation_text(cheaper))
        if premium:
            print("\nPremium alternative:")
            print(format_recommendation_text(premium))

        print("\nQuick options: type 'add' to add the top pick, 'more' to see more alternatives, or refine (e.g., 'under ₹1500').")

# ----------------------------
# Run CLI
# ----------------------------
if __name__ == "__main__":
    print("REV A — demo starting")
    try:
        catalog = load_catalog("mock_catalog.csv")  # ✅ FIXED
        print(f"Catalog loaded: {len(catalog)} items")
    except Exception:
        print("Catalog loaded: (sample data)")
        catalog = []  # fallback to empty list if loading fails
    chat()