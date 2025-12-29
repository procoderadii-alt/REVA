# reva_phase2_integrated.py
import os
import re
import json
import random
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# -------------------------------
# Catalog Loader
# -------------------------------
def load_catalog_from_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load product catalog from a CSV file and normalize fields.
    """
    df = pd.read_csv(path)
    catalog = []
    for _, row in df.iterrows():
        catalog.append({
            "id": row.get("id"),
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "brand": row.get("brand"),
            "style": row.get("style"),
            "gender": row.get("gender"),
            "color": row.get("color"),
            "material": row.get("material"),
            "season": row.get("season"),
            "size": row.get("size"),
            "price": float(row.get("price_inr", 0)),   # INR price for scoring
            "currency": row.get("currency", "INR"),
            "stock": int(row.get("stock", 0)),
            "rating": float(row.get("rating", 0)),
            "num_reviews": int(row.get("num_reviews", 0)),
            "sku": row.get("sku"),
            # image_urls stored as JSON string in CSV
            "image_urls": (
                json.loads(row["image_urls"]) if isinstance(row.get("image_urls"), str) else []
            ),
            # sizes stored as comma‑separated string
            "sizes": (
                row["sizes"].split(",") if isinstance(row.get("sizes"), str) else []
            ),
            "name": row.get("name"),
        })
    return catalog

# -------------------------------
# Load catalog once at startup
# -------------------------------
CATALOG_PATH = r"C:\Users\Arnav\OneDrive\Desktop\REVA\reva_mvp\Prototypes\products.csv"
catalog: List[Dict[str, Any]] = load_catalog_from_csv(CATALOG_PATH)

# -------------------------------
# Scoring Function
# -------------------------------
def score_item(item: Dict[str, Any],
               occasion: Optional[str] = None,
               category: Optional[str] = None,
               max_price: Optional[float] = None,
               affinity_tags: Optional[List[str]] = None) -> int:
    """
    Score an item based on category, occasion, budget, and affinity tags.
    """
    score = 0
    # Category match
    if category and item.get("category", "").lower() == category.lower():
        score += 3
    # Occasion/style match
    if occasion and occasion.lower() in str(item.get("style", "")).lower():
        score += 2
    # Budget match
    if max_price and item.get("price", 0) <= max_price:
        score += 2
    else:
        score -= 1
    # Affinity tags boost
    if affinity_tags:
        for tag in affinity_tags:
            if tag.lower() in str(item.get("style", "")).lower():
                score += 1
    return score
# ----------------------------
# Supabase client setup
# ----------------------------
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None
SUPABASE_URL = os.getenv("https://soxmqvjgxlfiqazjerhs.supabase.co", "")
SUPABASE_KEY = os.getenv("sb_secret_AXke-Yney_Twy5S_T3uw8w_CacfdkOP", "")
supabase: Optional[Any] = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️ Supabase init failed: {e}")
else:
    print("⚠️ Supabase env vars missing or client unavailable. Context will be in-memory.")
# ----------------------------
# Global config & state
# ----------------------------
IN_MEMORY_SESSIONS: Dict[str, Dict[str, Any]] = {}

LOYALTY_POINTS_PER_RS = 0.1   # accrual: 1 point per ₹10 spent equivalent
LOYALTY_VALUE_PER_POINT = 10  # redemption: 1 point = ₹10

CATEGORY_DISCOUNT = {
    "ethnic_wear": 0.10,
    "footwear": 0.08,
    "accessories": 0.12,
}

BUNDLE_DISCOUNT = 0.15
FREE_SHIPPING_THRESHOLD = 3000
DEFAULT_SHIPPING = 99

# Personas & tone templates
PERSONAS = ["friendly", "formal", "witty"]
PERSONA_TEMPLATES = {
    "friendly": {
        "reco_intro": "Love this vibe — it fits your {occasion_or_style}.",
        "budget": "If you want to save a bit, this one’s a smart pick.",
        "premium": "Or go premium — you’ll turn heads.",
        "nudge_shipping": "You’re ₹{delta} away from free shipping — add {suggestion}?",
        "nudge_bundle": "Add the bundle for extra savings (Save ₹{savings}).",
        "checkout": "Looks great! Ready to pay?",
        "frustrated": "Got you — let’s make this easy and affordable.",
        "price_sensitive": "Let’s focus on budget-friendly picks.",
    },
    "formal": {
        "reco_intro": "Recommended for your {occasion_or_style}.",
        "budget": "Alternative within budget constraints.",
        "premium": "Premium option with superior finish.",
        "nudge_shipping": "₹{delta} remaining for free shipping—consider {suggestion}.",
        "nudge_bundle": "Bundle available: Save ₹{savings}.",
        "checkout": "Proceed to payment when ready.",
        "frustrated": "Acknowledged. Presenting cost-optimized selections.",
        "price_sensitive": "Highlighting economical options.",
    },
    "witty": {
        "reco_intro": "This one screams your {occasion_or_style} energy.",
        "budget": "Wallet-friendly but high on style.",
        "premium": "Treat yourself — premium swagger unlocked.",
        "nudge_shipping": "₹{delta} shy of free shipping — toss in {suggestion}!",
        "nudge_bundle": "Bundle magic: Save ₹{savings}.",
        "checkout": "Shall we seal the deal?",
        "frustrated": "Let’s kill the chaos — simple, sharp picks now.",
        "price_sensitive": "Let’s keep the drip without the dip in wallet.",
    }
}

# ----------------------------
# Context Agent (Supabase + in-memory fallback)
# ----------------------------
import os
from typing import Any, Optional
from datetime import datetime, timezone

try:
    from supabase import create_client
except ImportError:
    create_client = None

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

supabase: Optional[Any] = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️ Supabase init failed: {e}")
else:
    print("⚠️ Supabase env vars missing or client unavailable. Context will be in-memory.")

# In-memory fallback store
IN_MEMORY_SESSIONS: dict[str, dict[str, Any]] = {}

def save_session(user_id: str, channel: str, context: dict[str, Any]) -> None:
    # Use timezone-aware UTC datetime
    context["updated_at"] = datetime.now(timezone.utc).isoformat()
    if supabase:
        payload = {
            "user_id": user_id,
            "channel": channel,
            "chat_history": context.get("chat_history", []),
            "cart_state": context.get("cart_state", []),
            "preferences": context.get("preferences", {}),
            "intent": context.get("intent", {}),
            "promotions": context.get("promotions", {}),
            "metrics": context.get("metrics", {}),
            "updated_at": context["updated_at"],
        }
        supabase.table("user_sessions").upsert(payload).execute()
    else:
        IN_MEMORY_SESSIONS[user_id] = {"channel": channel, **context}

def load_session(user_id: str, channel: str) -> dict[str, Any]:
    if supabase:
        resp = supabase.table("user_sessions").select("*").eq("user_id", user_id).execute()
        data = resp.data[0] if resp.data else None
        if data:
            return {
                "chat_history": data.get("chat_history", []),
                "cart_state": data.get("cart_state", []),
                "preferences": data.get("preferences", {}),
                "intent": data.get("intent", {}),
                "promotions": data.get("promotions", {}),
                "metrics": data.get("metrics", {}),
                "channel": data.get("channel", channel),
            }
    # fallback
    return IN_MEMORY_SESSIONS.get(user_id, {
        "chat_history": [],
        "cart_state": [],
        "preferences": {},
        "intent": {},
        "promotions": {},
        "metrics": {},
        "channel": channel,
    })
# ----------------------------
# Intent Agent (deterministic v1)
# ----------------------------
OCCASION_KEYWORDS = {
    "diwali": "diwali","wedding":"wedding","interview":"interview","office":"office",
    "party":"party","beach":"beach","brunch":"brunch","gym":"gym",
    "festival":"festival","date":"date","winter":"winter","summer":"summer",
    "casual":"casual","formal":"formal",
}

SIZE_MAP = {
    "xs":"XS","s":"S","small":"S","m":"M","medium":"M",
    "l":"L","large":"L","xl":"XL","xxl":"XXL","2xl":"XXL",
}

CATEGORY_KEYWORDS = [
    "kurta","saree","lehenga","sherwani","juttis","dupatta",
    "shirt","blazer","trousers","tie","formal shoes",
    "dress","heels","clutch","jewelry","coat","sweater",
    "belt","sneakers","sandals","tshirt","shorts","hoodie",
    "outerwear","tops","accessories","backpack","watch","scarf",
]

def parse_intent(user_text: str) -> Dict[str, Any]:
    low = user_text.lower()
    intent = "browse"
    if re.search(r"\b(add|put|pop|buy)\b", low):
        intent = "cart_add"
    elif re.search(r"\b(cart|show cart|view cart)\b", low):
        intent = "cart_review"
    elif re.search(r"\b(checkout|pay|place order)\b", low):
        intent = "checkout"
    elif re.search(r"\bclear cart\b", low):
        intent = "clear_cart"
    elif re.search(r"\bsize|sizes\b", low):
        intent = "sizes_query"
    elif re.search(r"\bchannel|switch\b", low):
        intent = "handoff"

    occasion = next((v for k,v in OCCASION_KEYWORDS.items() if k in low), None)
    category = next((c for c in CATEGORY_KEYWORDS if c in low), None)

    max_price = None
    m = re.search(r"(?:under|below|less than|upto|up to)\s*₹?\s*(\d{3,6})", low) or re.search(r"₹\s*(\d{3,6})", low)
    if m:
        try: max_price = int(m.group(1))
        except: max_price = None

    size = None
    for k,v in SIZE_MAP.items():
        if re.search(rf"\b{k}\b", low):
            size = v; break

    bundle_request = []
    if occasion == "diwali":
        bundle_request = ["kurta","juttis","dupatta"]
    elif occasion == "wedding":
        bundle_request = ["lehenga","jewelry","clutch"]
    elif occasion == "beach":
        bundle_request = ["swimsuit","cover-up","sandals"]

    sentiment = "neutral"
    if re.search(r"\bgreat|love|perfect|nice|cool|awesome\b", low): sentiment = "positive"
    elif re.search(r"\bexpensive|costly|bad|hate|annoyed|frustrated\b", low): sentiment = "frustrated"
    elif re.search(r"\bcheap|budget|low cost|affordable\b", low): sentiment = "price_sensitive"

    signals = 0
    for s in [occasion, category, max_price, size]:
        if s: signals += 1
    confidence = min(0.95, 0.5 + 0.12 * signals)

    return {
        "intent": intent,
        "occasion": occasion,
        "category": category,
        "max_price": max_price,
        "size": size,
        "bundle_request": bundle_request,
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
    }

# ----------------------------
# Recommendation Agent + Proactive Intelligence
# ----------------------------
def stylist_tip(occasion: Optional[str], style: Optional[str]) -> str:
    tips = {
        "diwali": "Choose elegant ethnic wear with festive accents — perfect for evening celebrations.",
        "wedding": "Premium ethnic with polished accessories helps you stand out.",
        "interview": "Crisp shirts and blazers — confidence starts with sharp dressing.",
        "office": "Professional yet comfortable — neutral colors and fitted silhouettes.",
        "party": "Bold colors or sequins — elevate with minimal accessories.",
        "beach": "Light fabrics and sunglasses — stay cool and stylish.",
        "brunch": "Casual chic with flats and a neat handbag.",
        "gym": "Breathable activewear — functionality first.",
        "festival": "Vibrant ethnic wear — jewelry completes the vibe.",
        "date": "Elegant with subtle accessories — add a touch of perfume.",
        "winter": "Layer up with coats and scarves for warmth and style.",
        "summer": "Light cottons — keep it breezy and fresh.",
        "casual": "Effortless basics — one statement piece does the trick.",
        "formal": "Classic shirts, trousers, blazers — timeless and sharp.",
    }
    return tips.get(occasion or "", f"Lean into your {style or 'personal'} vibe and accessorize smartly.")

def score_item(item, occasion=None, category=None, max_price=None, affinity_tags=None):
    """Scoring function to rank catalog items by relevance."""
    score = 0
    # Category match
    if category and item.get("category","").lower() == category.lower():
        score += 3
    # Occasion/style match
    if occasion and occasion.lower() in item.get("style","").lower():
        score += 2
    # Budget match
    if max_price and item.get("price",0) <= max_price:
        score += 2
    else:
        score -= 1
    # Affinity tags boost
    if affinity_tags:
        for tag in affinity_tags:
            if tag.lower() in item.get("tags","").lower():
                score += 1
    return score

def get_stylist_tip(intent):
    """Dynamic stylist tips based on occasion."""
    occasion = intent.get("occasion","").lower()
    if "office" in occasion:
        return "🧠 Professional yet comfortable — neutral colors and fitted silhouettes."
    elif "party" in occasion:
        return "🧠 Bold colors or sequins — elevate with minimal accessories."
    elif "casual" in occasion:
        return "🧠 Keep it relaxed — breathable fabrics and versatile basics."
    else:
        return "🧠 Accessorize smartly to match your vibe."

def recommend(catalog: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal recommendation engine:
    - Scores each item in the catalog using score_item
    - Returns the top 3 items with their details
    """
    occasion = intent.get("occasion")
    category = intent.get("category")
    max_price = intent.get("max_price")
    affinity_tags = intent.get("affinity_tags", [])

    scored_items = []
    for item in catalog:
        s = score_item(item, occasion, category, max_price, affinity_tags)
        scored_items.append((s, item))

    # Sort by score descending
    scored_items.sort(key=lambda x: x[0], reverse=True)

    # Take top 3
    top_items = [itm for _, itm in scored_items[:3]]

    # Build a rendered string for chat
    rendered_lines = []
    for i, itm in enumerate(top_items, start=1):
        rendered_lines.append(
            f"{i}. {itm['name']} — ₹{int(itm['price'])} ({itm['brand']}, {itm['color']}, Size: {itm.get('size')})"
        )
    rendered_text = "✨ Recommended for you:\n" + "\n".join(rendered_lines) if top_items else "No matches found."

    return {
        "items": top_items,
        "rendered": rendered_text
    }


def find_bundle_items(catalog: List[Dict[str, Any]], occasion: Optional[str]) -> List[Dict[str, Any]]:
    bundle_map = {
        "diwali": ["kurta","juttis","dupatta"],
        "wedding": ["lehenga","jewelry","clutch"],
        "beach": ["swimsuit","cover-up","sandals"],
        "office": ["shirt","trousers","blazer"],
        "party": ["dress","heels","clutch"],
    }
    if not occasion or occasion not in bundle_map:
        return []
    result = []
    for kw in bundle_map[occasion]:
        matches = [p for p in catalog if kw in (p.get("subcategory","").lower() + " " + p.get("category","").lower()) and p.get("stock",0) > 0]
        if matches:
            pick = sorted(matches, key=lambda x: x.get("price", 0))[0]
            result.append(pick)
    return result
import random

FREE_SHIPPING_THRESHOLD = 3000  # adjust as needed

def score_item(item, occasion=None, category=None, max_price=None):
    """Scoring function to rank catalog items by relevance."""
    score = 0
    if category and item.get("category","").lower() == category.lower():
        score += 3
    if occasion and occasion.lower() in item.get("style","").lower():
        score += 2
    if max_price and item.get("price",0) <= max_price:
        score += 2
    else:
        score -= 1
    return score

def get_stylist_tip(intent):
    """Dynamic stylist tips based on occasion."""
    occasion = intent.get("occasion","").lower()
    if "office" in occasion:
        return "🧠 Professional yet comfortable — neutral colors and fitted silhouettes."
    elif "party" in occasion:
        return "🧠 Bold colors or sequins — elevate with minimal accessories."
    elif "casual" in occasion:
        return "🧠 Keep it relaxed — breathable fabrics and versatile basics."
    else:
        return "🧠 Accessorize smartly to match your vibe."

def recommend_items(catalog, intent, context):
    """Main recommender block."""
    occasion = intent.get("occasion")
    category = intent.get("category")
    max_price = intent.get("budget")

    # Score and sort catalog
    scored = sorted(
        catalog,
        key=lambda x: score_item(x, occasion, category, max_price),
        reverse=True
    )

    # Pick top 3
    primary = scored[0] if scored else None
    budget = next((i for i in scored if max_price and i["price"] <= max_price), None)
    premium = next((i for i in scored if i["price"] > (max_price or 0)), None)

    # Build response
    lines = []
    if primary:
        lines.append(f"*{primary['name']}* ✨\n₹{primary['price']}\n({primary['category']})\nSizes: {primary.get('sizes','')} | Stock: {primary.get('stock','')}")
    if budget:
        lines.append(f"💸 Budget pick: *{budget['name']}* — ₹{budget['price']}")
    if premium:
        lines.append(f"💎 Premium option: *{premium['name']}* — ₹{premium['price']}")

    # Free shipping nudge
    subtotal = sum(item["price"] for item in context.get("cart_state",[]))
    if subtotal > 0 and subtotal < FREE_SHIPPING_THRESHOLD:
        remaining = FREE_SHIPPING_THRESHOLD - subtotal
        lines.append(f"🚚 You’re ₹{remaining} away from free shipping.")

    # Stylist tip
    lines.append(get_stylist_tip(intent))

    return {"rendered": "\n".join(lines)}

def recommend(catalog: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
    occasion = intent.get("occasion")
    category = intent.get("category")
    max_price = intent.get("max_price")
    affinity_tags = [t for t in [occasion, category] if t]

    candidates = []
    for item in catalog:
        if category:
            joined = (item.get("category","") + " " + item.get("subcategory","")).lower()
            if category.lower() not in joined:
                continue
        if int(item.get("stock", 0)) <= 0:
            continue
        s = score_item(item, occasion, category, max_price, affinity_tags)
        candidates.append((s, item))

    if not candidates:
        fallback = sorted([it for it in catalog if it.get("stock", 0) > 0], key=lambda x: (-(x.get("stock",0)), x.get("price",0)))
        candidates = [(score_item(it, occasion, category, max_price, affinity_tags), it) for it in fallback[:6]]

    candidates.sort(key=lambda x: (-x[0], x[1].get("price", 0)))

    primary = candidates[0][1] if candidates else None
    cheaper = [it for s,it in candidates if primary and it.get("price",0) < primary.get("price",0)]
    budget_alt = cheaper[0] if cheaper else (candidates[1][1] if len(candidates) > 1 else None)
    premium = [it for s,it in candidates if primary and it.get("price",0) > primary.get("price",0)]
    premium_alt = premium[0] if premium else (candidates[1][1] if len(candidates) > 1 else None)

    bundle_items = find_bundle_items(catalog, occasion)

    def brief(it):
        if not it: return None
        return {
            "item_id": str(it.get("id")),
            "name": it.get("name"),
            "category": it.get("category"),
            "subcategory": it.get("subcategory"),
            "style": it.get("style",""),
            "price": float(it.get("price", 0)),
            "score": round(score_item(it, occasion, category, max_price, affinity_tags), 2),
            "sizes": it.get("sizes", []),
            "stock": it.get("stock", 0),
        }

    bundle_total = sum(b.get("price", 0) for b in bundle_items)
    bundle_final = round(bundle_total * (1 - BUNDLE_DISCOUNT), 2) if len(bundle_items) >= 3 else bundle_total
    savings = round(bundle_total - bundle_final, 2)

    # stylist tip for primary
    tip = stylist_tip(occasion, (primary or {}).get("style"))

    return {
        "primary": brief(primary),
        "budget_alt": brief(budget_alt),
        "premium_alt": brief(premium_alt),
        "bundle": {
            "items": [brief(b) for b in bundle_items],
            "total": round(bundle_total, 2),
            "final": round(bundle_final, 2),
            "savings": round(savings, 2),
        },
        "tip": tip
    }

# ----------------------------
# Pricing Agent
# ----------------------------
def map_category_for_discount(item: Dict[str, Any]) -> Optional[str]:
    cat = (item.get("category") or "").lower()
    sub = (item.get("subcategory") or "").lower()
    if any(k in (cat + " " + sub) for k in ["kurta","saree","lehenga","sherwani","dupatta","ethnic"]):
        return "ethnic_wear"
    if any(k in (cat + " " + sub) for k in ["juttis","heels","shoes","sneakers","sandals","footwear"]):
        return "footwear"
    if any(k in (cat + " " + sub) for k in ["belt","jewelry","clutch","accessories","watch","scarf"]):
        return "accessories"
    return None

def calculate_pricing(cart_items: List[Dict[str, Any]], loyalty_points_available: int = 0) -> Dict[str, Any]:
    subtotal = sum(float(it.get("price", 0)) * int(it.get("quantity", 1)) for it in cart_items)

    category_discounts_value = 0.0
    for it in cart_items:
        cat_key = map_category_for_discount(it)
        if cat_key and cat_key in CATEGORY_DISCOUNT:
            category_discounts_value += float(it.get("price",0)) * int(it.get("quantity",1)) * CATEGORY_DISCOUNT[cat_key]

    bundle_discount_value = 0.0
    if len(cart_items) >= 3:
        bundle_discount_value = subtotal * BUNDLE_DISCOUNT

    shipping = 0 if subtotal >= FREE_SHIPPING_THRESHOLD else DEFAULT_SHIPPING

    max_loyalty_redeem_rs = loyalty_points_available * LOYALTY_VALUE_PER_POINT
    discounts_total_before_loyalty = category_discounts_value + bundle_discount_value
    preliminary_total = subtotal - discounts_total_before_loyalty + shipping
    loyalty_redeem_rs = min(max_loyalty_redeem_rs, preliminary_total)

    final = max(0.0, preliminary_total - loyalty_redeem_rs)

    return {
        "subtotal": round(subtotal, 2),
        "discounts": round(-(category_discounts_value + bundle_discount_value), 2),
        "loyalty": round(-loyalty_redeem_rs, 2),
        "shipping": round(shipping, 2),
        "final": round(final, 2),
        "savings": round(category_discounts_value + bundle_discount_value + loyalty_redeem_rs, 2),
        "applied": {
            "category_discounts": round(category_discounts_value, 2),
            "bundle_discount": round(bundle_discount_value, 2),
            "loyalty_points_used": int(loyalty_redeem_rs / LOYALTY_VALUE_PER_POINT)
        }
    }

# ----------------------------
# Proactive nudges & persona tone
# ----------------------------
def persona_selector(context: Dict[str, Any]) -> str:
    idx = context.setdefault("persona_idx", 0)
    persona = PERSONAS[idx % len(PERSONAS)]
    context["persona_idx"] = (idx + 1) % len(PERSONAS)
    return persona

def tone_line(persona: str, key: str, **kwargs) -> str:
    tpl = PERSONA_TEMPLATES.get(persona, PERSONA_TEMPLATES["friendly"]).get(key, "")
    return tpl.format(**kwargs)

def make_shipping_nudge(subtotal: float, persona: str, occasion: Optional[str], bundle: Dict[str, Any]) -> Optional[str]:
    if subtotal >= FREE_SHIPPING_THRESHOLD:
        return None
    delta = int(FREE_SHIPPING_THRESHOLD - subtotal)
    suggestion = None
    # Prefer cheapest item from bundle if exists
    items = [i for i in (bundle.get("items") or []) if i]
    if items:
        cheapest = sorted(items, key=lambda x: x.get("price", 0))[0]
        suggestion = cheapest.get("name")
    else:
        # generic suggestion based on occasion
        occ_map = {"diwali":"dupatta", "wedding":"clutch", "beach":"sandals", "office":"tie", "party":"jewelry"}
        suggestion = occ_map.get(occasion, "accessory")
    return tone_line(persona, "nudge_shipping", delta=delta, suggestion=suggestion)

def make_bundle_nudge(persona: str, bundle: Dict[str, Any]) -> Optional[str]:
    if not bundle.get("items"):
        return None
    savings = int(bundle.get("savings", 0))
    if savings <= 0:
        return None
    return tone_line(persona, "nudge_bundle", savings=savings)

def sentiment_preface(persona: str, sentiment: str) -> Optional[str]:
    if sentiment == "frustrated":
        return tone_line(persona, "frustrated")
    if sentiment == "price_sensitive":
        return tone_line(persona, "price_sensitive")
    return None

# ----------------------------
# UX Agent (channel-optimized with tips & nudges)
# ----------------------------
def format_whatsapp(reco: Dict[str, Any], persona: str, intent: Dict[str, Any], cart_subtotal: float) -> str:
    occ_or_style = intent.get("occasion") or reco["primary"].get("style") if reco.get("primary") else "your style"
    intro = tone_line(persona, "reco_intro", occasion_or_style=occ_or_style)
    tip_line = f"🧠 Stylist tip: {reco['tip']}"
    msg = [intro]

    def line(it):
        sizes = ", ".join(it.get("sizes", []) or [])
        stock = it.get("stock", 0)
        return f"*{it['name']}* ✨\n₹{int(it['price'])}\n({it['subcategory'] or it['category']})\nSizes: {sizes or 'Free'} | Stock: {stock}\n"

    if reco["primary"]:
        msg.append(line(reco["primary"]))
        msg.append("1️⃣ Add | 2️⃣ Size M | 3️⃣ Next | 4️⃣ Bundle")

    if reco["budget_alt"]:
        msg.append("💸 " + tone_line(persona, "budget"))
        msg.append(line(reco["budget_alt"]))
    if reco["premium_alt"]:
        msg.append("💎 " + tone_line(persona, "premium"))
        msg.append(line(reco["premium_alt"]))

    if reco["bundle"]["items"]:
        b = reco["bundle"]
        items = "\n".join([f"- {i['name']} ₹{int(i['price'])}" for i in b["items"] if i])
        msg.append(f"🧩 Bundle:\n{items}\nTotal: ₹{int(b['total'])} | Final: ₹{int(b['final'])} (Save ₹{int(b['savings'])})")

    # Nudges
    shipping_nudge = make_shipping_nudge(cart_subtotal, persona, intent.get("occasion"), reco["bundle"])
    bundle_nudge = make_bundle_nudge(persona, reco["bundle"])
    if shipping_nudge: msg.append("🚚 " + shipping_nudge)
    if bundle_nudge: msg.append("🎁 " + bundle_nudge)

    # Sentiment-aware preface
    preface = sentiment_preface(persona, intent.get("sentiment","neutral"))
    if preface: msg.insert(0, preface)

    msg.append(tip_line)
    return "\n".join(msg)

def format_web(reco: Dict[str, Any], persona: str, intent: Dict[str, Any], cart_subtotal: float) -> str:
    tip_line = f"<p class='tip'>🧠 {reco['tip']}</p>"
    def card(it):
        if not it: return ""
        sizes = ", ".join(it.get("sizes", []) or [])
        return f"""
<div class="product-card">
  <h3>{it['name']}</h3>
  <span class="price">₹{int(it['price'])}</span>
  <span class="meta">{it['subcategory'] or it['category']} | Sizes: {sizes}</span>
  <button data-item="{it['item_id']}">Add to Cart</button>
</div>
"""
    html = "<div class='reco-grid'>\n"
    html += card(reco["primary"])
    html += "<h4>Budget alternative</h4>" + card(reco["budget_alt"])
    html += "<h4>Premium alternative</h4>" + card(reco["premium_alt"])
    if reco["bundle"]["items"]:
        html += "<div class='bundle'>\n<h4>Recommended Bundle</h4>\n"
        for i in reco["bundle"]["items"]:
            html += card(i)
        html += f"<p>Total: ₹{int(reco['bundle']['total'])} | Final: ₹{int(reco['bundle']['final'])} (Save ₹{int(reco['bundle']['savings'])})</p>\n</div>"
    # Nudges
    shipping_nudge = make_shipping_nudge(cart_subtotal, persona, intent.get("occasion"), reco["bundle"])
    bundle_nudge = make_bundle_nudge(persona, reco["bundle"])
    if shipping_nudge: html += f"<p class='nudge shipping'>🚚 {shipping_nudge}</p>"
    if bundle_nudge: html += f"<p class='nudge bundle'>🎁 {bundle_nudge}</p>"
    html += tip_line
    html += "\n</div>"
    return html

def format_voice(reco: Dict[str, Any], persona: str, intent: Dict[str, Any], cart_subtotal: float) -> str:
    intro = tone_line(persona, "reco_intro", occasion_or_style=(intent.get("occasion") or (reco["primary"] or {}).get("style") or "your style"))
    b = reco["bundle"]
    speak = "<speak>"
    speak += intro + " "
    primary = reco["primary"]; budget = reco["budget_alt"]; premium = reco["premium_alt"]
    if primary: speak += f"{primary['name']} at {int(primary['price'])} rupees. Reply one to add. "
    if budget: speak += f"Budget pick: {budget['name']} at {int(budget['price'])}. "
    if premium: speak += f"Premium: {premium['name']} at {int(premium['price'])}. "
    if b["items"]:
        speak += "<break time=\"300ms\"/>"
        speak += f"Bundle available. Total {int(b['total'])}, final {int(b['final'])}, saving {int(b['savings'])}. "
    tip_line = f"Stylist tip: {reco['tip']}."
    speak += tip_line + " "
    shipping_nudge = make_shipping_nudge(cart_subtotal, persona, intent.get("occasion"), reco["bundle"])
    if shipping_nudge:
        speak += f"<break time=\"250ms\"/>{shipping_nudge} "
    speak += "</speak>"
    return speak

def format_by_channel(reco: Dict[str, Any], channel: str, persona: str, intent: Dict[str, Any], cart_subtotal: float) -> str:
    channel = (channel or "whatsapp").lower()
    if channel == "web":
        return format_web(reco, persona, intent, cart_subtotal)
    if channel == "voice":
        return format_voice(reco, persona, intent, cart_subtotal)
    return format_whatsapp(reco, persona, intent, cart_subtotal)

# ----------------------------
# Cart helpers & metrics
# ----------------------------
def add_to_cart(context: Dict[str, Any], item: Dict[str, Any], size: Optional[str] = None, quantity: int = 1) -> Dict[str, Any]:
    cart = context.setdefault("cart_state", [])
    entry = {
        "id": item.get("id"),
        "name": item.get("name"),
        "price": float(item.get("price", 0)),
        "size": size or (item.get("sizes", ["M"])[0]),
        "quantity": int(quantity),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "style": item.get("style", ""),
        "stock": int(item.get("stock", 0)),
    }
    cart.append(entry)
    # metrics: track upsell attempts baseline
    metrics = context.setdefault("metrics", {})
    metrics["items_added"] = metrics.get("items_added", 0) + 1
    return entry

def get_cart_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    cart = context.get("cart_state", [])
    subtotal = sum(float(it.get("price",0))*int(it.get("quantity",1)) for it in cart)
    return {"items": cart, "subtotal": subtotal}

def clear_cart(context: Dict[str, Any]) -> None:
    context["cart_state"] = []

def accrue_loyalty(cart_items: List[Dict[str, Any]]) -> int:
    spend = sum(float(it.get("price",0))*int(it.get("quantity",1)) for it in cart_items)
    return int(spend * LOYALTY_POINTS_PER_RS)

# ----------------------------
# Intent handlers
# ----------------------------
def resolve_reco_item_by_index(reco: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    order = ["primary", "budget_alt", "premium_alt"]
    if idx < 1 or idx > len(order): return None
    return reco.get(order[idx-1])

def find_catalog_item_by_id(catalog: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    return next((c for c in catalog if str(c.get("id")) == str(item_id)), None)

def handle_cart_add(context: Dict[str, Any], catalog: List[Dict[str, Any]], message: str, last_reco: Dict[str, Any]) -> str:
    low = message.lower()
    m_idx = re.search(r"add\s*(\d+)\b", low)
    m_id = re.search(r"(?:add|item)\s*(?:item\s*)?([A-Za-z0-9\-]+)", low)

    selected = None
    if m_idx:
        idx = int(m_idx.group(1))
        selected = resolve_reco_item_by_index(last_reco, idx)
    elif m_id:
        candidate_id = m_id.group(1)
        for key in ["primary","budget_alt","premium_alt"]:
            it = last_reco.get(key)
            if it and str(it.get("item_id")) == candidate_id:
                selected = it
                break
        if not selected:
            real = find_catalog_item_by_id(catalog, candidate_id)
            if real:
                selected = {
                    "item_id": str(real.get("id")),
                    "name": real.get("name"),
                    "category": real.get("category"),
                    "subcategory": real.get("subcategory"),
                    "price": float(real.get("price", 0)),
                    "sizes": real.get("sizes", []),
                    "stock": real.get("stock", 0),
                }

    if not selected:
        selected = last_reco.get("primary")
    if not selected:
        return "I couldn’t identify which item to add. Say 'add 1' for the first item or 'add item <id>'."

    pref_size = (context.get("preferences", {}).get("size")) or "M"
    real = find_catalog_item_by_id(catalog, selected["item_id"]) or next((c for c in catalog if c.get("name") == selected["name"]), None)
    if not real:
        return "That item isn’t available right now."

    entry = add_to_cart(context, real, size=pref_size, quantity=1)
    prefs = context.setdefault("preferences", {})
    if selected.get("sizes"):
        prefs["size_options"] = selected["sizes"]

    # metrics: upsell nudge shown?
    context.setdefault("metrics", {})["upsell_offers"] = context["metrics"].get("upsell_offers", 0) + 1

    save_session(context.get("user_id","demo"), context.get("channel","whatsapp"), context)
    return f"Added '{entry['name']}' (Size {entry['size']}) to your cart. Want to add the bundle for extra savings?"
def handle_user_message(user_id: str, channel: str, catalog: list[dict], user_text: str) -> dict[str, str]:
    context = load_session(user_id, channel)

    # --- SIZE intent ---
    size_match = re.search(r"size\s+([a-z0-9]+)", user_text.lower())
    if size_match:
        chosen_size = size_match.group(1).upper()
        context["preferences"]["size"] = chosen_size
        save_session(user_id, channel, context)
        return {"rendered": f"✅ Size set to {chosen_size}."}

    # --- REMOVE intent ---
    if "remove" in user_text.lower():
        tokens = user_text.lower().split()
        if len(tokens) > 1:
            target = " ".join(tokens[1:]).strip()
            removed_item = None
            for idx, entry in enumerate(context["cart_state"]):
                if target in entry["item"].lower():
                    removed_item = context["cart_state"].pop(idx)
                    break
            if removed_item:
                save_session(user_id, channel, context)
                return {"rendered": f"🗑️ Removed {removed_item['item']} (Size {removed_item.get('size','?')}) from your cart."}
            else:
                return {"rendered": f"⚠️ No cart item matched '{target}'."}
        else:
            if context["cart_state"]:
                removed = context["cart_state"].pop()
                save_session(user_id, channel, context)
                return {"rendered": f"🗑️ Removed {removed['item']} (Size {removed.get('size','?')}) from your cart."}
            else:
                return {"rendered": "🛒 Cart is already empty."}

    # --- ADD intent ---
    if "add" in user_text.lower():
        if "last_reco" in context:
            item = context["last_reco"]
            size = context["preferences"].get("size", "M")
            context["cart_state"].append({"item": item["name"], "price": item["price"], "size": size})
            save_session(user_id, channel, context)
            return {"rendered": f"Added '{item['name']}' (Size {size}) to your cart. Want to add the bundle for extra savings?"}
        else:
            return {"rendered": "⚠️ No item selected to add."}

    # --- CART intent ---
    if "cart" in user_text.lower():
        if not context["cart_state"]:
            return {"rendered": "🛒 Cart is empty."}
        lines = ["🛒 Cart:"]
        subtotal = 0
        for idx, entry in enumerate(context["cart_state"], 1):
            subtotal += entry["price"]
            lines.append(f"{idx}. {entry['item']} — ₹{entry['price']} x 1 (Size: {entry['size']})")
        lines.append(f"Subtotal: ₹{subtotal}")
        return {"rendered": "\n".join(lines)}

    # --- Recommendation intent ---
    # Parse simple intent from user_text (occasion, category, budget)
    intent = {}
    if "office" in user_text.lower():
        intent["occasion"] = "office"
    elif "party" in user_text.lower():
        intent["occasion"] = "party"
    elif "casual" in user_text.lower():
        intent["occasion"] = "casual"

    if "kurta" in user_text.lower():
        intent["category"] = "kurta"
    elif "shirt" in user_text.lower():
        intent["category"] = "shirt"
    elif "coat" in user_text.lower():
        intent["category"] = "coat"

    # Extract budget if mentioned
    budget_match = re.search(r"under\s*₹?(\d+)", user_text.lower())
    if budget_match:
        intent["budget"] = int(budget_match.group(1))

    # Call recommender
    reco = recommend_items(catalog, intent, context)
    if reco:
        # Save last recommendation for add intent
        if "rendered" in reco and catalog:
            context["last_reco"] = catalog[0]  # store top item for add
        save_session(user_id, channel, context)
        return reco

    # --- fallback ---
    return {"rendered": "🤔 I couldn’t find a good match. Try specifying occasion, category, or budget."}

def handle_sizes_query(context: Dict[str, Any], last_reco: Dict[str, Any]) -> str:
    cart = context.get("cart_state", [])
    target_sizes = None
    target_name = None

    if cart:
        last_item = cart[-1]
        target_sizes = last_item.get("sizes") or context.get("preferences", {}).get("size_options") or []
        target_name = last_item.get("name")
    if not target_sizes and last_reco and last_reco.get("primary"):
        target_sizes = last_reco["primary"].get("sizes") or []
        target_name = last_reco["primary"].get("name")

    if not target_sizes:
        return "I don’t have size info yet. Try 'show kurtas in size M' or pick an item first."

    sizes_str = ", ".join(target_sizes)
    return f"Available sizes for {target_name}: {sizes_str}. Say 'size M' to set your preferred size."

def handoff_message(prev_channel: str, new_channel: str, context: Dict[str, Any]) -> str:
    cart = context.get("cart_state", [])
    prefs = context.get("preferences", {})
    items = len(cart)
    size = prefs.get("size", "M")
    return f"Switching to {new_channel.title()} — your {items} cart item(s) and size {size} preference are preserved."

# ----------------------------
# Main handler: orchestrates flow per message
# ----------------------------
def handle_user_message(user_id: str, channel: str, catalog: List[Dict[str, Any]], message: str) -> Dict[str, Any]:
    # 1. Load context & continuity
    context = load_session(user_id, channel)
    prev_channel = context.get("channel", channel)
    context["user_id"] = user_id
    context["channel"] = channel

    continuity_note = None
    if prev_channel != channel:
        continuity_note = handoff_message(prev_channel, channel, context)

    # 2. Parse intent
    intent = parse_intent(message)
    context["intent"] = intent

    # Maintain chat history
    hist = context.setdefault("chat_history", [])
    hist.append({"t": time.time(), "user": message, "intent": intent})
    context["chat_history"] = hist[-10:]

    # persona
    persona = persona_selector(context)

    # 3. Route by intent
    rendered = ""
    reco = None
    nudge = ""
    tip = ""

    if intent["intent"] == "clear_cart":
        clear_cart(context)
        rendered = "🗑️ Cart cleared."

    elif intent["intent"] == "cart_review":
        summary = get_cart_summary(context)
        if not summary["items"]:
            rendered = "🛒 Your cart is empty."
        else:
            lines = []
            for i, it in enumerate(summary["items"], start=1):
                lines.append(f"{i}. {it['name']} — ₹{int(it['price'])} x {it['quantity']} (Size: {it.get('size')})")
            rendered = "🛒 Cart:\n" + "\n".join(lines) + f"\nSubtotal: ₹{int(summary['subtotal'])}"

    elif intent["intent"] == "checkout":
        summary = get_cart_summary(context)
        if not summary["items"]:
            rendered = "🛒 Your cart is empty — add some items first."
        else:
            earned = accrue_loyalty(summary["items"])
            pricing = calculate_pricing(summary["items"], loyalty_points_available=earned)
            rendered = (
                f"Subtotal: ₹{int(pricing['subtotal'])}\n"
                f"Discounts: ₹{int(abs(pricing['discounts']))}\n"
                f"Loyalty: ₹{int(abs(pricing['loyalty']))}\n"
                f"Shipping: ₹{int(pricing['shipping'])}\n"
                f"Payable: ₹{int(pricing['final'])}\n"
                f"✅ {tone_line(persona, 'checkout')}"
            )
            context.setdefault("promotions", {})["pricing_snapshot"] = pricing
            metrics = context.setdefault("metrics", {})
            metrics["checkouts_initiated"] = metrics.get("checkouts_initiated", 0) + 1
            metrics["aov_last"] = pricing["final"]
            nudge = "🚚 Add more items to unlock free shipping!"
            tip = "🧠 Stylist tip: Neutral colors pair well with festive accessories."

    elif intent["intent"] == "sizes_query":
        prev_reco = context.get("last_reco") or recommend(catalog, intent)
        context["last_reco"] = prev_reco
        rendered = handle_sizes_query(context, prev_reco)

    elif intent["intent"] == "cart_add":
        prev_reco = context.get("last_reco")
        if not prev_reco:
            prev_reco = recommend(catalog, intent)
            context["last_reco"] = prev_reco
        rendered = handle_cart_add(context, catalog, message, prev_reco)

    elif intent["intent"] == "handoff":
        rendered = continuity_note or f"Handoff prepared for {channel.title()}."

    else:
        # browse / default → recommendations with tips and nudges
        reco = recommend(catalog, intent)
        context["last_reco"] = reco
        cart_subtotal = get_cart_summary(context)["subtotal"]
        rendered = format_by_channel(reco, channel, persona, intent, cart_subtotal)
        tip = get_stylist_tip(intent)
        if cart_subtotal > 0 and cart_subtotal < 3000:
            nudge = f"🚚 You’re ₹{3000 - cart_subtotal} away from free shipping."

    # Continuity note prepend
    if continuity_note:
        rendered = continuity_note + "\n\n" + rendered

    # 4. Save context
    save_session(user_id, channel, context)

    # ✅ Return structured JSON
    return {
        "intent": intent,
        "recommendations": reco,
        "rendered": rendered,
        "cart": context.get("cart_state", []),
        "nudge": nudge,
        "tip": tip,
        "metrics": context.get("metrics", {})
    }

if __name__ == "__main__":
    catalog = load_catalog_from_csv("products.csv")
    user_id = "demo_user_pune"

    print("\n🛍️ REVA is ready. Type 'exit' to quit.\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("REVA: 👋 Bye! Happy shopping!")
            break
        out = handle_user_message(user_id, "whatsapp", catalog, user_text)
        print("REVA:", out["rendered"])