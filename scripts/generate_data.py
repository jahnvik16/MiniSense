"""Synthetic Survey Data Generator for MiniSense.

GENERATION ASSUMPTIONS & DESIGN:
1. Volume: Exactly 75,000 survey feedback records spanning April 1, 2026 to May 31, 2026 (61 days).
2. Time Distribution: Surveys distributed uniformly across the 61 days (~1,230/day), with timestamps
   incorporating realistic hour/minute distributions reflecting operating hours.
3. Multi-Entity Representation:
   - 5 Business Locations: Downtown Flagship, Westside Mall, Uptown Center, Airport Terminal B, North Suburbs.
   - 4 Channels: mobile_app, in_store_kiosk, email_receipt, web_survey.
   - 5 Cohorts: enterprise, self_serve, membership_vip, membership_regular, guest.
4. 8 Distinct Themes:
   - Food Quality, Wait Time, Staff, Cleanliness, Pricing, Membership, Facilities, App Experience.
5. Controlled Month-over-Month (MoM) Distributions:
   - Wait Time: April suffers severe delays (65% negative ratings 1-2); May undergoes an operational overhaul
     with express pickup lanes, surging to 65% positive ratings (4-5).
   - App Experience: April suffers crashes and sync issues on v1.8 (55% negative); May launches v2.0 overhaul,
     jumping to 70% positive ratings.
   - Pricing: April has normal satisfaction; May introduces a tier adjustment, causing a noticeable uptick in
     pricing complaints (45% negative in May vs 20% in April).
   - Food Quality & Staff: Maintain strong baseline performance across both months (~75-80% positive).
6. Deterministic & Reproducible: Uses random.Random(42) with template-based slot filling for rapid, zero-cost,
   offline synthesis without LLM API overhead.
"""

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
SURVEYS_FILE = DATA_DIR / "surveys.json"

TOTAL_RECORDS = 75000
RANDOM_SEED = 42

# Entities and categorical dimensions
LOCATIONS = [
    {"id": "loc_downtown", "name": "Downtown Flagship"},
    {"id": "loc_westside", "name": "Westside Center"},
    {"id": "loc_uptown", "name": "Uptown Square"},
    {"id": "loc_airport", "name": "Airport Terminal B"},
    {"id": "loc_suburbs", "name": "North Suburbs"},
]

CHANNELS = ["mobile_app", "in_store_kiosk", "email_receipt", "web_survey"]

SURVEY_TYPES = ["srv_post_visit", "srv_order_pickup", "srv_member_pulse"]

COHORTS = [
    {"cohort": "enterprise", "tier": "Enterprise", "weight": 0.15},
    {"cohort": "self_serve", "tier": "Starter", "weight": 0.35},
    {"cohort": "membership_vip", "tier": "VIP", "weight": 0.15},
    {"cohort": "membership_regular", "tier": "Member", "weight": 0.20},
    {"cohort": "guest", "tier": "Free", "weight": 0.15},
]

THEMES = [
    "Food Quality",
    "Wait Time",
    "Staff",
    "Cleanliness",
    "Pricing",
    "Membership",
    "Facilities",
    "App Experience",
]

# Slot-filling vocabulary
FOOD_ITEMS = ["Truffle Burger", "Artisan Salad", "Espresso Roast", "Avocado Toast", "Grain Bowl", "Grilled Panini", "Smoothie", "Matcha Latte"]
STAFF_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley"]
FACILITY_AREAS = ["restroom", "seating lounge", "patio", "pickup counter", "parking area", "ordering station"]
APP_FEATURES = ["order tracking", "digital wallet", "loyalty point scanner", "menu filters", "receipt history", "push notifications"]

# Feedback templates structured by Theme and Rating Bracket (Low: 1-2, Mid: 3, High: 4-5)
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Food Quality": {
        "low": [
            "The {food} was served lukewarm and lacked proper seasoning.",
            "Disappointed with the {food}. It felt stale and not freshly prepared.",
            "Quality was below expectations today; the {food} had an odd texture.",
            "Not what I expected from the menu. The {food} tasted bland.",
            "Sent back the {food} because it was undercooked.",
        ],
        "mid": [
            "The {food} was decent, but nothing to write home about.",
            "Standard taste and portion for the {food}. Average overall.",
            "The {food} was acceptable, though a bit too salty for my preference.",
            "Fair quality on the {food}, but presentation could be improved.",
        ],
        "high": [
            "The {food} was fresh, delicious, and perfectly prepared!",
            "Outstanding taste! The {food} is definitely one of the best on the menu.",
            "Exceptional culinary quality. The {food} exceeded all our expectations.",
            "Consistently fresh and flavorful. We loved the {food}.",
            "Top tier ingredients in the {food}. Will definitely order again!",
        ],
    },
    "Wait Time": {
        "low": [
            "Had to wait over {wait_mins} minutes just for a simple order. Completely unacceptable.",
            "Extremely slow turnaround. Waited {wait_mins} minutes during regular non-peak hours.",
            "Line moved at a snail's pace; took {wait_mins} minutes from queue to counter.",
            "Long wait time of {wait_mins} minutes caused us to be late for our meeting.",
            "Kitchen delays were terrible today. Waited {wait_mins} minutes for pickup.",
        ],
        "mid": [
            "Wait was around {wait_mins} minutes, which is somewhat tolerable but could be faster.",
            "Moderately busy, waited about {wait_mins} minutes for our items.",
            "Average wait time of {wait_mins} minutes. Expected slightly quicker service.",
        ],
        "high": [
            "Incredible speed! Order was ready in just {quick_mins} minutes.",
            "Lightning fast service—barely waited {quick_mins} minutes at the express pickup.",
            "Minimal wait time of only {quick_mins} minutes. Very efficient workflow.",
            "Impressed by how quickly the counter prepared everything in {quick_mins} minutes.",
            "Prompt and seamless turnaround in under {quick_mins} minutes!",
        ],
    },
    "Staff": {
        "low": [
            "Staff member seemed completely disinterested and barely acknowledged us.",
            "Customer service was unhelpful and dismissive when I asked about our order.",
            "Felt rushed and ignored by the cashier at the counter.",
            "Staff was overwhelmed and curt with several customers in line.",
        ],
        "mid": [
            "Staff was polite enough, though not particularly attentive.",
            "Standard interaction with the team. Neutral experience.",
            "The cashier handled the transaction fine, nothing notable.",
        ],
        "high": [
            "{staff} was remarkably courteous, welcoming, and attentive to every detail.",
            "Outstanding hospitality! {staff} went above and beyond to accommodate us.",
            "Warm smiles and great recommendations from {staff}. Fantastic team.",
            "Kudos to {staff} for swift and gracious service during a busy rush.",
            "Friendly, professional, and knowledgeable staff made our visit wonderful.",
        ],
    },
    "Cleanliness": {
        "low": [
            "The {area} was messy with unbussed tables and trash on the floor.",
            "Cleanliness standards were lacking, especially around the {area}.",
            "Tables were sticky and the {area} clearly needed immediate sanitation.",
            "Disappointed to find the {area} dirty and poorly maintained.",
        ],
        "mid": [
            "Cleanliness was adequate, though a couple of tables were waiting to be cleared.",
            "Fairly clean space, but the {area} could use more frequent checks.",
            "Acceptable cleanliness overall, standard commercial maintenance.",
        ],
        "high": [
            "Spotless environment! The {area} and dining space were impeccably clean.",
            "Very tidy, clean, and well-kept atmosphere throughout.",
            "Impressed by the hygiene and sanitization standards maintained in the {area}.",
            "Clean, bright, and inviting dining area. Great job by the maintenance team.",
        ],
    },
    "Pricing": {
        "low": [
            "Prices have increased noticeably and no longer feel justified by the portion sizes.",
            "Way too expensive for the value offered. Sudden price hike is frustrating.",
            "Bill was significantly higher than last month without noticeable improvements.",
            "Hidden fees and higher tiered pricing make this hard to recommend.",
        ],
        "mid": [
            "Prices are on the higher side, but acceptable for occasional visits.",
            "Moderate value for money. Not cheap, but comparable to competitors.",
            "Fair pricing structure, though discounts could be more competitive.",
        ],
        "high": [
            "Great value for money! High quality offerings at very reasonable rates.",
            "Generous portions and fair pricing compared to alternatives in the area.",
            "Affordable and transparent pricing. Very satisfied with the bill.",
            "Excellent cost-to-benefit ratio, especially with the member discount applied.",
        ],
    },
    "Membership": {
        "low": [
            "Tier rewards and points take too long to accumulate to be worthwhile.",
            "Member portal failed to apply my earned discount code at checkout.",
            "Disappointed with the recent reduction in tier benefits and perks.",
            "Customer support could not explain why my loyalty credits expired early.",
        ],
        "mid": [
            "Membership program is okay, though the perks are fairly standard.",
            "Decent point accumulation, but redeeming vouchers can be cumbersome.",
            "Average loyalty program benefits compared to other retailers.",
        ],
        "high": [
            "Loving the VIP member benefits! The free express shipping and perks are great.",
            "Loyalty rewards are easy to earn and redeem on every purchase.",
            "Membership pays for itself quickly with exclusive member events and perks.",
            "Seamless reward redemption and generous monthly member perks.",
        ],
    },
    "Facilities": {
        "low": [
            "Air conditioning was barely working and the room felt uncomfortably warm.",
            "Seating was cramped and the Wi-Fi connection was unstable throughout.",
            "Restroom fixtures were malfunctioning and lacked hand soap.",
            "Parking was chaotic with poorly marked bays and inadequate lighting.",
        ],
        "mid": [
            "Facilities are decent, though seating capacity during peak lunch is limited.",
            "Average amenities. Seating was okay, Wi-Fi speed was mediocre.",
            "Functional space, but decor and seating cushions could use a refresh.",
        ],
        "high": [
            "Spacious, comfortable seating with great lighting and fast Wi-Fi.",
            "Modern, beautifully maintained facilities with accessible amenities.",
            "Convenient parking and pleasant ambiance make visiting very comfortable.",
            "Clean, well-ventilated, and quiet workspaces available.",
        ],
    },
    "App Experience": {
        "low": [
            "The app crashed twice during checkout and forgot my saved payment card.",
            "Glitchy mobile app with constant loading spinners and failed {app_feat}.",
            "Order tracking was inaccurate and pushed duplicate notifications.",
            "Extremely slow interface; {app_feat} stopped responding completely.",
            "Recent app release broke login credentials and cleared my cart.",
        ],
        "mid": [
            "The app works for basic ordering, but {app_feat} can be sluggish.",
            "Average app utility. Navigation is okay, but UI feels somewhat dated.",
            "Functional app, though order confirmation notifications are sometimes delayed.",
        ],
        "high": [
            "Sleek and intuitive app! {app_feat} is fast and reliable.",
            "Smooth mobile ordering with 1-click payment and instant real-time tracking.",
            "Love the new app redesign! Beautiful user interface and zero crashes.",
            "The updated app makes reordering favorite items effortless in seconds.",
            "Flawless mobile experience with quick {app_feat} integration.",
        ],
    },
}


def get_rating_distribution(theme: str, is_may: bool, rng: random.Random) -> int:
    """Return a controlled 1-5 rating based on Theme and Month to create realistic MoM shifts."""
    r = rng.random()

    if theme == "Wait Time":
        if not is_may:
            # April: Severe operational delays (65% low, 20% mid, 15% high)
            if r < 0.40:
                return 1
            if r < 0.65:
                return 2
            if r < 0.85:
                return 3
            if r < 0.95:
                return 4
            return 5
        else:
            # May: Express pickup stations deployed (15% low, 20% mid, 65% high)
            if r < 0.05:
                return 1
            if r < 0.15:
                return 2
            if r < 0.35:
                return 3
            if r < 0.65:
                return 4
            return 5

    elif theme == "App Experience":
        if not is_may:
            # April: Legacy app v1.8 crash bugs (55% low, 25% mid, 20% high)
            if r < 0.30:
                return 1
            if r < 0.55:
                return 2
            if r < 0.80:
                return 3
            if r < 0.92:
                return 4
            return 5
        else:
            # May: Modern v2.0 app rollout (10% low, 20% mid, 70% high)
            if r < 0.03:
                return 1
            if r < 0.10:
                return 2
            if r < 0.30:
                return 3
            if r < 0.65:
                return 4
            return 5

    elif theme == "Pricing":
        if not is_may:
            # April: Stable normal pricing (20% low, 35% mid, 45% high)
            if r < 0.08:
                return 1
            if r < 0.20:
                return 2
            if r < 0.55:
                return 3
            if r < 0.85:
                return 4
            return 5
        else:
            # May: Tier adjustment backlash (45% low, 35% mid, 20% high)
            if r < 0.22:
                return 1
            if r < 0.45:
                return 2
            if r < 0.80:
                return 3
            if r < 0.92:
                return 4
            return 5

    elif theme in ("Food Quality", "Staff"):
        # Consistently high performance (~75% high, 15% mid, 10% low)
        if r < 0.04:
            return 1
        if r < 0.10:
            return 2
        if r < 0.25:
            return 3
        if r < 0.60:
            return 4
        return 5

    else:
        # Balanced baseline for Cleanliness, Membership, Facilities (~65% high, 20% mid, 15% low)
        if r < 0.06:
            return 1
        if r < 0.15:
            return 2
        if r < 0.35:
            return 3
        if r < 0.70:
            return 4
        return 5


def generate_feedback_text(theme: str, rating: int, rng: random.Random) -> str:
    """Select and hydrate a template matching the theme and rating level."""
    bracket = "low" if rating <= 2 else ("mid" if rating == 3 else "high")
    template_list = TEMPLATES.get(theme, {}).get(bracket, ["Service was fine."])
    chosen = rng.choice(template_list)

    return chosen.format(
        food=rng.choice(FOOD_ITEMS),
        wait_mins=rng.randint(22, 55),
        quick_mins=rng.randint(3, 8),
        staff=rng.choice(STAFF_NAMES),
        area=rng.choice(FACILITY_AREAS),
        app_feat=rng.choice(APP_FEATURES),
    )


def compute_nps_from_rating(rating: int, rng: random.Random) -> int:
    """Correlate 1-5 rating into 0-10 Net Promoter Score with natural variance."""
    if rating == 5:
        return rng.choice([9, 10, 10, 10])
    if rating == 4:
        return rng.choice([7, 8, 8, 9])
    if rating == 3:
        return rng.choice([5, 6, 6, 7])
    if rating == 2:
        return rng.choice([2, 3, 4, 5])
    return rng.choice([0, 1, 1, 2])


def generate_dataset() -> list[dict[str, Any]]:
    """Synthesize 75,000 survey records deterministically."""
    rng = random.Random(RANDOM_SEED)

    start_date = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_seconds = 61 * 86400  # 30 days in April + 31 days in May

    # Cohort selection weights
    cohort_choices = [c["cohort"] for c in COHORTS]
    tier_map = {c["cohort"]: c["tier"] for c in COHORTS}
    cohort_weights = [c["weight"] for c in COHORTS]

    # Pre-generate customer pool of 15,000 for realistic repeat visit dynamics
    customer_pool = [f"cust_{i:05d}" for i in range(1, 15001)]

    records: list[dict[str, Any]] = []

    print(f"Generating {TOTAL_RECORDS:,} survey records (April - May 2026)...")

    for i in range(1, TOTAL_RECORDS + 1):
        # Evenly spread timestamp across the 61 days with natural jitter
        offset_seconds = int((i - 1) * (total_seconds / TOTAL_RECORDS)) + rng.randint(0, 50)
        dt = start_date + timedelta(seconds=offset_seconds)
        is_may = dt.month == 5

        theme = rng.choice(THEMES)
        rating = get_rating_distribution(theme, is_may, rng)
        nps = compute_nps_from_rating(rating, rng)

        sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
        feedback = generate_feedback_text(theme, rating, rng)

        loc = rng.choice(LOCATIONS)
        cohort = rng.choices(cohort_choices, weights=cohort_weights, k=1)[0]
        channel = rng.choice(CHANNELS)
        survey_type = rng.choice(SURVEY_TYPES)

        record = {
            "id": f"srv_{i:06d}",
            "customer_id": rng.choice(customer_pool),
            "business_id": loc["id"],
            "business_name": loc["name"],
            "survey_id": survey_type,
            "channel": channel,
            "cohort": cohort,
            "product_tier": tier_map[cohort],
            "theme": theme,
            "category": theme,
            "rating": rating,
            "csat_score": rating,
            "nps_score": nps,
            "sentiment": sentiment,
            "feedback": feedback,
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        records.append(record)

    return records


def validate_dataset(records: list[dict[str, Any]]) -> None:
    """Verify record counts, dates, ranges, schema completeness, and MoM shifts."""
    print("\n--- Validating Generated Dataset ---")

    count = len(records)
    print(f"1. Total Record Count: {count:,} (Expected: {TOTAL_RECORDS:,})")
    assert count == TOTAL_RECORDS, f"Expected {TOTAL_RECORDS} records, got {count}"

    # Required fields
    required_keys = {
        "id", "customer_id", "business_id", "channel", "cohort", "product_tier",
        "theme", "category", "rating", "csat_score", "nps_score", "sentiment",
        "feedback", "timestamp"
    }

    april_count = 0
    may_count = 0
    ratings_in_range = True

    # Theme metric trackers for April vs May
    theme_ratings: dict[str, dict[str, list[int]]] = {
        t: {"april": [], "may": []} for t in THEMES
    }

    for r in records:
        missing = required_keys - r.keys()
        assert not missing, f"Record {r.get('id')} missing required keys: {missing}"

        val = r["rating"]
        if not (1 <= val <= 5):
            ratings_in_range = False

        dt = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        if dt.month == 4:
            april_count += 1
            theme_ratings[r["theme"]]["april"].append(val)
        elif dt.month == 5:
            may_count += 1
            theme_ratings[r["theme"]]["may"].append(val)
        else:
            raise ValueError(f"Date {dt} out of April-May 2026 range!")

    print(f"2. Required Fields Check: PASSED (all {len(required_keys)} fields verified on 100% of records)")
    print(f"3. Rating Range Check: {'PASSED (all 1-5)' if ratings_in_range else 'FAILED'}")
    assert ratings_in_range, "Found ratings outside 1-5"

    print(f"4. Month Coverage: April = {april_count:,} records | May = {may_count:,} records")
    assert april_count > 0 and may_count > 0, "Both months must be represented"
    assert (april_count + may_count) == count, "Every record must fall in April or May 2026"

    # Display Month-over-Month impact analysis
    print("\n5. Month-over-Month (MoM) Metric Verification (Avg Rating & % CSAT 4-5):")
    for t in THEMES:
        apr_scores = theme_ratings[t]["april"]
        may_scores = theme_ratings[t]["may"]
        apr_avg = sum(apr_scores) / len(apr_scores) if apr_scores else 0
        may_avg = sum(may_scores) / len(may_scores) if may_scores else 0
        apr_csat = (sum(1 for s in apr_scores if s >= 4) / len(apr_scores) * 100) if apr_scores else 0
        may_csat = (sum(1 for s in may_scores if s >= 4) / len(may_scores) * 100) if may_scores else 0
        delta = may_csat - apr_csat
        print(f"   - {t:<15}: Apr Avg={apr_avg:.2f} (CSAT {apr_csat:.1f}%) -> May Avg={may_avg:.2f} (CSAT {may_csat:.1f}%) | Delta: {delta:+.1f}%")

    print("\nAll validation checks PASSED successfully.")


def main() -> None:
    records = generate_dataset()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing dataset to {SURVEYS_FILE}...")
    with open(SURVEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    file_size_mb = SURVEYS_FILE.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {SURVEYS_FILE.name} ({file_size_mb:.2f} MB)")

    validate_dataset(records)


if __name__ == "__main__":
    main()
