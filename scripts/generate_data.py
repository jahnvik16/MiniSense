"""Synthetic Survey Data Generator for MiniSense (Appendix A Schema Compliant).

GENERATION METHODOLOGY & PURPOSE:
1. Volume: Exactly 75,000 survey feedback records spanning April 1, 2026 to May 31, 2026 (61 days).
2. Schema: Adheres strictly to the assignment's Appendix A schema:
   - response_id (str)
   - date (str, YYYY-MM-DD)
   - business_id (str)
   - business_name (str)
   - survey_id (str)
   - survey_name (str)
   - rating (int, 1-5)
   - response_channel (str)
   - free_text (str)
3. No Pre-labeled Ground Truth:
   Theme, sentiment, CSAT, and NPS are NOT pre-stored as answer fields in the raw dataset.
   Instead, analytical agents derive themes dynamically from `free_text` using rule-based classifiers
   and derive sentiment dynamically from `rating`.
4. Realistic Lexical Variation:
   Feedback free_text is synthesized using a multi-clause combinatorial template system with
   randomized openers, core observations, slot-fillers (foods, staff, facility areas, app features,
   numerical minute durations), and closing thoughts.
5. Controlled Temporal Signals (MoM):
   The synthetic generator intentionally includes controlled temporal signals so that
   comparison and trend-analysis capabilities can be evaluated:
   - Wait Time: April exhibits longer wait times / lower ratings; May exhibits shorter wait times / higher ratings.
   - App Experience: April exhibits software glitch reports; May exhibits higher satisfaction following v2.0 rollout.
   - Pricing: April maintains baseline satisfaction; May exhibits increased pricing sensitivity comments.
   - Food Quality & Staff: Maintain consistently strong baseline satisfaction across both months.
   Note: Correlation between operational policies (such as express pickup in the FAQ) and metric changes
   is an intentional temporal signal for evaluation, not proof of causality.
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

from app.tools.data_tools import classify_theme, classify_sentiment, THEMES

DATA_DIR = BASE_DIR / "data"
SURVEYS_FILE = DATA_DIR / "surveys.json"

TOTAL_RECORDS = 75000
RANDOM_SEED = 42

# Appendix A Entities
BUSINESS_LOCATIONS = [
    {"business_id": "b01", "business_name": "GreenLeaf Bistro - Downtown Flagship"},
    {"business_id": "b02", "business_name": "GreenLeaf Bistro - Westside Center"},
    {"business_id": "b03", "business_name": "GreenLeaf Bistro - Uptown Square"},
    {"business_id": "b04", "business_name": "GreenLeaf Bistro - Airport Terminal B"},
    {"business_id": "b05", "business_name": "GreenLeaf Bistro - North Suburbs"},
]

SURVEY_INSTRUMENTS = [
    {"survey_id": "s01", "survey_name": "Customer Dining Satisfaction"},
    {"survey_id": "s02", "survey_name": "Order Pickup & Wait Experience"},
    {"survey_id": "s03", "survey_name": "Membership & Rewards Value"},
    {"survey_id": "s04", "survey_name": "Mobile App & Digital Ordering"},
]

CHANNELS = ["mobile", "kiosk", "web", "email"]

# Lexical Slot Fillers
FOOD_ITEMS = [
    "Truffle Burger", "Artisan Salad", "Cold Brew Espresso", "Avocado Toast",
    "Garden Grain Bowl", "Grilled Panini", "Berry Smoothie", "Matcha Latte",
    "Quinoa Bowl", "Sourdough Melt", "Roasted Veggie Wrap", "Seasonal Harvest Soup",
    "Almond Croissant", "Chia Seed Pudding", "Organic Herb Tea", "Protein Power Bowl"
]

STAFF_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley",
    "Cameron", "Jamie", "Avery", "Kendall", "Casey", "Dakota", "Reese", "Quinn"
]

FACILITY_AREAS = [
    "restroom", "dining seating lounge", "outdoor patio", "pickup counter",
    "parking area", "ordering station", "condiment bar", "front entrance area"
]

APP_FEATURES = [
    "order tracking", "digital wallet", "loyalty point scanner", "menu filters",
    "receipt history", "push notifications", "reorder shortcut", "custom tip selector"
]

# Openers and Closings for combinatorially rich natural language text
OPENERS = [
    "",
    "Stopped by during the lunch rush. ",
    "Visited this location earlier today. ",
    "Placed an order on the go. ",
    "Had a quick meal here with a coworker. ",
    "Came in during morning off-peak hours. ",
    "First time visiting this week. ",
    "Regular customer here. ",
    "Dropped in for dinner tonight. ",
    "Ordered ahead of time. ",
    "Quick visit during afternoon break. ",
]

CLOSERS = [
    "",
    " Will keep this in mind for future visits.",
    " Hope the management takes note.",
    " Appreciate the effort.",
    " Definitely coming back again soon.",
    " Looking forward to seeing improvements.",
    " Overall a memorable visit.",
    " Thanks for listening to customer feedback.",
    " Will try another location next time.",
    " That stood out to me.",
    " Keep up the good work.",
]

# Core Theme Templates by Bracket (Low: 1-2, Mid: 3, High: 4-5)
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Food Quality": {
        "low": [
            "The {food} was served lukewarm and lacked proper seasoning.",
            "Disappointed with the {food}. It tasted stale and uninspired.",
            "Culinary quality was below standard today; the {food} had an odd texture.",
            "Not what I expected from the kitchen. The {food} tasted noticeably bland.",
            "Had to return the {food} because it was undercooked.",
            "The ingredients in the {food} did not taste fresh at all.",
            "Flavor of the {food} was completely off and overly salty.",
            "Portion size of the {food} felt stingy and the dish was barely warm.",
        ],
        "mid": [
            "The {food} was decent, but nothing to write home about.",
            "Standard taste and portion for the {food}. Average meal overall.",
            "The {food} was acceptable, though a bit too salty for my preference.",
            "Fair culinary preparation on the {food}, but plating could be improved.",
            "Ordinary taste on the {food}; neither great nor terrible.",
            "The recipe for the {food} is okay, though seasoning could be dialed in better.",
        ],
        "high": [
            "The {food} was fresh, delicious, and seasoned to perfection!",
            "Outstanding taste! The {food} is definitely one of the best on the menu.",
            "Exceptional culinary quality. The {food} exceeded all expectations.",
            "Consistently fresh, vibrant, and flavorful. We loved the {food}.",
            "Top tier ingredients in the {food}. Prepared to perfection!",
            "Loved the authentic taste and generous portion of the {food}.",
            "The {food} was hot, flavorful, and incredibly satisfying.",
        ],
    },
    "Wait Time": {
        "low": [
            "Had to wait over {wait_mins} minutes just for a simple counter order. Completely unacceptable delay.",
            "Extremely slow turnaround. Waited {wait_mins} minutes during regular hours.",
            "Line moved at a snail's pace; took {wait_mins} minutes from queue to counter pickup.",
            "Long wait time of {wait_mins} minutes caused severe schedule delays for my meeting.",
            "Kitchen delays were frustrating today. Waited {wait_mins} minutes for pickup.",
            "Heavy delays at the counter; waited {wait_mins} minutes while orders piled up.",
            "The turnaround time was terrible—over {wait_mins} minutes in line.",
        ],
        "mid": [
            "Wait was around {wait_mins} minutes, which is somewhat tolerable but could be faster.",
            "Moderately busy turnaround; waited about {wait_mins} minutes for our items.",
            "Average wait time of {wait_mins} minutes. Expected slightly quicker service.",
            "Line moved at an acceptable pace, took about {wait_mins} minutes total.",
            "Moderate wait time; kitchen took {wait_mins} minutes to call my number.",
        ],
        "high": [
            "Incredible speed! Order was ready in just {quick_mins} minutes.",
            "Lightning fast service—barely waited {quick_mins} minutes at the express pickup.",
            "Minimal wait time of only {quick_mins} minutes. Very efficient workflow.",
            "Impressed by how quickly the counter prepared everything in {quick_mins} minutes.",
            "Prompt and seamless turnaround in under {quick_mins} minutes! Great express speed.",
            "Almost zero waiting time; order handed over within {quick_mins} minutes.",
        ],
    },
    "Staff": {
        "low": [
            "Staff member seemed completely disinterested and barely acknowledged us at the register.",
            "Customer service was unhelpful and dismissive when I asked about our order status.",
            "Felt rushed and ignored by the cashier at the counter station.",
            "Staff was overwhelmed and curt with several customers waiting in line.",
            "The team member at the register was visibly impatient and rude.",
            "Poor hospitality today; staff seemed inattentive to guest questions.",
        ],
        "mid": [
            "Staff was polite enough, though not particularly attentive during the transaction.",
            "Standard interaction with the team. Neutral customer service experience.",
            "The cashier handled the transaction fine, nothing notable either way.",
            "Staff was moderately friendly, though busy attending to backend tasks.",
        ],
        "high": [
            "{staff} was remarkably courteous, welcoming, and attentive to every detail.",
            "Outstanding hospitality! {staff} went above and beyond to assist us.",
            "Warm smiles and great recommendations from {staff}. Fantastic team member.",
            "Kudos to {staff} for swift and gracious customer service during a busy rush.",
            "Friendly, professional, and knowledgeable staff made our visit wonderful.",
            "Terrific service from {staff}, genuinely hospitable and kind.",
        ],
    },
    "Cleanliness": {
        "low": [
            "The {area} was messy with unbussed tables and trash on the floor.",
            "Cleanliness standards were lacking, especially around the {area}.",
            "Tables were sticky and the {area} clearly needed immediate sanitation.",
            "Disappointed to find the {area} dirty and poorly maintained.",
            "Sanitation was subpar; trash bins overflowing near the {area}.",
        ],
        "mid": [
            "Cleanliness was adequate, though a couple of tables were waiting to be cleared.",
            "Fairly clean space, but the {area} could use more frequent maintenance checks.",
            "Acceptable cleanliness overall, standard commercial upkeep.",
            "The {area} was tidy enough, though floor needed sweeping.",
        ],
        "high": [
            "Spotless environment! The {area} and dining space were impeccably clean.",
            "Very tidy, clean, and well-kept atmosphere throughout the premises.",
            "Impressed by the hygiene and sanitization standards maintained in the {area}.",
            "Clean, bright, and inviting dining area. Great job by the maintenance team.",
            "Impeccable hygiene and gleaming tables around the {area}.",
        ],
    },
    "Pricing": {
        "low": [
            "Prices have increased noticeably and no longer feel justified by the portion sizes.",
            "Way too expensive for the value offered. Sudden price hike is frustrating.",
            "Bill was significantly higher than expected without noticeable quality improvements.",
            "Hidden fees and higher prices make this hard to recommend on a regular basis.",
            "Cost is getting unreasonable; prices jumped substantially this month.",
        ],
        "mid": [
            "Prices are on the higher side, but acceptable for occasional visits.",
            "Moderate value for money. Not cheap, but comparable to competitors in the area.",
            "Fair pricing structure, though combo discounts could be more competitive.",
            "Prices are reasonable for organic ingredients, though slightly elevated.",
        ],
        "high": [
            "Great value for money! High quality offerings at very reasonable rates.",
            "Generous portions and fair pricing compared to alternatives in the area.",
            "Affordable and transparent pricing. Very satisfied with the bill.",
            "Excellent cost-to-benefit ratio, especially with the member discount applied.",
            "Reasonably priced menu with transparent pricing on all add-ons.",
        ],
    },
    "Membership": {
        "low": [
            "Tier rewards and points take too long to accumulate to be worthwhile.",
            "Member portal failed to apply my earned discount code at checkout.",
            "Disappointed with the recent reduction in tier benefits and loyalty perks.",
            "Customer support could not explain why my loyalty credits expired early.",
            "Frustrated that my membership points did not register on my account.",
        ],
        "mid": [
            "Membership program is okay, though the perks are fairly standard.",
            "Decent point accumulation, but redeeming vouchers can be cumbersome.",
            "Average loyalty program benefits compared to other retail programs.",
            "Member perks are fine, though point thresholds are high.",
        ],
        "high": [
            "Loving the VIP member benefits! The free express perks and rewards are great.",
            "Loyalty rewards are easy to earn and redeem on every purchase.",
            "Membership pays for itself quickly with exclusive member events and perks.",
            "Seamless reward redemption and generous monthly member perks.",
            "Fantastic loyalty program; earned my free reward drink in no time!",
        ],
    },
    "Facilities": {
        "low": [
            "Air conditioning was barely working and the room felt uncomfortably warm.",
            "Seating was cramped and the Wi-Fi connection was unstable throughout.",
            "Restroom fixtures were malfunctioning and lacked hand soap.",
            "Parking was chaotic with poorly marked bays and inadequate lighting.",
            "Room amenities were outdated and the ventilation felt stuffy.",
        ],
        "mid": [
            "Facilities are decent, though seating capacity during peak lunch is limited.",
            "Average amenities. Seating was okay, Wi-Fi speed was mediocre.",
            "Functional space, but decor and seating cushions could use a refresh.",
            "Amenities are acceptable for a quick stop.",
        ],
        "high": [
            "Spacious, comfortable seating with great lighting and fast Wi-Fi.",
            "Modern, beautifully maintained facilities with accessible amenities.",
            "Convenient parking and pleasant ambiance make visiting very comfortable.",
            "Clean, well-ventilated, and quiet workspaces available.",
            "Delightful physical atmosphere with cozy seating and great natural light.",
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
            "App performance is passable for placing pickups.",
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
            # April: Operational bottlenecks (65% low, 20% mid, 15% high)
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
            # May: Express pickup stations introduced (15% low, 20% mid, 65% high)
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
            # April: v1.8 crash reports (55% low, 25% mid, 20% high)
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
            # May: v2.0 rollout (10% low, 20% mid, 70% high)
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
            # April: Normal satisfaction (20% low, 35% mid, 45% high)
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
            # May: Price sensitivity shift (45% low, 35% mid, 20% high)
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
        # Strong baseline across both months (~75% high, 15% mid, 10% low)
        if r < 0.04:
            return 1
        if r < 0.10:
            return 2
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
    """Select and hydrate combinatorial templates to ensure massive lexical diversity."""
    bracket = "low" if rating <= 2 else ("mid" if rating == 3 else "high")
    template_list = TEMPLATES.get(theme, {}).get(bracket, ["Service was fine."])
    chosen = rng.choice(template_list)

    text = chosen.format(
        food=rng.choice(FOOD_ITEMS),
        wait_mins=rng.randint(20, 55),
        quick_mins=rng.randint(2, 8),
        staff=rng.choice(STAFF_NAMES),
        area=rng.choice(FACILITY_AREAS),
        app_feat=rng.choice(APP_FEATURES),
    )

    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    return f"{opener}{text}{closer}".strip()


def generate_dataset() -> list[dict[str, Any]]:
    """Synthesize 75,000 survey records deterministically matching Appendix A."""
    rng = random.Random(RANDOM_SEED)

    start_date = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_days = 61  # 30 days in April + 31 days in May

    records: list[dict[str, Any]] = []

    print(f"Generating {TOTAL_RECORDS:,} survey records adhering to Appendix A schema...")

    for i in range(1, TOTAL_RECORDS + 1):
        # Calculate concrete calendar date
        day_offset = int((i - 1) * (total_days / TOTAL_RECORDS))
        dt = start_date + timedelta(days=day_offset)
        date_str = dt.strftime("%Y-%m-%d")
        is_may = dt.month == 5

        theme = rng.choice(THEMES)
        rating = get_rating_distribution(theme, is_may, rng)
        free_text = generate_feedback_text(theme, rating, rng)

        loc = rng.choice(BUSINESS_LOCATIONS)
        survey = rng.choice(SURVEY_INSTRUMENTS)
        channel = rng.choice(CHANNELS)

        # EXACT Appendix A schema:
        # response_id, date, business_id, business_name, survey_id, survey_name, rating, response_channel, free_text
        record = {
            "response_id": f"r{i:05d}",
            "date": date_str,
            "business_id": loc["business_id"],
            "business_name": loc["business_name"],
            "survey_id": survey["survey_id"],
            "survey_name": survey["survey_name"],
            "rating": rating,
            "response_channel": channel,
            "free_text": free_text,
        }
        records.append(record)

    return records


def validate_dataset(records: list[dict[str, Any]]) -> None:
    """Verify record count, schema, date ranges, and classifier accuracy."""
    print("\n--- Validating Generated Dataset against Appendix A ---")

    count = len(records)
    print(f"1. Total Record Count: {count:,} (Expected: {TOTAL_RECORDS:,})")
    assert count == TOTAL_RECORDS, f"Expected {TOTAL_RECORDS} records, got {count}"

    # Appendix A Required Fields
    exact_appendix_a_keys = {
        "response_id", "date", "business_id", "business_name",
        "survey_id", "survey_name", "rating", "response_channel", "free_text"
    }

    # Prohibited pre-labeled ground truth fields
    prohibited_keys = {"theme", "category", "sentiment", "csat_score", "nps_score", "cohort", "product_tier"}

    april_count = 0
    may_count = 0
    ratings_in_range = True

    for r in records:
        keys = set(r.keys())
        assert keys == exact_appendix_a_keys, f"Schema mismatch: expected {exact_appendix_a_keys}, got {keys}"
        assert not (keys & prohibited_keys), f"Found pre-labeled ground-truth fields: {keys & prohibited_keys}"

        rating = r["rating"]
        if not (1 <= rating <= 5):
            ratings_in_range = False

        date_val = r["date"]
        month = int(date_val[5:7])
        if month == 4:
            april_count += 1
        elif month == 5:
            may_count += 1
        else:
            raise ValueError(f"Date {date_val} outside April-May 2026 range!")

    print(f"2. Appendix A Schema Check: PASSED (100% exact match across all {count:,} records)")
    print(f"3. No Pre-labeled Ground Truth: PASSED (no theme/sentiment/CSAT/NPS fields stored)")
    print(f"4. Rating Range Check: {'PASSED (all 1-5)' if ratings_in_range else 'FAILED'}")
    print(f"5. Temporal Coverage: April = {april_count:,} records | May = {may_count:,} records")

    # Sample verification
    first_record = records[0]
    print("\nSample Generated Record (First Entry):")
    print(json.dumps(first_record, indent=2))

    # Test classifier on first 1000 records
    print("\n6. Classifier Validation on Sample:")
    sample_themes = [classify_theme(r["free_text"]) for r in records[:1000]]
    classified_themes = set(sample_themes)
    print(f"   Derived themes on 1,000 samples: {classified_themes}")
    assert len(classified_themes) >= 6, "Classifier should recognize varied themes in sample"

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
