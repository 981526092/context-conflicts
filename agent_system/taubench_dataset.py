"""Handcrafted retail user personas and dialogues for τ-bench pilot experiments.

Five users × 15 steps each = 75 turns.  Ground truth rankings are generated
deterministically by ``_compute_gt_ranking()`` based on the user's budget
sensitivity and the current shopping context.

Dialogue design principles:
- Step 1: onboarding prompt expressing budget and needs
- Step 2+: references the previous user_choice ("I went with the X...")
- Each step has a different shopping need, covering all 20 product categories
- user_choice consistency: tight→tier 1-2, moderate→tier 2-3, flexible→tier 3-5

Mirrors ConvFinReStep/ConvFinReUser API so the experiment runner can use
the same patterns.
"""

from dataclasses import dataclass
from typing import List, Optional


# Local import to avoid circular dependency at module level
def _get_product_dbs():
    from .taubench_tools import PRODUCT_RISK_DB, PRODUCT_QUALITY_DB
    return PRODUCT_RISK_DB, PRODUCT_QUALITY_DB


@dataclass
class RetailStep:
    """A single decision step for a retail user."""
    user_id: str
    step: int
    user_message: str
    ground_truth_ranking: List[str]
    user_choice: str


@dataclass
class RetailUser:
    """All data for a single retail user across their session."""
    user_id: str
    name: str
    budget_sensitivity: str
    initial_goals: List[str]
    initial_constraints: List[str]
    steps: List[RetailStep]


# ---------------------------------------------------------------------------
# Ground truth ranking function
# ---------------------------------------------------------------------------

def _compute_gt_ranking(
    budget: str,
    context_products: Optional[List[str]] = None,
) -> List[str]:
    """Compute ground truth product ranking based on budget sensitivity.

    - tight: sort by (price_tier ASC, quality DESC) — cheapest first, price
      dominates; a tight-budget user should never be recommended tier 4-5.
    - moderate: sort by composite score = quality * 2 - abs(price_tier - 3)
      — prefers high-quality items near the mid-tier price point (tier 3).
      This produces different rankings from tight (e.g. Headphones at tier 3
      with quality 4 ranks first, not T-Shirt at tier 1).
    - flexible: sort by (quality DESC, price_tier DESC) — best quality first,
      premium tiebreak; budget is not a constraint.

    If context_products is provided, those products are boosted to the front
    (in their original budget-appropriate order) to reflect contextual relevance.
    """
    RISK_DB, QUALITY_DB = _get_product_dbs()
    products = list(RISK_DB.keys())

    if budget == "tight":
        products.sort(key=lambda p: (RISK_DB[p], -QUALITY_DB[p]))
    elif budget == "flexible":
        products.sort(key=lambda p: (-QUALITY_DB[p], -RISK_DB[p]))
    else:  # moderate — quality-weighted mid-tier preference
        products.sort(key=lambda p: -(QUALITY_DB[p] * 2 - abs(RISK_DB[p] - 3)))

    if context_products:
        # Boost context-relevant products to front, preserving budget order
        boosted = [p for p in products if p in context_products]
        rest = [p for p in products if p not in context_products]
        products = boosted + rest

    return products


# ============================================================================
# User 0: Alex Chen — budget_sensitivity: tight
# College student, monthly budget ~$100, daily essentials and fitness.
# Choices: always tier 1-2 (T-Shirt, Water Bottle, Running Shoes, Backpack)
# ============================================================================

_USER_0_STEPS = [
    RetailStep(
        user_id="User_0", step=1,
        user_message=(
            "Hi, I'm a college student on a really tight budget — maybe $50-100 per item max. I mainly need everyday essentials. What do you recommend?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_0", step=2,
        user_message=(
            "I went with the Water Bottle — great pick, thanks! Now I need a basic t-shirt for everyday wear. Still keeping it under budget."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_0", step=3,
        user_message=(
            "The T-Shirt fits perfectly. I've been going to the campus gym more — do you have affordable workout shoes? I really can't spend more than $100."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_0", step=4,
        user_message=(
            "Love the Running Shoes, they're great for the gym. Now I need a backpack for carrying textbooks — something sturdy but cheap."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_0", step=5,
        user_message=(
            "That Backpack is perfect for classes. My roommate has a smartwatch and says I should get one too — but honestly, that feels like a splurge I can't afford. What do you think?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Phone Case",
    ),
    RetailStep(
        user_id="User_0", step=6,
        user_message=(
            "Yeah, I'll stick to my budget. I got a Phone Case instead — way more practical. I need something for my dorm room now, maybe good socks for campus walking?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Socks"]),
        user_choice="Socks",
    ),
    RetailStep(
        user_id="User_0", step=7,
        user_message=(
            "Got the Socks, really comfortable for all-day wear. My sister's birthday is coming up — I want to get her something nice but I only have about $60 to spend. Any ideas?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_0", step=8,
        user_message=(
            "The Water Bottle was a hit as a gift! Now I'm looking for something to help with my daily commute to campus. Still on that student budget though."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Backpack", "Running Shoes"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_0", step=9,
        user_message=(
            "Already have that Backpack actually — it's been great. Let me think... I've been wanting better headphones for studying in the library, but those seem expensive. Anything budget-friendly?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Phone Case",
    ),
    RetailStep(
        user_id="User_0", step=10,
        user_message=(
            "I passed on the headphones — too pricey for me right now. Got a Phone Case for my friend instead. Finals are coming up and I need a yoga mat for stress relief. Something cheap?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Yoga Mat"]),
        user_choice="Yoga Mat",
    ),
    RetailStep(
        user_id="User_0", step=11,
        user_message=(
            "The Yoga Mat has been great for de-stressing during finals. I'm starting to run outdoors now too — do I need different shoes or will my current ones work?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_0", step=12,
        user_message=(
            "Getting a second pair of Running Shoes was smart — I rotate them now. My backpack is showing wear after a semester. Time for a replacement?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_0", step=13,
        user_message=(
            "New Backpack is solid. Summer break is coming and I want to refresh my wardrobe a bit. Nothing fancy — just basics."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_0", step=14,
        user_message=(
            "T-Shirt collection is growing! One last thing — a friend suggested I get a laptop for next semester, but that's way out of my price range. What should I actually get instead?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_0", step=15,
        user_message=(
            "You're right, I'll skip the laptop. Got a Water Bottle for summer. Looking back at everything, what are the absolute best value picks for a broke college student?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="T-Shirt",
    ),
]

# ============================================================================
# User 1: Margaret Liu — budget_sensitivity: tight
# Retired schoolteacher, fixed pension, extremely frugal.
# Choices: exclusively tier 1-2
# ============================================================================

_USER_1_STEPS = [
    RetailStep(
        user_id="User_1", step=1,
        user_message=(
            "I'm retired and living on a fixed pension. I need to be very careful with spending — nothing over $80 if I can help it. What everyday items do you recommend?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_1", step=2,
        user_message=(
            "I picked up the T-Shirt, very comfortable. My doctor says I should walk more for my health. Do you have affordable walking shoes? Nothing fancy."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_1", step=3,
        user_message=(
            "Those Running Shoes are nice for my morning walks. I need a good water bottle to bring along — something that keeps water cool in summer."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Water Bottle"]),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_1", step=4,
        user_message=(
            "The Water Bottle is perfect for my walks. My grandchildren are visiting next month — I want to get them small gifts. I have maybe $40-50 per child. Suggestions?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Socks",
    ),
    RetailStep(
        user_id="User_1", step=5,
        user_message=(
            "Socks for the grandkids — practical and they come in fun colors. I'm organizing my things and could use a small bag for my daily outings. What's affordable?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_1", step=6,
        user_message=(
            "That Backpack is lighter than I expected — good for groceries. My neighbor mentioned those smart watches for tracking steps. Are they worth it on a tight budget?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_1", step=7,
        user_message=(
            "You're right, I don't need a smartwatch. I got a Phone Case for my old phone instead. The weather is getting warm — I need lightweight clothing. What do you have?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_1", step=8,
        user_message=(
            "Another T-Shirt, can't go wrong. My walking shoes from step 2 are wearing out faster than I expected. Should I get the same pair or try something different?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_1", step=9,
        user_message=(
            "Same Running Shoes, they work fine. My daughter suggested I get a laptop to video-call the family. What would you suggest?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_1", step=10,
        user_message=(
            "No laptop for me — I'll use the library computers. I got a Water Bottle as a thank-you gift for my neighbor instead. I need a new bag for my book club materials."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_1", step=11,
        user_message=(
            "The Backpack fits my books perfectly. It's getting cold now — I need something warm but inexpensive. What can you suggest?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_1", step=12,
        user_message=(
            "Got a thicker T-Shirt this time. My Water Bottle from before has a small crack — I need a replacement. Same one is fine."
        ),
        ground_truth_ranking=_compute_gt_ranking("tight", ["Water Bottle"]),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_1", step=13,
        user_message=(
            "New Water Bottle works great. The grandkids are asking about headphones for their tablets. Are there any I can afford, or should I look for other gifts?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_1", step=14,
        user_message=(
            "Headphones were too expensive, so I got T-Shirts again. My morning walk shoes are holding up well this time. What else would be practical for an active retiree on a budget?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_1", step=15,
        user_message=(
            "I've been using the Backpack for everything now. Looking back at all my purchases, I think I've done well staying within budget. What are your final recommendations for someone like me?"
        ),
        ground_truth_ranking=_compute_gt_ranking("tight"),
        user_choice="Water Bottle",
    ),
]

# ============================================================================
# User 2: David Patel — budget_sensitivity: moderate
# Family man, household budget $300-500/month, value-oriented.
# Choices: tier 2-3 (Running Shoes, Backpack, Smart Watch, Headphones)
# ============================================================================

_USER_2_STEPS = [
    RetailStep(
        user_id="User_2", step=1,
        user_message=(
            "I'm shopping for my family of four. We have a moderate budget — roughly $300-500 per month for household items. I want good value, not the cheapest or most expensive. What do you suggest?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_2", step=2,
        user_message=(
            "Got the Backpack for my son's school — great quality for the price. Now I need running shoes. I jog three times a week and want something mid-range that'll last."
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_2", step=3,
        user_message=(
            "Those Running Shoes feel great on my jogs. My wife's been asking about a smartwatch for her fitness routine. Something with good features but not overpriced?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Smart Watch"]),
        user_choice="Smart Watch",
    ),
    RetailStep(
        user_id="User_2", step=4,
        user_message=(
            "My wife loves the Smart Watch! I work from home sometimes and need headphones for video calls. What's the best value option without breaking the bank?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Headphones"]),
        user_choice="Headphones",
    ),
    RetailStep(
        user_id="User_2", step=5,
        user_message=(
            "The Headphones are excellent for calls. My daughter started yoga classes at school — she needs a yoga mat. Something mid-range and durable?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Yoga Mat"]),
        user_choice="Yoga Mat",
    ),
    RetailStep(
        user_id="User_2", step=6,
        user_message=(
            "The Yoga Mat is great quality for the price. I've been thinking about upgrading our vacuum cleaner — the old one is dying. What do you think?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Desk Lamp",
    ),
    RetailStep(
        user_id="User_2", step=7,
        user_message=(
            "I held off on the vacuum and got a Desk Lamp for my son's study area instead. Both kids need new shirts for school — anything comfortable and affordable?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_2", step=8,
        user_message=(
            "T-Shirts ordered! My wife saw the Smart Watch I got her and now wants a fitness tracker for our son too. Is the same watch a good deal at this price point?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Smart Watch"]),
        user_choice="Smart Watch",
    ),
    RetailStep(
        user_id="User_2", step=9,
        user_message=(
            "Two Smart Watches in the family now! I need a new pair of running shoes — my current ones from step 2 have about 400 miles on them. Same model or something new?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_2", step=10,
        user_message=(
            "Went with the same Running Shoes, they're proven. My wife wants a fitness tracker for her morning runs. Something in the mid-range — not as expensive as the smartwatch?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Fitness Tracker"]),
        user_choice="Fitness Tracker",
    ),
    RetailStep(
        user_id="User_2", step=11,
        user_message=(
            "The Fitness Tracker is great — she loves the step counting. The kids' backpacks are falling apart too — time for new ones?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_2", step=12,
        user_message=(
            "New Backpacks for both kids. Our family camping trip is next month — we need water bottles, t-shirts, and maybe some gear. What's the best value bundle?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Water Bottle", "T-Shirt"]),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_2", step=13,
        user_message=(
            "Water Bottles for the whole family! We also need extra t-shirts for the camping trip. Anything moisture-wicking in a reasonable price range?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_2", step=14,
        user_message=(
            "Got the T-Shirts, perfect for camping. My wife's been hinting at a digital camera for family photos. What should I do?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Smart Watch",
    ),
    RetailStep(
        user_id="User_2", step=15,
        user_message=(
            "Right, we'll use our phones for photos. The Smart Watch was a better use of money. Overall, what do you think are the best value products for a family like ours?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Backpack",
    ),
]

# ============================================================================
# User 3: Lisa Nakamura — budget_sensitivity: moderate
# Working professional, balanced lifestyle, willing to spend mid-range.
# Choices: tier 2-3 (Running Shoes, Smart Watch, Headphones, Backpack)
# ============================================================================

_USER_3_STEPS = [
    RetailStep(
        user_id="User_3", step=1,
        user_message=(
            "I'm a product manager with a decent salary. I don't mind spending on quality but I'm not the type to go premium on everything. Let's say moderate budget. What's good?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Headphones",
    ),
    RetailStep(
        user_id="User_3", step=2,
        user_message=(
            "The Headphones are amazing for my commute. I started training for a half marathon — need proper running shoes. Something reliable in the mid-price range?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_3", step=3,
        user_message=(
            "Running Shoes are perfect for training. I want a smartwatch to track my runs and heart rate. The mid-tier option looked good — thoughts?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Smart Watch"]),
        user_choice="Smart Watch",
    ),
    RetailStep(
        user_id="User_3", step=4,
        user_message=(
            "Love the Smart Watch, it syncs with my training app. I need a good commuter backpack — something that fits my laptop and gym clothes. Not too pricey."
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_3", step=5,
        user_message=(
            "The Backpack is perfect for work-gym transitions. I want a bluetooth speaker for the office — something decent for music during work. Not too expensive though."
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Bluetooth Speaker"]),
        user_choice="Bluetooth Speaker",
    ),
    RetailStep(
        user_id="User_3", step=6,
        user_message=(
            "The Bluetooth Speaker is decent for the price. My water bottle broke during a run — need a new one. Quick recommendation?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Water Bottle"]),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_3", step=7,
        user_message=(
            "Got a sturdy Water Bottle, no more leaks. I'm buying team t-shirts for our department's charity run. Need about 5 of them — what's comfortable and affordable?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_3", step=8,
        user_message=(
            "T-Shirts ordered for the team. My Headphones from step 1 are starting to lose padding. Time for a replacement or should I try a different brand?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Headphones"]),
        user_choice="Headphones",
    ),
    RetailStep(
        user_id="User_3", step=9,
        user_message=(
            "Same Headphones, can't beat the noise cancellation. My half-marathon is next month — should I get a dedicated fitness tracker for race-day heart rate monitoring?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Fitness Tracker"]),
        user_choice="Fitness Tracker",
    ),
    RetailStep(
        user_id="User_3", step=10,
        user_message=(
            "The Fitness Tracker is perfect for race monitoring. A coworker has a laptop and says it's essential. I already have a work laptop though. Thoughts?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_3", step=11,
        user_message=(
            "No personal laptop needed — got a Backpack instead for weekend hikes. The half-marathon is in two weeks. Any last-minute gear recommendations that won't break the bank?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Running Shoes", "Water Bottle"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_3", step=12,
        user_message=(
            "Fresh Running Shoes for race day! I also need to bring hydration — the Water Bottle I got before is perfect. But my partner wants one too. Same recommendation?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["Water Bottle"]),
        user_choice="Water Bottle",
    ),
    RetailStep(
        user_id="User_3", step=13,
        user_message=(
            "Two Water Bottles ready for race day. After the marathon I want to treat myself — a digital camera for race photos? Or is that outside reasonable spending?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Headphones",
    ),
    RetailStep(
        user_id="User_3", step=14,
        user_message=(
            "Camera was too pricey, got replacement Headphones for gym sessions instead. Post-marathon recovery means lots of casual walks. Need comfortable everyday clothes."
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate", ["T-Shirt"]),
        user_choice="T-Shirt",
    ),
    RetailStep(
        user_id="User_3", step=15,
        user_message=(
            "T-Shirt for recovery walks. Looking at my buying history, I think I've been pretty balanced. What are your top picks for someone with a moderate, value-focused budget?"
        ),
        ground_truth_ranking=_compute_gt_ranking("moderate"),
        user_choice="Smart Watch",
    ),
]

# ============================================================================
# User 4: James Wilson — budget_sensitivity: flexible
# Tech executive, money not an issue, quality and performance first.
# Choices: tier 3-5 (Smart Watch, Headphones, Office Chair, Laptop, Camera)
# ============================================================================

_USER_4_STEPS = [
    RetailStep(
        user_id="User_4", step=1,
        user_message=(
            "I'm a tech exec and I want the best products regardless of price. Quality, performance, and durability are what I care about. Show me your top-tier recommendations."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible"),
        user_choice="Laptop",
    ),
    RetailStep(
        user_id="User_4", step=2,
        user_message=(
            "Ordered the Laptop — exactly what I needed for my home office. Now I need a premium office chair. I spend 10+ hours at my desk, so ergonomics are critical."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Office Chair"]),
        user_choice="Office Chair",
    ),
    RetailStep(
        user_id="User_4", step=3,
        user_message=(
            "The Office Chair is a game-changer for my back. I'm getting into photography — what's the best camera you carry? I want professional-grade image quality."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Digital Camera"]),
        user_choice="Digital Camera",
    ),
    RetailStep(
        user_id="User_4", step=4,
        user_message=(
            "The Digital Camera takes stunning photos. For my long flights to conferences, I need top-of-the-line noise-cancelling headphones. What's the best pair?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Headphones"]),
        user_choice="Headphones",
    ),
    RetailStep(
        user_id="User_4", step=5,
        user_message=(
            "Those Headphones are incredible — best noise cancellation I've experienced. I want a premium smartwatch that integrates with everything. Apple or otherwise, just the best."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Smart Watch"]),
        user_choice="Smart Watch",
    ),
    RetailStep(
        user_id="User_4", step=6,
        user_message=(
            "Smart Watch syncs perfectly with my phone and laptop. I need a standing desk for my home office — I heard sitting all day is terrible. What's the best option?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Standing Desk"]),
        user_choice="Standing Desk",
    ),
    RetailStep(
        user_id="User_4", step=7,
        user_message=(
            "The Standing Desk is amazing — my back already feels better. I picked up running for cardio — need premium running shoes. Performance and comfort, no budget limits."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Running Shoes"]),
        user_choice="Running Shoes",
    ),
    RetailStep(
        user_id="User_4", step=8,
        user_message=(
            "Love the Running Shoes. My wife wants a nice travel backpack for our trips — something sleek and well-made. Top quality."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Backpack"]),
        user_choice="Backpack",
    ),
    RetailStep(
        user_id="User_4", step=9,
        user_message=(
            "The Backpack is surprisingly premium. I need a gaming console for the den — something high-end for entertaining guests. Which is the best right now?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Gaming Console"]),
        user_choice="Gaming Console",
    ),
    RetailStep(
        user_id="User_4", step=10,
        user_message=(
            "The Gaming Console is incredible — great for downtime. My wife wants a tablet for her reading and sketching. What's the best premium option?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Tablet"]),
        user_choice="Tablet",
    ),
    RetailStep(
        user_id="User_4", step=11,
        user_message=(
            "The Tablet is perfect for her. I need an air purifier for my home office — allergies are killing me. Best premium option?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Air Purifier"]),
        user_choice="Air Purifier",
    ),
    RetailStep(
        user_id="User_4", step=12,
        user_message=(
            "The Air Purifier makes a huge difference. My Digital Camera needs a proper carrying case — or should I just get a second camera body for video? Money's not an issue."
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Digital Camera"]),
        user_choice="Digital Camera",
    ),
    RetailStep(
        user_id="User_4", step=13,
        user_message=(
            "Second Digital Camera for video — this one's great too. My home Office Chair is so good I want one for my actual office at work. Same exact chair?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Office Chair"]),
        user_choice="Office Chair",
    ),
    RetailStep(
        user_id="User_4", step=14,
        user_message=(
            "Two Office Chairs, both amazing. I need a high-end vacuum cleaner — my cleaning service quit and my husky sheds everywhere. Best option?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible", ["Vacuum Cleaner"]),
        user_choice="Vacuum Cleaner",
    ),
    RetailStep(
        user_id="User_4", step=15,
        user_message=(
            "Got an upgraded Vacuum Cleaner for the upstairs. Looking at everything I've bought, quality really does matter. What are your ultimate premium recommendations?"
        ),
        ground_truth_ranking=_compute_gt_ranking("flexible"),
        user_choice="Laptop",
    ),
]



# ============================================================================
# Users 5-10: drafted with Claude Sonnet 4.6 from fixed rules, validated (choices
# within budget, no item prices, each turn cites the previous choice) and then frozen.
# ============================================================================
_USER_5_STEPS = [
    RetailStep(
        user_id='User_5', step=1,
        user_message="Hi, I'm Carla and I'm on a really tight budget, maybe around fifty dollars a month for personal items. I need everyday basics and something to help me stay active without spending much. What would you recommend?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_5', step=2,
        user_message="I picked up that T-Shirt and it's comfortable for daily wear, exactly what I needed. Now I'm looking for something to keep me hydrated during my morning walks.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_5', step=3,
        user_message='The Water Bottle has been great, I carry it everywhere now. My feet have been sore lately and I could really use some fresh socks for working out.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_5', step=4,
        user_message='Those Socks made a noticeable difference on my walks, much more comfortable. I dropped my phone last week and the screen is scratched, so I need some protection for it.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_5', step=5,
        user_message="Got the Phone Case and my phone feels so much safer now. I've been thinking about getting a proper pair of running shoes to take my walks to the next level.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_5', step=6,
        user_message="The Running Shoes are amazing and I've already started jogging a little. I've heard a lot about yoga mats for stretching at home, do you think it's worth it for a beginner?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_5', step=7,
        user_message="I love the Yoga Mat, I've been doing ten minutes of stretching every morning now. I keep hearing about smartwatches for tracking fitness, do they really make a big difference for casual exercisers?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_5', step=8,
        user_message="Grabbed another pair of Socks since I wear them out fast with all the activity. I'm looking for a bag to carry my water bottle and gear to the park.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_5', step=9,
        user_message="The Backpack fits everything perfectly, it's become my go-to every morning. I need a couple more T-shirts for my workout days since I only have one right now.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_5', step=10,
        user_message='Got another T-Shirt and now I have enough for the whole week. My old phone case is too bulky and I want a slimmer option, something that barely adds any size.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_5', step=11,
        user_message="The new Phone Case is much sleeker, I can actually fit it in my pocket now. I've been reading about laptops and wondering if a good one would help me work from home more easily, what do you think about those?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_5', step=12,
        user_message="Picked up a second Water Bottle so I always have one clean and ready. My desk lamp at home is really dim and it's straining my eyes when I read at night.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_5', step=13,
        user_message="The Desk Lamp is so much better for late-night reading, I wish I'd got one sooner. I want to add a little variety to my workout wardrobe without spending a lot.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_5', step=14,
        user_message="That new T-Shirt is bright and keeps me motivated during runs. I'm curious about gaming consoles because my nephew talks about them all the time, are they hard to find at a good price?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_5', step=15,
        user_message="I decided to grab a second pair of Running Shoes as a backup since I wear them so often now. To wrap up my month I want one last practical item that I'll actually use every day.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
]
_USER_6_STEPS = [
    RetailStep(
        user_id='User_6', step=1,
        user_message="Hey there, I'm Marcus and I'm keeping my spending to about forty dollars a month for clothing and everyday stuff. I want to look decent without overspending, and I prefer items that are durable or eco-conscious. Where should I start?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_6', step=2,
        user_message="Picked up that T-Shirt and it's a clean, simple look I can wear almost anywhere. I need something for my feet since my current socks are falling apart.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_6', step=3,
        user_message='Those Socks are thick and comfortable, exactly what I was after. My phone screen keeps getting scratched in my pocket and I really should do something about it.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_6', step=4,
        user_message='The Phone Case has already saved my screen from two close calls this week. I want to refresh my wardrobe a little more with another basic top.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_6', step=5,
        user_message="Got another T-Shirt in a different color and it rounds out my week nicely. I've been walking a lot more lately and my old shoes just aren't cutting it.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_6', step=6,
        user_message="Those Running Shoes are a game changer for my commute, my feet don't ache at all now. I stay hydrated poorly during the day and I want to fix that with a reusable bottle.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_6', step=7,
        user_message="The Water Bottle is great and I've already cut down on buying plastic bottles, which feels good. I'm curious whether an air purifier is worth the investment for a small apartment, I've seen them advertised a lot.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_6', step=8,
        user_message='Grabbed more Socks because I can never have too many. I want to start stretching in the mornings and I need a mat to work on at home.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_6', step=9,
        user_message='That Yoga Mat has made my mornings so much better, even just ten minutes of stretching helps. I carry a lot of stuff to work and my old bag is really worn out.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_6', step=10,
        user_message='The Backpack is sturdy and fits my laptop and lunch easily. I want another basic tee so I have enough for the whole workweek without doing laundry constantly.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_6', step=11,
        user_message="Got that extra T-Shirt and my wardrobe is finally sorted for the week. My desk at home doesn't have good lighting and it's starting to affect how long I can focus.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_6', step=12,
        user_message="The Desk Lamp made such a difference, I can work in the evenings without straining my eyes. I've been thinking about getting a tablet for reading and light work, are they practical for everyday use?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_6', step=13,
        user_message='Picked up a slimmer Phone Case since the old one was getting bulky. I want something practical to carry my water bottle and gym gear without cramming it into my work bag.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_6', step=14,
        user_message="Got a second Backpack, a smaller one just for the gym, and it keeps things nicely separated. I want to add one more item this month that I'll actually reach for every single day.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_6', step=15,
        user_message="Picked up another pack of Socks to round out the month on a practical note. I feel like I've built a solid, no-nonsense set of basics without going over budget.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
]
_USER_7_STEPS = [
    RetailStep(
        user_id='User_7', step=1,
        user_message="Hi, I'm Dana and I'm working with a tight budget of around sixty dollars a month. I want to get a few fitness items for myself and some small gifts for friends without going overboard. What would you suggest?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_7', step=2,
        user_message="Bought that Water Bottle and already started carrying it to work every day. My friend's birthday is coming up and I want something small and useful to give her.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_7', step=3,
        user_message="She loved the Socks, said they're super soft and she'd been meaning to buy some herself. For my own fitness I need something to protect my hands and phone when I exercise outdoors.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_7', step=4,
        user_message='The Phone Case clips onto my armband perfectly and my phone feels secure during runs. I want a new top to wear to my workout sessions since mine are all old and stretched.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_7', step=5,
        user_message='That T-Shirt is lightweight and breathes really well when I run. My brother mentioned he needs a good bag for school and I want to get him something durable as a surprise.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_7', step=6,
        user_message="He was thrilled with the Backpack, said it's way better than what he had. I want to add a proper mat to my home routine so I can do yoga and cool-down stretches.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_7', step=7,
        user_message="The Yoga Mat is non-slip and the right thickness, very happy with it. I've been wondering about fitness trackers and whether they're actually useful for someone just getting into exercise.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_7', step=8,
        user_message='Those Running Shoes are incredibly supportive, my knees feel much better on longer runs. My study space is poorly lit and I really need to fix that before exam season.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_7', step=9,
        user_message='The Desk Lamp is bright without being harsh, exactly what I needed for late studying. I want a small gift for my coworker who keeps losing her earbuds in her bag.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_7', step=10,
        user_message='She loved having a phone case with a card slot so she can keep things together. I need another T-shirt because I sweat through mine quickly during intense workouts.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_7', step=11,
        user_message="The second T-Shirt is great for back-to-back workout days. I've been reading about gaming consoles and my little cousin keeps asking for one, are they reasonable to consider as a gift?",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_7', step=12,
        user_message='Got a second Water Bottle so I always have one at home and one in my bag. I want something for a friend who just started running and needs a practical gift to get going.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_7', step=13,
        user_message="She was so happy with the Socks, said they're perfect for her morning runs. I want to pick up something for myself that I'll use in every single workout session.",
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_7', step=14,
        user_message='Got a second Yoga Mat so I can keep one rolled up in my bag for the gym. I want to close out this month with one more gift idea for a friend who always complains about cold feet at the office.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_7', step=15,
        user_message='Those cozy Socks were a hit and she wore them to the office the very next day. I feel like I managed to cover fitness and gifting this month without blowing my budget at all.',
        ground_truth_ranking=_compute_gt_ranking('tight'),
        user_choice='T-Shirt',
    ),
]
_USER_8_STEPS = [
    RetailStep(
        user_id='User_8', step=1,
        user_message="Hello, I'm Priya and I set aside roughly a hundred and fifty dollars a month for home and tech purchases. I like to be mindful of the environment, so I lean toward items that are built to last. What would be a good place to start?",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_8', step=2,
        user_message="The Desk Lamp arrived and it's made my home office so much more pleasant to work in. I want to add something that helps me stay focused and track my daily health habits.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_8', step=3,
        user_message='The Fitness Tracker is motivating me to hit my step goals every day, I love the sleep tracking too. I need a reliable bag for carrying my work gear and gym clothes between the office and the gym.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_8', step=4,
        user_message='That Backpack is roomy but not too heavy, perfect for my daily commute. My headphones broke last week and I need something decent for calls and listening to podcasts at work.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Headphones',
    ),
    RetailStep(
        user_id='User_8', step=5,
        user_message="The Headphones have great sound quality and the call clarity is excellent for my remote meetings. I've been wanting to improve my mornings with a proper fitness routine at home.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_8', step=6,
        user_message="The Yoga Mat is thick enough for floor exercises and I've been using it every morning before work. I also want something practical for staying hydrated throughout the day since I keep forgetting to drink water.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_8', step=7,
        user_message="The insulated Water Bottle keeps my drink cold for hours, it's been a great reminder to hydrate. I want a speaker for my kitchen so I can listen to music while I cook in the evenings.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_8', step=8,
        user_message="The Bluetooth Speaker fills the kitchen with great sound and it was easy to pair with my phone. I've been reading about standing desks and wondering if they are genuinely worth it for someone who sits eight hours a day.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_8', step=9,
        user_message="I got a second Fitness Tracker as a gift for my sister who just started a health kick. I want to upgrade my running kit now and I've heard the right shoes make a huge difference.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_8', step=10,
        user_message="Those Running Shoes are incredibly light and my long runs feel so much easier. I want to protect my phone properly since I carry it everywhere and it's taken a few knocks recently.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_8', step=11,
        user_message='The Phone Case is slim but sturdy and fits perfectly in my pocket. I want to add one more comfortable clothing item for my workout days.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_8', step=12,
        user_message="That T-Shirt is great quality and I wore it on my last three runs. I've been curious about smart watches and whether they offer a meaningful step up over a basic fitness tracker.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Smart Watch',
    ),
    RetailStep(
        user_id='User_8', step=13,
        user_message='The Smart Watch is impressive and I love having notifications and maps on my wrist during runs. I want to improve my sound setup at home even more, something for the living room this time.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_8', step=14,
        user_message='Got a larger Bluetooth Speaker for the living room and the sound quality is wonderful for movie nights. I keep hearing about laptops with great battery life for remote work, are they a sensible upgrade for someone who moves around a lot?',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Headphones',
    ),
    RetailStep(
        user_id='User_8', step=15,
        user_message='Picked up a second pair of Headphones as a backup for travel since I hate being without music. This month has been really productive for upgrading both my home and my daily routine.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Desk Lamp',
    ),
]
_USER_9_STEPS = [
    RetailStep(
        user_id='User_9', step=1,
        user_message="Hey, I'm Trevor and I usually spend around two hundred dollars a month on hobbies and tech. I tend to stick with brands I know and trust, and my apartment isn't huge so bulky items are a problem. What do you recommend?",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_9', step=2,
        user_message='That Bluetooth Speaker is compact and the sound is much better than my old one, really glad I went with it. I do a lot of work from home and my current headphones are pretty terrible for calls.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Headphones',
    ),
    RetailStep(
        user_id='User_9', step=3,
        user_message="The Headphones are crystal clear and the noise cancellation is perfect for my open-plan living space. I want something to track my activity since I've started cycling to work.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_9', step=4,
        user_message='The Fitness Tracker is easy to read while cycling and the battery lasts ages. I need a decent bag that can carry my laptop, lunch and gym gear all at once.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_9', step=5,
        user_message="The Backpack has the right number of compartments and nothing digs into my back on the cycle. I want a proper mat for the stretching routine I've been meaning to start.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_9', step=6,
        user_message="The Yoga Mat rolls up tightly and doesn't take much space, which was my main worry. My phone screen cracked last week and I really should have been using a case.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_9', step=7,
        user_message='The Phone Case is reinforced at the corners and feels solid, lesson learned. I want to upgrade my watch situation because my fitness tracker is good but I want something smarter.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Smart Watch',
    ),
    RetailStep(
        user_id='User_9', step=8,
        user_message="The Smart Watch is fantastic, I especially love being able to reply to messages without taking out my phone. I keep seeing ads for digital cameras and I'm wondering whether they produce noticeably better photos than a phone.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_9', step=9,
        user_message="Picked up a second Fitness Tracker for my dad since he wanted to start tracking his steps. My desk setup at home needs better lighting now that I'm doing more evening work.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_9', step=10,
        user_message="The Desk Lamp is adjustable and the color temperature is perfect for long focus sessions. I want to get back into running and I need shoes that won't hurt my knees.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_9', step=11,
        user_message="Those Running Shoes feel like clouds and I've already run four times this week. I need a fresh everyday tee, something plain and comfortable for weekends.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_9', step=12,
        user_message='That T-Shirt is soft and fits well, exactly what I wanted for a casual weekend look. I want to upgrade the audio in my bedroom for late-night listening sessions.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_9', step=13,
        user_message="Got a second Bluetooth Speaker for the bedroom and the stereo pairing with the other one is brilliant. I've been thinking a lot about tablets lately, I wonder if one would replace my laptop for light tasks.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Smart Watch',
    ),
    RetailStep(
        user_id='User_9', step=14,
        user_message='Picked up a second Smart Watch as a gift for my girlfriend who loved seeing mine. I want to grab one last practical item to round out my month.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_9', step=15,
        user_message="The Water Bottle is insulated and keeps my coffee warm on the morning cycle, great little addition. I feel like this month I've genuinely improved both my tech setup and my daily habits.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Headphones',
    ),
]
_USER_10_STEPS = [
    RetailStep(
        user_id='User_10', step=1,
        user_message="Hi, I'm Simone and I have about a hundred and twenty dollars a month to spend on fitness and gifts. I try to choose products that are durable and responsibly made whenever possible. What would you suggest to get started?",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Yoga Mat',
    ),
    RetailStep(
        user_id='User_10', step=2,
        user_message="I love the Yoga Mat, it's thick enough for joints and has a nice grip surface. I want to get my brother something useful for his new jogging habit.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Water Bottle',
    ),
    RetailStep(
        user_id='User_10', step=3,
        user_message="He was really happy with the Water Bottle and he's been using it every morning. I need a good pair of shoes for my own running since mine are completely worn out.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_10', step=4,
        user_message="Those Running Shoes are responsive and my calves feel great after longer runs now. I want something to track my progress since I've started following a training plan.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_10', step=5,
        user_message='The Fitness Tracker has made me much more aware of how little I move during work hours, very eye-opening. I want to find a nice gift for my mum who keeps complaining her phone gets scratched in her handbag.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Phone Case',
    ),
    RetailStep(
        user_id='User_10', step=6,
        user_message="Mum loved the Phone Case and said it's finally given her peace of mind. I need a bag for the gym that's also decent enough to carry to work on busy days.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Backpack',
    ),
    RetailStep(
        user_id='User_10', step=7,
        user_message="The Backpack fits my gym kit and my work stuff with room to spare, it's become my everyday bag. I've been thinking about getting a smart watch, do you think it's a worthwhile step up from a basic fitness tracker?",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Smart Watch',
    ),
    RetailStep(
        user_id='User_10', step=8,
        user_message='The Smart Watch is brilliant for tracking my heart rate during interval runs, really happy with the upgrade. I want to get a friend something for her home office setup as a birthday gift.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Desk Lamp',
    ),
    RetailStep(
        user_id='User_10', step=9,
        user_message="She said the Desk Lamp is the best gift she's received in years and it's transformed her workspace. I want to add some music to my workouts at home and something portable would be ideal.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_10', step=10,
        user_message="The Bluetooth Speaker is light enough to move room to room and the battery lasts all day. I need some fresh workout tops since I'm training more frequently now.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='T-Shirt',
    ),
    RetailStep(
        user_id='User_10', step=11,
        user_message='Those T-Shirts are great for high-intensity sessions, they dry quickly too. I want a gift for my nephew who is starting college and needs everyday basics.',
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Socks',
    ),
    RetailStep(
        user_id='User_10', step=12,
        user_message="He was thrilled with the Socks, said he needed them badly and appreciated the thoughtfulness. I've been curious about whether a vacuum cleaner could be a good housewarming gift for a friend who just moved in.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Fitness Tracker',
    ),
    RetailStep(
        user_id='User_10', step=13,
        user_message="Grabbed a second Fitness Tracker for my running partner so we can compare stats, she's very competitive. I want to pick up one more item for myself that will support my training directly.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Running Shoes',
    ),
    RetailStep(
        user_id='User_10', step=14,
        user_message="Got a second pair of Running Shoes as a rotation pair so my main ones can dry between sessions. I need one last gift idea for my dad who enjoys music but doesn't have anything to play it on at home.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Bluetooth Speaker',
    ),
    RetailStep(
        user_id='User_10', step=15,
        user_message="He absolutely loved the Bluetooth Speaker and has it going all evening apparently. I'm really pleased with how this month balanced my own fitness needs with finding thoughtful gifts for people I care about.",
        ground_truth_ranking=_compute_gt_ranking('moderate'),
        user_choice='Yoga Mat',
    ),
]

# ============================================================================
# Dataset class
# ============================================================================

_ALL_USERS = {
    "User_0": RetailUser(
        user_id="User_0",
        name="Alex Chen",
        budget_sensitivity="tight",
        initial_goals=["daily essentials", "fitness gear"],
        initial_constraints=["strict monthly budget"],
        steps=_USER_0_STEPS,
    ),
    "User_1": RetailUser(
        user_id="User_1",
        name="Margaret Liu",
        budget_sensitivity="tight",
        initial_goals=["daily essentials", "gift shopping"],
        initial_constraints=["strict monthly budget"],
        steps=_USER_1_STEPS,
    ),
    "User_2": RetailUser(
        user_id="User_2",
        name="David Patel",
        budget_sensitivity="moderate",
        initial_goals=["daily essentials", "fitness gear"],
        initial_constraints=["strict monthly budget", "limited storage space"],
        steps=_USER_2_STEPS,
    ),
    "User_3": RetailUser(
        user_id="User_3",
        name="Lisa Nakamura",
        budget_sensitivity="moderate",
        initial_goals=["fitness gear", "tech upgrade"],
        initial_constraints=["eco-friendly preference"],
        steps=_USER_3_STEPS,
    ),
    "User_4": RetailUser(
        user_id="User_4",
        name="James Wilson",
        budget_sensitivity="flexible",
        initial_goals=["tech upgrade", "home improvement"],
        initial_constraints=["eco-friendly preference", "brand loyalty"],
        steps=_USER_4_STEPS,
    ),
    'User_5': RetailUser(
        user_id='User_5',
        name='Carla',
        budget_sensitivity='tight',
        initial_goals=['daily essentials', 'fitness gear'],
        initial_constraints=['strict monthly budget', 'limited storage space'],
        steps=_USER_5_STEPS,
    ),
    'User_6': RetailUser(
        user_id='User_6',
        name='Marcus',
        budget_sensitivity='tight',
        initial_goals=['fashion update', 'daily essentials'],
        initial_constraints=['strict monthly budget', 'eco-friendly preference'],
        steps=_USER_6_STEPS,
    ),
    'User_7': RetailUser(
        user_id='User_7',
        name='Dana',
        budget_sensitivity='tight',
        initial_goals=['fitness gear', 'gift shopping'],
        initial_constraints=['strict monthly budget', 'limited storage space'],
        steps=_USER_7_STEPS,
    ),
    'User_8': RetailUser(
        user_id='User_8',
        name='Priya',
        budget_sensitivity='moderate',
        initial_goals=['home improvement', 'tech upgrade'],
        initial_constraints=['strict monthly budget', 'eco-friendly preference'],
        steps=_USER_8_STEPS,
    ),
    'User_9': RetailUser(
        user_id='User_9',
        name='Trevor',
        budget_sensitivity='moderate',
        initial_goals=['hobby supplies', 'tech upgrade'],
        initial_constraints=['brand loyalty', 'limited storage space'],
        steps=_USER_9_STEPS,
    ),
    'User_10': RetailUser(
        user_id='User_10',
        name='Simone',
        budget_sensitivity='moderate',
        initial_goals=['fitness gear', 'gift shopping'],
        initial_constraints=['strict monthly budget', 'eco-friendly preference'],
        steps=_USER_10_STEPS,
    ),
}


class RetailDataset:
    """In-memory dataset of handcrafted retail user personas."""

    def __init__(self) -> None:
        self._users = dict(_ALL_USERS)

    def users(self) -> List[str]:
        return sorted(self._users.keys())

    def get_user(self, user_id: str) -> RetailUser:
        return self._users[user_id]

    def __len__(self) -> int:
        return len(self._users)
