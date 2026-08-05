"""
generate_data.py - build a realistic (but 100% synthetic & free) telecom dataset
for the fictional French mobile operator "HexaMobile".

Why synthetic instead of a Kaggle download?
  - No licensing, no privacy/GDPR concerns (no real people).
  - We control the shape, so we can design SQL questions with known answers.
  - It is fully reproducible: every run with the same SEED yields the same rows,
    which is CRITICAL - our golden SQL results must never drift between runs.

Output: five CSV files written next to this script (in the data/ folder):
  plans.csv, customers.csv, usage.csv, bills.csv, tickets.csv

Run:
  python data/generate_data.py
"""

import os
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# 0. Reproducibility - seed BOTH the stdlib RNG and Faker's RNG.
#    Without this, the data (and therefore our golden answers) would change
#    on every run.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
fake = Faker("fr_FR")     # French names/addresses -> realistic for the French market
Faker.seed(SEED)

# Write CSVs next to THIS file, regardless of the current working directory.
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Reference data: French regions -> representative cities.
#    We keep region<->city consistent (a customer in Lyon is in
#    Auvergne-Rhone-Alpes), which lets us ask clean "by region" questions.
#    Ile-de-France is weighted heavier because it holds ~18% of France's people.
# ---------------------------------------------------------------------------
REGIONS = {
    "Ile-de-France":            ["Paris", "Boulogne-Billancourt", "Saint-Denis", "Versailles", "Nanterre"],
    "Auvergne-Rhone-Alpes":     ["Lyon", "Grenoble", "Saint-Etienne", "Clermont-Ferrand", "Annecy"],
    "Provence-Alpes-Cote d'Azur": ["Marseille", "Nice", "Toulon", "Aix-en-Provence", "Cannes"],
    "Nouvelle-Aquitaine":       ["Bordeaux", "Limoges", "Poitiers", "La Rochelle", "Pau"],
    "Occitanie":                ["Toulouse", "Montpellier", "Nimes", "Perpignan", "Beziers"],
    "Hauts-de-France":          ["Lille", "Amiens", "Roubaix", "Dunkerque", "Calais"],
    "Grand Est":                ["Strasbourg", "Reims", "Metz", "Nancy", "Mulhouse"],
    "Pays de la Loire":         ["Nantes", "Angers", "Le Mans", "Saint-Nazaire", "Cholet"],
    "Bretagne":                 ["Rennes", "Brest", "Quimper", "Lorient", "Vannes"],
    "Normandie":                ["Rouen", "Le Havre", "Caen", "Cherbourg", "Evreux"],
}
REGION_NAMES = list(REGIONS.keys())
# Selection weights (roughly population-shaped): Ile-de-France first.
REGION_WEIGHTS = [30, 14, 12, 10, 10, 9, 8, 8, 6, 6]

# ---------------------------------------------------------------------------
# 2. Plans. NULL is used deliberately for "unlimited" (data_gb / voice_minutes /
#    sms_included) - real catalogues work this way, AND NULLs are a classic
#    text-to-SQL trap, so they make the later model eval more discriminating.
# ---------------------------------------------------------------------------
PLANS = [
    # plan_id, name,            price,  data_gb, voice_min, sms,  unlimited_data, contract
    (1, "Prepaid Mini",          4.99,   2,      120,   200,   0, "prepaid"),
    (2, "Prepaid Plus",          9.99,   20,     None,  None,  0, "prepaid"),   # unlimited calls/sms
    (3, "Essential 5GB",        12.99,   5,      None,  None,  0, "postpaid"),
    (4, "Smart 50GB",           19.99,   50,     None,  None,  0, "postpaid"),
    (5, "Smart 100GB",          24.99,   100,    None,  None,  0, "postpaid"),
    (6, "Unlimited Max",        39.99,   None,   None,  None,  1, "postpaid"),  # unlimited data
    (7, "Family 200GB",         49.99,   200,    None,  None,  0, "postpaid"),
    (8, "Data SIM 30GB",        14.99,   30,     0,     0,     0, "postpaid"),  # data-only: no voice/sms
]
PLAN_IDS = [p[0] for p in PLANS]
# Postpaid plans are more common than prepaid in France; weight accordingly.
PLAN_WEIGHTS = [6, 8, 14, 20, 16, 12, 8, 6]
PRICE_BY_PLAN = {p[0]: p[2] for p in PLANS}
UNLIMITED_DATA_PLANS = {p[0] for p in PLANS if p[6] == 1}
DATA_CAP_BY_PLAN = {p[0]: p[3] for p in PLANS}  # None means unlimited

# 2024 months as (year, month) and as "YYYY-MM" strings.
MONTHS = [(2024, m) for m in range(1, 13)]
def ym_str(y, m): return f"{y}-{m:02d}"


# ---------------------------------------------------------------------------
# 3. Customers
# ---------------------------------------------------------------------------
N_CUSTOMERS = 2000
CHURN_RATE = 0.15   # 15% of customers have left

def make_customers():
    rows = []
    for cid in range(1, N_CUSTOMERS + 1):
        region = random.choices(REGION_NAMES, weights=REGION_WEIGHTS, k=1)[0]
        city = random.choice(REGIONS[region])
        plan_id = random.choices(PLAN_IDS, weights=PLAN_WEIGHTS, k=1)[0]

        # Signup any time from 2021-01-01 to 2024-11-30 (so 2024 has activity).
        signup = fake.date_between(date(2021, 1, 1), date(2024, 11, 30))

        churned = random.random() < CHURN_RATE
        churn_date = None
        status = "active"
        if churned:
            # Churn happens AFTER signup, somewhere up to 2024-12-31.
            churn_date = fake.date_between(signup, date(2024, 12, 31))
            status = "churned"

        rows.append({
            "customer_id": cid,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "age": random.randint(18, 75),
            "city": city,
            "region": region,
            "plan_id": plan_id,
            "signup_date": signup.isoformat(),
            "status": status,
            "churn_date": churn_date.isoformat() if churn_date else None,
        })
    return rows


def customer_active_in(cust, y, m):
    """Was this customer a paying subscriber during month (y, m)?"""
    first_of_month = date(y, m, 1)
    signup = date.fromisoformat(cust["signup_date"])
    if signup > date(y, m, 28):            # signed up after this month
        return False
    if cust["churn_date"]:
        churn = date.fromisoformat(cust["churn_date"])
        if churn < first_of_month:         # already left before this month
            return False
    return True


# ---------------------------------------------------------------------------
# 4. Monthly usage + the matching bill (generated together, month by month).
#    Roaming spikes in summer (Jul/Aug) - a nice seasonal pattern to query.
# ---------------------------------------------------------------------------
def make_usage_and_bills(customers):
    usage_rows, bill_rows = [], []
    usage_id = bill_id = 0

    for cust in customers:
        cid = cust["customer_id"]
        plan_id = cust["plan_id"]
        cap = DATA_CAP_BY_PLAN[plan_id]                 # None = unlimited
        base_price = PRICE_BY_PLAN[plan_id]

        for (y, m) in MONTHS:
            if not customer_active_in(cust, y, m):
                continue
            month = ym_str(y, m)

            # ---- usage ----
            if plan_id in UNLIMITED_DATA_PLANS:
                data_used = round(random.uniform(15, 90), 2)      # heavy users
            elif cap == 0:                                        # data-only edge? no
                data_used = round(random.uniform(1, 30), 2)
            else:
                # Most stay under cap; a minority overshoot it (=> overage fee).
                data_used = round(random.uniform(0.5, cap * 1.2), 2)

            voice_used = random.randint(0, 900)
            sms_used = random.randint(0, 300)

            # Roaming is rare, but far more likely in the summer holidays.
            roam_prob = 0.28 if m in (7, 8) else 0.08
            roaming_used = 1 if random.random() < roam_prob else 0

            usage_id += 1
            usage_rows.append({
                "usage_id": usage_id,
                "customer_id": cid,
                "month": month,
                "data_used_gb": data_used,
                "voice_minutes_used": voice_used,
                "sms_sent": sms_used,
                "roaming_used": roaming_used,
            })

            # ---- bill (derived from usage) ----
            amount = base_price
            # Overage: 2 EUR per GB over the cap (capped/unlimited plans never overage).
            if cap not in (None, 0) and data_used > cap:
                amount += round((data_used - cap) * 2.0, 2)
            # Roaming surcharge.
            if roaming_used:
                amount += round(random.uniform(3, 20), 2)
            amount = round(amount, 2)

            status = random.choices(
                ["paid", "late", "unpaid"], weights=[88, 9, 3], k=1
            )[0]

            bill_id += 1
            bill_rows.append({
                "bill_id": bill_id,
                "customer_id": cid,
                "month": month,
                "amount": amount,
                "status": status,
            })

    return usage_rows, bill_rows


# ---------------------------------------------------------------------------
# 5. Support tickets
# ---------------------------------------------------------------------------
N_TICKETS = 3000
TICKET_CATEGORIES = ["billing", "network", "roaming", "device", "plan_change"]
TICKET_CAT_WEIGHTS = [30, 28, 12, 18, 12]
TICKET_CHANNELS = ["phone", "app", "store", "chat"]

def make_tickets(customers):
    rows = []
    for tid in range(1, N_TICKETS + 1):
        cust = random.choice(customers)
        created = fake.date_between(date(2024, 1, 1), date(2024, 12, 31))
        category = random.choices(TICKET_CATEGORIES, weights=TICKET_CAT_WEIGHTS, k=1)[0]
        status = random.choices(
            ["resolved", "open", "escalated"], weights=[75, 15, 10], k=1
        )[0]
        # Only resolved/escalated tickets have a resolution time.
        resolution_hours = (
            round(random.uniform(0.5, 96), 1) if status != "open" else None
        )
        rows.append({
            "ticket_id": tid,
            "customer_id": cust["customer_id"],
            "created_date": created.isoformat(),
            "category": category,
            "channel": random.choice(TICKET_CHANNELS),
            "status": status,
            "resolution_hours": resolution_hours,
        })
    return rows


# ---------------------------------------------------------------------------
# 6. Write everything to CSV
# ---------------------------------------------------------------------------
def main():
    plans_df = pd.DataFrame(PLANS, columns=[
        "plan_id", "plan_name", "monthly_price", "data_gb",
        "voice_minutes", "sms_included", "is_unlimited_data", "contract_type",
    ])

    customers = make_customers()
    usage, bills = make_usage_and_bills(customers)
    tickets = make_tickets(customers)

    customers_df = pd.DataFrame(customers)
    usage_df = pd.DataFrame(usage)
    bills_df = pd.DataFrame(bills)
    tickets_df = pd.DataFrame(tickets)

    out = {
        "plans.csv": plans_df,
        "customers.csv": customers_df,
        "usage.csv": usage_df,
        "bills.csv": bills_df,
        "tickets.csv": tickets_df,
    }
    for name, df in out.items():
        path = os.path.join(HERE, name)
        df.to_csv(path, index=False)
        print(f"  wrote {name:15s} {len(df):>6} rows")

    print("\nDone. Synthetic HexaMobile dataset generated in data/.")


if __name__ == "__main__":
    main()
