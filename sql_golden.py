"""
sql_golden.py - the GOLDEN text-to-SQL dataset for HexaMobile.

Each entry = one English question + the reference ("gold") SQL that answers it.
This file is the single source of truth for what "correct" means. sql_make_golden.py
runs each gold query and freezes its result into sql_golden.csv, which the eval then
compares model output against (EXECUTION accuracy: same data = correct).

Difficulty gradient (like the IPL project's easy/hard/brutal):
  easy   (1-8)   : single table, one aggregate or filter
  medium (9-16)  : joins, GROUP BY, ORDER BY ... LIMIT, averages
  hard   (17-24) : ratios/percentages, HAVING thresholds, NULL traps, anti-joins,
                   multi-join with the plans catalogue

Convention (borrowed from the IPL project): "which/who" questions put name+value in
the gold, and the evaluator tolerates the model returning value-only or name+value.

Run this file directly to VALIDATE every gold query against telco.db:
  python sql_golden.py
"""

GOLDEN = [
    # ==================== EASY ====================
    dict(id=1, diff="easy",
         q="How many customers are there in total?",
         sql="SELECT COUNT(*) AS n FROM customers"),

    dict(id=2, diff="easy",
         q="How many customers are currently active?",
         sql="SELECT COUNT(*) AS n FROM customers WHERE status = 'active'"),

    dict(id=3, diff="easy",
         q="How many support tickets have the status 'open'?",
         sql="SELECT COUNT(*) AS n FROM tickets WHERE status = 'open'"),

    dict(id=4, diff="easy",
         q="What is the monthly price of the Unlimited Max plan?",
         sql="SELECT monthly_price FROM plans WHERE plan_name = 'Unlimited Max'"),

    dict(id=5, diff="easy",
         q="How many customers are in the Ile-de-France region?",
         sql="SELECT COUNT(*) AS n FROM customers WHERE region = 'Ile-de-France'"),

    dict(id=6, diff="easy",
         q="How many postpaid plans does HexaMobile offer?",
         sql="SELECT COUNT(*) AS n FROM plans WHERE contract_type = 'postpaid'"),

    dict(id=7, diff="easy",
         q="How many customers have churned?",
         sql="SELECT COUNT(*) AS n FROM customers WHERE status = 'churned'"),

    dict(id=8, diff="easy",
         q="What is the highest monthly price among all plans?",
         sql="SELECT MAX(monthly_price) AS max_price FROM plans"),

    # ==================== MEDIUM ====================
    dict(id=9, diff="medium",
         q="How many customers are on each plan? Give the plan name and the customer count.",
         sql="""SELECT p.plan_name, COUNT(*) AS n
                FROM customers c JOIN plans p ON c.plan_id = p.plan_id
                GROUP BY p.plan_name"""),

    dict(id=10, diff="medium",
         q="Which region has the most customers? Give the region and the count.",
         sql="""SELECT region, COUNT(*) AS n FROM customers
                GROUP BY region ORDER BY n DESC LIMIT 1"""),

    dict(id=11, diff="medium",
         q="What is the total revenue from bills that were paid?",
         sql="SELECT SUM(amount) AS revenue FROM bills WHERE status = 'paid'"),

    dict(id=12, diff="medium",
         q="What is the average resolution time in hours for resolved tickets?",
         sql="SELECT AVG(resolution_hours) AS avg_hours FROM tickets WHERE status = 'resolved'"),

    dict(id=13, diff="medium",
         q="List the top 3 regions by number of customers, most first. Give the region and the count.",
         sql="""SELECT region, COUNT(*) AS n FROM customers
                GROUP BY region ORDER BY n DESC LIMIT 3"""),

    dict(id=14, diff="medium",
         q="How many tickets are there in each category? Give the category and the count.",
         sql="SELECT category, COUNT(*) AS n FROM tickets GROUP BY category"),

    dict(id=15, diff="medium",
         q="What is the average bill amount across all bills?",
         sql="SELECT AVG(amount) AS avg_amount FROM bills"),

    dict(id=16, diff="medium",
         q="Which plan generates the most total revenue from paid bills? Give the plan name and the revenue.",
         sql="""SELECT p.plan_name, SUM(b.amount) AS revenue
                FROM bills b
                JOIN customers c ON b.customer_id = c.customer_id
                JOIN plans p ON c.plan_id = p.plan_id
                WHERE b.status = 'paid'
                GROUP BY p.plan_name ORDER BY revenue DESC LIMIT 1"""),

    # ==================== HARD ====================
    dict(id=17, diff="hard",
         q="Which month had the highest roaming usage rate (the percentage of usage records with roaming used)? Give the month and the percentage.",
         sql="""SELECT month, 100.0 * SUM(roaming_used) / COUNT(*) AS pct
                FROM usage GROUP BY month ORDER BY pct DESC LIMIT 1"""),

    dict(id=18, diff="hard",
         q="What is the churn rate, i.e. the percentage of all customers whose status is churned? Give a single number.",
         sql="""SELECT 100.0 * SUM(CASE WHEN status = 'churned' THEN 1 ELSE 0 END) / COUNT(*) AS churn_pct
                FROM customers"""),

    dict(id=19, diff="hard",
         q="Which region has the highest churn rate among regions with at least 100 customers? Give the region and the churn percentage.",
         sql="""SELECT region, 100.0 * SUM(CASE WHEN status = 'churned' THEN 1 ELSE 0 END) / COUNT(*) AS churn_pct
                FROM customers GROUP BY region HAVING COUNT(*) >= 100
                ORDER BY churn_pct DESC LIMIT 1"""),

    dict(id=20, diff="hard",
         q="What is the average monthly data usage in GB for customers on unlimited-data plans?",
         sql="""SELECT AVG(u.data_used_gb) AS avg_gb
                FROM usage u
                JOIN customers c ON u.customer_id = c.customer_id
                JOIN plans p ON c.plan_id = p.plan_id
                WHERE p.is_unlimited_data = 1"""),

    dict(id=21, diff="hard",
         q="Which customer has paid the most in total across all their paid bills? Give the customer id and the total amount.",
         sql="""SELECT customer_id, SUM(amount) AS total
                FROM bills WHERE status = 'paid'
                GROUP BY customer_id ORDER BY total DESC LIMIT 1"""),

    dict(id=22, diff="hard",
         q="What is the average bill amount for customers on the Unlimited Max plan?",
         sql="""SELECT AVG(b.amount) AS avg_amount
                FROM bills b
                JOIN customers c ON b.customer_id = c.customer_id
                JOIN plans p ON c.plan_id = p.plan_id
                WHERE p.plan_name = 'Unlimited Max'"""),

    dict(id=23, diff="hard",
         q="Which support channel has the fastest average resolution time, considering only resolved tickets? Give the channel and the average hours.",
         sql="""SELECT channel, AVG(resolution_hours) AS avg_hours
                FROM tickets WHERE status = 'resolved'
                GROUP BY channel ORDER BY avg_hours ASC LIMIT 1"""),

    dict(id=24, diff="hard",
         q="How many customers have never opened a support ticket?",
         sql="""SELECT COUNT(*) AS n FROM customers
                WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM tickets)"""),
]

# Only questions whose row ORDER is part of the answer (top-N lists).
ORDER_SENSITIVE_IDS = {13}


if __name__ == "__main__":
    import os, sqlite3
    from collections import Counter

    DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco.db")
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    print(f"Validating {len(GOLDEN)} gold queries against {os.path.basename(DB)}\n")

    ok = 0
    for g in GOLDEN:
        try:
            rows = cur.execute(g["sql"]).fetchall()
            preview = rows[0] if rows else "(empty)"
            print(f"[{g['diff']:6}] #{g['id']:2}  OK  rows={len(rows):<3} sample={str(preview)[:60]}")
            ok += 1
        except Exception as e:
            print(f"[{g['diff']:6}] #{g['id']:2}  FAIL  {e}")
    print(f"\n{ok}/{len(GOLDEN)} passed | spread: {dict(Counter(g['diff'] for g in GOLDEN))}")
    conn.close()
