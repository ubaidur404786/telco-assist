"""
rag_golden.py - the golden dataset for the RAG (knowledge) engine.

Each entry has:
  q         : the customer question
  reference : a short reference answer (ground truth, for the judge to compare against)
  source    : the doc that SHOULD answer it (for retrieval hit-rate) - None if the
              knowledge base does not cover it

The last item is DELIBERATELY out-of-scope: HexaMobile's docs say nothing about home
fibre. A good bot must say "I don't have that information" rather than invent an answer.
Testing refusal is how you measure faithfulness / anti-hallucination.
"""

GOLDEN = [
    dict(id=1,
         q="How much does the Unlimited Max plan cost per month?",
         reference="Unlimited Max costs 39.99 euros per month.",
         source="plans-and-pricing.md"),

    dict(id=2,
         q="What are the pay-as-you-go roaming data charges outside the EU?",
         reference="Outside the EU/EEA, data is 10 euros per GB in Zone 1 and 15 euros per GB in Zone 2.",
         source="roaming.md"),

    dict(id=3,
         q="How do I cancel my postpaid contract?",
         reference="In the HexaMobile app go to Account > Manage subscription > Cancel my plan, choose a date, and confirm. It takes effect at the end of the billing cycle and there are no fees.",
         source="cancel-contract.md"),

    dict(id=4,
         q="Is there a fee for cancelling early?",
         reference="No. There are no early termination fees on any HexaMobile plan.",
         source="cancel-contract.md"),

    dict(id=5,
         q="How do I keep my phone number when switching to another operator?",
         reference="Do not cancel first. Request your RIO number by calling 3179 from your line, give it to your new operator, and they handle the port. It takes 1 to 3 business days.",
         source="cancel-contract.md"),

    dict(id=6,
         q="What happens if my monthly payment fails?",
         reference="The direct debit is retried after 5 days with a grace period. After two failed attempts outgoing service may be suspended until paid. There is no late fee.",
         source="billing-and-payments.md"),

    dict(id=7,
         q="How much is the data overage charge on capped plans?",
         reference="Overage is 2 euros per additional GB. Unlimited and prepaid plans never incur overage.",
         source="billing-and-payments.md"),

    dict(id=8,
         q="How do I activate my eSIM?",
         reference="In the app go to Account > Activate eSIM, then scan the QR code with the target device or tap Activate on this device.",
         source="sim-activation.md"),

    dict(id=9,
         q="My phone suddenly has no signal. What should I try first?",
         reference="Toggle Airplane mode on and off, restart the phone, and check the app for a network outage in your area.",
         source="troubleshooting-no-signal.md"),

    dict(id=10,
         q="What APN should I use for mobile data?",
         reference="The APN should be set to hexa.mobile.",
         source="troubleshooting-no-signal.md"),

    dict(id=11,
         q="What travel passes does HexaMobile offer for non-EU travel?",
         reference="World Pass 1 GB for 9.99 euros (valid 7 days, 30 minutes of calls) and World Pass 5 GB for 29.99 euros (valid 30 days, 120 minutes of calls).",
         source="roaming.md"),

    # ---- out-of-scope: the KB does not cover home internet ----
    dict(id=12,
         q="Does HexaMobile offer home fibre internet, and how much is it?",
         reference="The knowledge base does not cover home internet. The assistant should say it does not have that information and suggest contacting support, rather than inventing an answer.",
         source=None),
]
