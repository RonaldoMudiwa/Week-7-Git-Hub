"""Makes data/sample_complaints.csv: 300 MADE UP bank complaints.

These are not real complaints. They're built from short phrases so the
tests and the demo have something to work with offline. They use the same
columns as the real CFPB Consumer Complaint Database, so the real file can
be dropped in later without changing any code.

Run it again and you get exactly the same file, because the random number
generator always starts from the same seed.

    python tools/make_sample_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 7
COUNT = 300
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_complaints.csv"

COLUMNS = ["Date received", "Product", "Issue", "Consumer complaint narrative",
           "Company", "State", "Complaint ID"]

COMPANIES = ["Northgate Bank", "Riverside Credit Union", "Summit Financial", "Harbour Savings",
             "Oakfield Lending", "Bluewater Card Services"]
STATES = ["CA", "TX", "NY", "FL", "IL", "GA", "OH", "WA", "NC", "AZ"]

# product -> issue -> phrases that complaints about it might use
PRODUCTS: dict[str, dict[str, list[str]]] = {
    "Credit card": {
        "Fees or interest": [
            "I was charged a late fee of {money} even though I paid on time",
            "the interest rate on my credit card went up without any warning",
            "they charged an annual fee that was never mentioned when I opened the card",
        ],
        "Problem with a purchase shown on your statement": [
            "there is a charge on my credit card statement for {money} that I never made",
            "I disputed a purchase from an online store but the card company refused to remove it",
            "a merchant charged my card twice and the card issuer will not refund the duplicate",
        ],
    },
    "Checking or savings account": {
        "Managing an account": [
            "my checking account was frozen and I cannot get to my own money",
            "the bank closed my savings account without telling me why",
            "I have been locked out of online banking for two weeks",
        ],
        "Problem caused by your funds being low": [
            "I was hit with three overdraft fees of {money} in one day",
            "the bank processed the largest payment first so I got extra overdraft charges",
            "a deposit was held for days and my account went negative",
        ],
    },
    "Mortgage": {
        "Trouble during payment process": [
            "my mortgage payment was not applied and now they say I am late",
            "the servicer increased my escrow payment by {money} with no explanation",
            "I sent my monthly mortgage payment but it was returned to me",
        ],
        "Applying for a mortgage or refinancing an existing mortgage": [
            "my refinance application has been delayed for months with no update",
            "the lender changed the closing costs at the last minute",
            "I was promised a rate lock on my home loan but it was not honoured",
        ],
    },
    "Debt collection": {
        "Attempts to collect debt not owed": [
            "a debt collector keeps calling me about a debt that is not mine",
            "they are trying to collect {money} for a loan I already paid off",
            "the collection agency says I owe money on an account I never opened",
        ],
        "Communication tactics": [
            "the collector calls my phone several times a day and threatens me",
            "a debt collector contacted my employer and my family about the debt",
            "they keep calling late at night even after I asked them to stop",
        ],
    },
    "Credit reporting": {
        "Incorrect information on your report": [
            "my credit report shows a late payment that never happened",
            "there is an account on my credit report that does not belong to me",
            "the credit bureau lists a closed account as still open",
        ],
        "Problem with a credit reporting company's investigation": [
            "I disputed an error on my credit report but the bureau did not investigate",
            "the credit reporting agency verified wrong information without checking",
            "they never sent me the results of my credit report dispute",
        ],
    },
    "Money transfer": {
        "Fraud or scam": [
            "I was tricked into sending {money} through a payment app to a scammer",
            "someone hacked my account and sent a wire transfer of {money} to a stranger",
            "my money transfer went to a fraud account and the company will not help",
        ],
        "Other transaction problem": [
            "my international money transfer never arrived",
            "the wire transfer was delayed for a week with no explanation",
            "I sent money to the wrong person by mistake and cannot get it back",
        ],
    },
}

# Shared bits that make the complaints less tidy, like real ones.
OPENINGS = ["", "On XX/XX/2026 ", "Last month ", "For the past few weeks ", "Recently ",
            "I am writing because "]
ENDINGS = [
    "", " I have called customer service many times and nobody helps.",
    " This is very stressful and I want this fixed.", " Please help me resolve this.",
    " I want a full refund.", " The bank was rude when I called.",
    " I have been a customer for years and this is unacceptable.",
    " My account number is XXXX and nobody will listen.",
]


def money(rng: random.Random) -> str:
    # The real database writes amounts like {$120.00}.
    return "{$" + f"{rng.randint(15, 2500)}.00" + "}"


def make_rows(count: int = COUNT, seed: int = SEED) -> list[list[str]]:
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    rows = []
    for number in range(count):
        product = rng.choice(list(PRODUCTS))
        issue = rng.choice(list(PRODUCTS[product]))
        phrase = rng.choice(PRODUCTS[product][issue]).format(money=money(rng))
        narrative = (rng.choice(OPENINGS) + phrase + "." + rng.choice(ENDINGS)).strip()
        narrative = narrative[0].upper() + narrative[1:]
        received = start + timedelta(days=rng.randint(0, 300))
        rows.append([received.isoformat(), product, issue, narrative, rng.choice(COMPANIES),
                     rng.choice(STATES), str(9_000_000 + number)])
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(make_rows())
    print(f"Wrote {COUNT} made up complaints to {OUT}")


if __name__ == "__main__":
    main()
