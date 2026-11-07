# Banking Complaint Analyzer

A Python program that reads complaints customers send to banks and works
out what each one is about, how upset and how urgent the customer is, and
what goes wrong most often.

Project 7 of a 12 week portfolio. Weeks 1 to 6 covered OOP, databases, web
APIs and FastAPI. This week is about working with text: natural language
processing (NLP) and a first machine learning model.

## Status

Day 3 of 7. The complaint model, reading complaint files, cleaning text
into words, word statistics, a simple keyword classifier with a fair way to
mark it, a mood score and urgency rules.

## The idea

A bank can get thousands of complaints a week. Nobody can read them all, so
this program reads them for you, like a post room sorting letters: what is
this about, how upset is the customer, does it need someone today?

## The data

The real data is the **CFPB Consumer Complaint Database**: millions of real
complaints about US banks, published by the US Consumer Financial
Protection Bureau. Each one already has a **product** (like "Mortgage") and
an **issue** (like "Trouble during payment process"), which is what lets us
check whether the program sorts complaints correctly. Private details are
already hidden as `XXXX`.

`data/sample_complaints.csv` is **made up**. It's 300 short complaints built
from phrases by `tools/make_sample_data.py`, so the tests and the demo work
offline. They're tidier and more alike than real complaints, so any score
measured on them will look better than it would on real data. The sample
uses exactly the same columns as the real file, so the real file can be
dropped in without changing any code.

```bash
python tools/make_sample_data.py      # makes the same 300 every time
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                           # run the tests
```

Needs Python 3.10 or newer.

## Reading complaints

```python
from complaints import CsvComplaintSource

source = CsvComplaintSource("data/sample_complaints.csv")
complaints = source.load()
complaints[0].product        # 'Mortgage'
source.report                # LoadReport(read=300, skipped_no_text=0, skipped_bad=0)
```

- **`Complaint`** is one complaint: its id, date, product, issue, the text
  the customer wrote (the narrative), the company and the US state. It's
  frozen and checks its own values.
- **`ComplaintSource`** is an abstract class. `CsvComplaintSource` reads
  CFPB style files, and `ListSource` holds complaints already in memory for
  tests. Another source, like a database, could be added later without
  changing anything else.
- **One at a time.** Sources hand out complaints one by one, like a
  conveyor belt, so even a 3 GB file never has to fit in memory at once.
  `load(limit=1000)` reads just the first thousand.
- **Nothing is skipped silently.** Most real complaints have no narrative,
  because customers can choose not to share it. Those rows, and rows with
  broken dates, are skipped and counted in `source.report`.
- Both date styles the CFPB uses are understood: `2026-10-28` and
  `10/28/26`. So is the invisible mark Excel adds to the start of files.

## Cleaning text

A computer can't read. Before it can count words, the text has to be
tidied, like sorting post by looking at the address and not the stamp.

```python
from complaints import TextCleaner, ngrams

cleaner = TextCleaner()
cleaner.tokens("On XX/XX/2026 I was charged a LATE fee of {$35.00}!")
# ['charged', 'late', 'fee', 'money']

ngrams(["late", "fee", "charged"], 2)
# ['late fee', 'fee charged']
```

| Step | Example |
| --- | --- |
| Hidden details removed | `XX/XX/2026`, `XXXX` -> gone |
| Amounts become a word | `{$35.00}` -> `money` |
| Lower case | `LATE` -> `late` |
| Punctuation removed | `fee!` -> `fee` |
| Stop words removed | `I was a of` -> gone |

- **Tokens** are the useful words left at the end.
- **Stop words** are words like "the", "and" and "was". They're in every
  complaint, so they tell us nothing about what it's about.
- **N-grams** are words that sit next to each other. "late fee" says much
  more than "late" and "fee" on their own.

## Word statistics

```python
from complaints import WordStats, counts_by, top_issues

stats = WordStats()
stats.top_words(complaints, 5)            # [('account', 101), ('money', 100), ...]
stats.top_phrases(complaints, 2, 3)       # [('past weeks', 57), ('last month', 48), ...]
counts_by(complaints)                     # [('Debt collection', 62), ('Credit reporting', 61), ...]
top_issues(complaints, "Mortgage")        # the most common mortgage problems
stats.distinctive_words(complaints, "Money transfer", 3)
# [('transfer', 180.74), ('wire', 82.16), ('fraud', 60.25)]
```

**Distinctive words** are the ones used a lot in one kind of complaint but
hardly anywhere else. "account" is the most common word overall, but it
turns up everywhere, so it's no help for telling complaints apart.
"escrow" only turns up in mortgage complaints, so it's a great clue.

The score is how many times more often a word is used inside a group than
outside it. 1 is added to every count first, so a word never seen outside
the group gets a big score instead of a divide by zero error. This trick
is called smoothing.

## The keyword classifier

A **classifier** is a program that sorts things into groups. This one
guesses a complaint's product.

```python
from complaints import KeywordClassifier, evaluate, train_test_split

train, test = train_test_split(complaints, test_share=0.2)
classifier = KeywordClassifier().fit(train)

classifier.predict("The debt collector keeps calling me at night")
# Prediction(label='Debt collection', confidence=1.0)

result = evaluate(classifier, test)
result.accuracy            # 0.915 on the made up sample
result.per_label           # {'Mortgage': (6, 9), ...} right, out of
```

- **How it works.** For each product it learns its 15 most distinctive
  words, like a post room worker with a cheat sheet. To sort a new
  complaint, it adds up the scores of the clue words it finds, and the
  product with the most wins. `confidence` is that product's share of the
  total score.
- **No clues at all?** It guesses the most common product, with a
  confidence of 0, so you can tell it was only a guess.
- **`Classifier` is abstract.** On Day 4 a machine learning classifier
  will plug into the same place, and `evaluate` will mark both the same
  way. Starting with something simple matters: if the clever version can't
  beat it, the clever version isn't worth having.

### Marking it fairly

You can't mark a student on the exact questions they revised from, or
they'd get full marks just by remembering. So `train_test_split` keeps
20% of the complaints back. The classifier learns from the other 80% and
is marked only on complaints it has never seen.

- **Stratified.** Each product is split separately, so a rare product
  can't end up only on one side by bad luck.
- **Seeded.** The shuffle is the same every run, so a change to the code
  can be compared fairly with the run before.
- On the made up sample it gets **91.5%** right. Real complaints are
  messier, so expect less on real data. Mortgages are its weakest product
  (6 out of 9), because the sample's mortgage complaints share fewer
  special words.

## How upset is the customer?

```python
from complaints import LexiconSentimentScorer

scorer = LexiconSentimentScorer()
scorer.score("This is a scam and totally unacceptable. The staff were rude.")
# Sentiment(score=-0.792, words=('scam', 'unacceptable', 'rude'))  mood: angry
scorer.score("I am not happy").mood       # unhappy
```

This is called **sentiment analysis**. It uses a **lexicon**: a word list
where each word has a score, from -3 ("unacceptable", "scam") to +3. The
scores of the words it finds are added up and squashed into -1 to +1.

| Score | Mood |
| --- | --- |
| above -0.1 | calm |
| -0.1 to -0.5 | unhappy |
| below -0.5 | angry |

- **"not" flips the next word.** "not happy" counts as negative. It looks
  up to three words back, so "not at all happy" works too.
- **"very" makes the next word stronger.** "very rude" scores lower than
  "rude".
- `SentimentScorer` is abstract, so a smarter scorer could replace this one
  later.

## How urgent is it?

```python
from complaints import UrgencyAssessor

urgency = UrgencyAssessor().assess(
    "Someone hacked my account and sent a wire transfer of {$500.00}. Unacceptable.")
urgency.priority     # 5
urgency.reasons      # ['Possible fraud: money may still be at risk (hacked)', 'Customer is angry']
```

Like A&E at a hospital: a broken arm is seen before a sore throat,
whatever order they came in. Each complaint gets a priority from 1 (can
wait) to 5 (deal with it today).

| Rule | Points | Looks for |
| --- | --- | --- |
| `FraudRule` | 3 | fraud, scam, hacked, stolen, identity theft |
| `LockedOutRule` | 2 | frozen, locked out, cannot get, cannot access |
| `HarassmentRule` | 2 | threatened, harassed, late at night |
| `RepeatContactRule` | 1 | many times, nobody helps, no update |
| Angry mood | 1 | a mood score below -0.5 |

Priority is 1 plus the points, capped at 5. Every priority comes with its
reasons, so a person can see why and disagree.

- Each rule is its own small class with one job. A new kind of emergency is
  a new class added to the list, and nothing else changes.
- `PhraseRule` does the work for most rules. They just give it their words
  and points. Phrases of up to three words are matched, so stop words are
  kept here, or "locked out" would become just "locked".
- On the made up sample: 169 complaints are priority 1 and 13 are priority
  5. Most complaints aren't emergencies, which is what you'd hope.

## Tests

```bash
pytest          # 97 tests
pytest --cov    # with a coverage report, fails below 95%
```

## Week plan

| Day | Focus |
| --- | --- |
| 1 | Complaint model, reading complaint files, cleaning text |
| 2 | Word statistics and a simple keyword classifier |
| 3 | Mood score and urgency rules |
| 4 | Machine learning classifier, compared with the keyword one |
| 5 | Saving the trained model, and reports with charts |
| 6 | Command line interface |
| 7 | Final tests and README |
