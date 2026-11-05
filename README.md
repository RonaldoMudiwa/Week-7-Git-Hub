# Banking Complaint Analyzer

A Python program that reads complaints customers send to banks and works
out what each one is about, how upset and how urgent the customer is, and
what goes wrong most often.

Project 7 of a 12 week portfolio. Weeks 1 to 6 covered OOP, databases, web
APIs and FastAPI. This week is about working with text: natural language
processing (NLP) and a first machine learning model.

## Status

Day 1 of 7. The complaint model, reading complaint files, and cleaning text
into words.

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

## Tests

```bash
pytest          # 39 tests
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
