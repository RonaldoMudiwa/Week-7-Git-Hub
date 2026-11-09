# Banking Complaint Analyzer

A Python program that reads complaints customers send to banks and works
out what each one is about, how upset and how urgent the customer is, and
what goes wrong most often.

Project 7 of a 12 week portfolio. Weeks 1 to 6 covered OOP, databases, web
APIs and FastAPI. This week is about working with text: natural language
processing (NLP) and a first machine learning model.

## Status

Day 5 of 7. Reading and cleaning complaints, word statistics, a mood
score, urgency rules, a keyword classifier and a machine learning one
compared fairly, the analyzer that ties them together, saving trained
models, and reports as text, CSV and a web page with charts.

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

## The machine learning classifier

```python
from complaints import KeywordClassifier, MLClassifier, compare, confusion_matrix

scores = compare({"keyword": KeywordClassifier(), "ml": MLClassifier()}, train, test)
scores["keyword"].accuracy     # 0.915
scores["ml"].accuracy          # 1.0

ml = MLClassifier().fit(train)
ml.predict("They charged me twice for one purchase")
# Prediction(label='Credit card', confidence=0.631)
ml.top_words("Mortgage", 3)    # [('mortgage payment', 1.737), ('mortgage', 1.737), ('payment', 1.424)]
```

It works in two steps, joined into a scikit-learn **pipeline**:

1. **TF-IDF** turns each complaint into numbers, one per word. A word
   scores high if it's used a lot in this complaint but is rare across all
   of them. "account" is everywhere, so it scores low. "escrow" is rare, so
   it scores high.
2. **Logistic regression** learns a weight for every word and product:
   how much seeing that word should push the guess towards that product.
   It's like the keyword classifier's cheat sheet, except it works out every
   weight itself, including negative ones ("this word means it's NOT a
   mortgage").

Settings, and why:

| Setting | Why |
| --- | --- |
| Single words and pairs | "late fee" means more than "late" and "fee" |
| Our own `TextCleaner` | The same cleaning as the rest of the program |
| `class_weight="balanced"` | Rare products aren't drowned out by common ones |
| `strength` (C=5) | Too high and it memorises the training complaints instead of learning, called overfitting |

`top_words` shows what it learned, so it isn't a black box you have to
trust blindly.

### Keyword vs machine learning

Both were trained on the same 241 complaints and marked on the same 59
they had never seen.

| | Accuracy | Mortgages right | Bank accounts right |
| --- | --- | --- | --- |
| Keyword | 91.5% | 6 of 9 | 6 of 8 |
| Machine learning | 100% | 9 of 9 | 8 of 8 |

**Be careful with that 100%.** The made up sample is built from a small
set of phrases, so test complaints look a lot like training ones. Real
complaints are messier, and the score will be lower. What this does show is
that the machine learning version beats the simple baseline on the same
test, which is the point of having a baseline.

### The confusion matrix

One accuracy number hides which mistakes are made. A **confusion matrix**
shows real products down the side and guesses across the top. Everything
on the diagonal is right. Here's the keyword classifier's:

```
real / guess  Checking o Credit car Credit rep Debt colle Money tran   Mortgage
Checking o             6          0          0          2          0          0
Credit car             0          9          0          0          0          0
Credit rep             0          0         12          0          0          0
Debt colle             0          0          0         12          0          0
Money tran             0          0          0          0          9          0
Mortgage               0          0          0          3          0          6
```

Its mistakes all go one way: 5 complaints wrongly put in "Debt collection".

- **Precision**: of everything it called debt collection, how much really
  was? 12 of 17, about 71%. Low precision means false alarms.
- **Recall**: of the real debt collection complaints, how many did it find?
  12 of 12, 100%. Low recall means it misses things.

## Putting it together

```python
from complaints import ComplaintAnalyzer, MLClassifier, Summary, TextReport, HtmlReport, CsvExport

analyzer = ComplaintAnalyzer(MLClassifier().fit(train))
results = analyzer.analyse_all(complaints)       # most urgent first

summary = Summary.of(results)
print(TextReport().render(summary))
HtmlReport().save(summary, "data/report.html")
CsvExport().save(results, "data/results.csv")    # opens in Excel
```

`ComplaintAnalyzer` runs the classifier and the urgency check over every
complaint and sorts them: most urgent first, and with the same priority,
oldest first, because it's been waiting longest. It's handed a classifier
rather than making one, so the keyword or the machine learning one can be
passed in without changing anything.

```
Complaints analysed: 300

By priority (5 = deal with it today)
  5  ###                                      13
  4  ########                                 33
  3  ########                                 34
  2  ############                             51
  1  ######################################## 169

Most urgent
  [5] 9000068 Money transfer: Someone hacked my account and sent a wire transfer of {$1...
        - Possible fraud: money may still be at risk (hacked)
        - Customer has already tried to get help (nobody will listen)
```

- **`TextReport`** for the terminal, **`HtmlReport`** for a web page with
  two bar charts drawn by matplotlib, both subclasses of `ReportBuilder`.
  `Summary` works out the numbers once and both just show them.
- The charts are saved inside the HTML page as text (base64), so the page
  is one file that opens offline and can be emailed.
- Complaint text is escaped in the web page, so a complaint with `<` in it
  can't break the page.
- **`CsvExport`** writes one row per complaint with its priority, mood,
  guessed product and reasons, ready for Excel.

## Saving a trained model

```python
from complaints import save_model, load_model

save_model(classifier, "data/model.pkl")
classifier = load_model("data/model.pkl")       # no need to train again
```

Like saving a game: training on a big file can take minutes, so you save
the result and carry on next time.

- It's written to a temporary file first and then swapped in, so a crash
  never leaves half a model.
- The file is labelled with a format name and version. Opening an old or
  unrelated file gives a clear error instead of a crash.
- **Only load model files you made yourself.** Python's pickle can run
  code hidden inside a file when it's opened, so a model downloaded from a
  stranger could do anything to your computer.

## Tests

```bash
pytest          # 138 tests
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
