"""A more forgiving answer parser for the steering batteries.

`core.parse_answer` reads only the leading catalogue name and is deliberately
conservative. Several models rarely answer in that shape: Llama-3.3-70B restates
the question before answering, prefixes the answer with "the best answer is", or
emits a lettered multiple-choice list; Qwen3-32B usually opens with a literal
"?" (the prompts end in "is", so the model completes the question first) and then
restates it. The frozen parser scores all of that as "no constellation", which
understates those models badly while saying nothing about where the sky
representation actually put the answer.

What this adds, and nothing more:

  0. The text is folded to unmarked letters, so the diacritics the models
     actually write ("Boötes") match the catalogue's plain spelling, and known
     alternative names are rewritten to their catalogue form: spelling variants
     anywhere ("Pisces Austrinus", "Scorpio"), English names anywhere when the
     phrase can mean nothing else ("the Great Dog"), and English names that are
     also ordinary words only where the sentence is naming a constellation
     ("depicted as a centaur" is not an answer of Centaurus).
  1. Leading noise is removed - punctuation the prompt's own phrasing invites
     ("?", "!", quotes, bullets), then a restatement of the question ("the
     closest constellation to X constellation is"), then an answer preamble
     ("the best answer is"), cycling until nothing more is stripped.
  2. The question's own object is deleted wherever it occurs: "to X", "near X",
     "of the X constellation" for the SOURCE constellation only. The prompt asks
     for a neighbour of the source, so the source in that position is always the
     question being echoed, never an answer. A leading restatement of the
     question goes too - after the literal "?" the model wrote when it finished
     the question, or in the explicit "The constellation X is located..." shape
     without one - together with any apposition renaming the same constellation
     ("(the Whale)", ", also known as the Peacock"). When such a restatement is
     all there is, the completion names nothing but its own source, and the
     verdict is `ramble_A` rather than a silent "no constellation" or a credited
     answer of A. Other names are never touched.
  3. An enumerated list ("A. Orion B. Caelum C. Canis Major", "A: ...", bullets,
     numbers) is read as a SET of candidates. A marker opens a list only when a
     word follows it, so a decimal in running text ("1. 1 degrees away") is not
     an enumeration. A constellation named BEFORE the first marker is an answer
     the model has already given, and the list that follows only elaborates it,
     so the text up to the marker is read instead. A list truncated to one
     visible item counts as a single answer.
  4. Any first clause naming two or more constellations is likewise read as a
     set, whether separated by commas, "and", "or", "?" or nothing at all. Only
     a full stop ends that clause, because a run of questions ("Pisces? Cetus?
     Andromeda?") is the model offering alternatives, not answering.

A candidate set is scored by the MEAN angular distance of its members to an
endpoint, so a quiz whose options cluster around the destination counts as
within 30 degrees of it. A single-name answer keeps the original semantics.

`core.parse_answer` is unchanged and is still what matches a single leading
name; callers keep both scorings side by side so they stay comparable.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from core import angle, display, parse_answer

# Punctuation a completion may open with before any content. A bullet that opens
# the first item of a list is NOT noise: stripping it hid that item from the
# list rule below, which then read the enumeration as starting at item two.
# Markdown emphasis ("**Orion**") is still stripped, having no space after it.
NOISE = re.compile(r'^(?:[\s?!.,:;"\'‘’“”]|[*•–—-](?!\s))+')
# "the closest constellation to <X> is", "the neighboring constellations to <X> are:",
# and "the constellation bordering <X> is" (an extra steering template)
ECHO = re.compile(
    r'^\s*(?:the\s+)?(?:(?:closest|nearest|neighbou?ring|adjacent|bordering)\s+'
    r'constellations?\s+(?:to|of)|constellations?\s+bordering)\s+.{0,48}?\s+(?:is|are|include[sd]?)\s*[:,]?\s*', re.I)
# "the best answer is", "answer:", "the correct answer is"
PREAMBLE = re.compile(
    r'^\s*(?:the\s+)?(?:best\s+|correct\s+|right\s+)?answer\s*(?:is|:)\s*', re.I)
# "A. " / "A) " / "A: " / "1. " / "* " / "- " opening a list ITEM. A word must
# follow the marker, so running text carrying a decimal ("Taurus, which is 1. 1
# degrees away") is not mistaken for an enumeration.
MARKER = re.compile(r'(?:^|\s)(?:[A-Ha-h][.):]|\d{1,2}[.)]|[*•–-])\s+(?=[A-Za-z*"\'(])')

CATEGORIES = ['no_constellation', 'ramble_A', 'equals_A', 'equals_B',
              'within30_A_not_A', 'within30_B_not_B', 'other']

# Spelling variants of a catalogue name, read anywhere. "Pisces Austrinus" is a
# misspelling of Piscis Austrinus and has to be rewritten before the shorter
# "Pisces" can claim its first word.
VARIANTS = {
    'Piscis Austrinus': ['Pisces Austrinus'],
    'Scorpius': ['Scorpio'],
}

# English names of constellations. A multi-word phrase names the constellation
# and nothing else, so it is read anywhere, as "Southern Cross" always has been.
ENGLISH = {
    'Crux': ['Southern Cross'],
    'Piscis Austrinus': ['Southern Fish'],
    'Triangulum Australe': ['Southern Triangle'],
    'Corona Borealis': ['Northern Crown'],
    'Corona Australis': ['Southern Crown'],
    'Canis Major': ['Great Dog', 'Greater Dog'],
    'Canis Minor': ['Little Dog', 'Lesser Dog'],
    'Cetus': ['Whale'],
}

# English names that are also ordinary words: "Sagittarius is often depicted as
# a centaur" is not an answer of Centaurus, and "named after the microscope" is
# etymology, not a name. These are read only where the sentence is naming a
# constellation - after "constellation of", "known as", "called", an opening
# bracket or a list marker, optionally through "the".
ENGLISH_CUED = {
    'Centaurus': ['Centaur'],
    'Microscopium': ['Microscope'],
    'Telescopium': ['Telescope'],
    'Pavo': ['Peacock'],
    'Camelopardalis': ['Giraffe'],
    'Tucana': ['Toucan'],
    'Bootes': ['Herdsman', 'Ploughman'],
}
CUE = (r'(?:constellations?\s+(?:of|named|called)|known\s+as|called|nicknamed|'
       r'[A-Ha-h][.):]|\d{1,2}[.)]|\(|\*\*|[*•–-]\s)\s*(?:the\s+)?')


def fold(text):
    """Drop combining marks, so "Boötes" reads as the catalogue's "Bootes"."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))


def alternation(*mappings):
    """(surface form -> catalogue name, pattern), longest form first.

    Longest first so a name never loses to its own prefix. Mappings are merged
    by concatenation, since one catalogue name can appear in several of them.
    """
    forms = {f.casefold(): name for mapping in mappings
             for name, spellings in mapping.items() for f in spellings}
    return forms, re.compile(
        r'\b(' + '|'.join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r')\b', re.I)


ALIAS, ALIAS_RE = alternation(VARIANTS, ENGLISH)
CUED, _CUED_RE = alternation(ENGLISH_CUED)
CUED_RE = re.compile('(' + CUE + ')' + _CUED_RE.pattern, re.I)


def normalize(text):
    """Fold diacritics, then rewrite known alternative names to catalogue names."""
    out = ALIAS_RE.sub(lambda m: ALIAS[m.group(1).casefold()], fold(text))
    return CUED_RE.sub(lambda m: m.group(1) + CUED[m.group(2).casefold()], out)


def strip_lead(text, limit=8):
    """Remove leading punctuation, question restatements and answer preambles."""
    out = text
    for _ in range(limit):
        prev = out
        out = PREAMBLE.sub('', ECHO.sub('', NOISE.sub('', out), count=1), count=1)
        if out == prev:
            return out
    return out


def drop_question_restatement(text, source_name):
    """Delete a leading restatement of the question, and say whether there was one.

    The prompts end in "is", so a model often completes the question before
    answering it, and the literal "?" marks where it did: a source named
    straight after that "?" is the prompt again, not an answer. A completion
    that opens "The constellation <source> is located in the southern sky" is
    the same restatement without the question mark, and is taken as one only in
    that explicit "constellation X" / "X constellation" shape, so a bare leading
    name stays the answer it looks like.

    What follows the name in apposition - "(the Whale)", ", also known as the
    Peacock", ", the Herdsman" - is another word for the source, not a second
    constellation, so it goes with it. Only a continuation naming the source
    itself is absorbed; anything else is left for the parser to read.
    """
    alt = '|'.join(re.escape(f) for f in
                   sorted({source_name, display(source_name)}, key=len, reverse=True))
    named = rf'(?:constellation\s+(?:of\s+)?(?:the\s+)?(?:{alt})|(?:{alt})\s+constellation)'
    head = re.match(
        rf'\s*(?:\?\s*(?:the\s+)?(?:constellation\s+(?:of\s+)?(?:the\s+)?)?(?:{alt})'
        rf'(?:\s+constellation)?|(?:the\s+)?{named})\b', text, re.I)
    if head is None:
        return text, False
    rest = text[head.end():]
    while True:
        # An apposition restating the same constellation, once per turn.
        more = re.match(rf'\s*[,(\[–—-]\s*(?:(?:also\s+)?(?:known\s+as|called|nicknamed|named)\s+'
                        rf'|or\s+)?(?:the\s+)?(?:{alt})(?:\s+constellation)?\s*[)\],]?',
                        rest, re.I)
        if more is None:
            return rest, True
        rest = rest[more.end():]


def drop_question_object(text, source_name):
    """Delete "to/near/of (the) <source>" - the question's object, never an answer.

    "borders/bordering <source>" is the same object after the verb the extra steering
    template uses ("the constellation that borders the Antlia constellation is Bootes").
    """
    alt = '|'.join(re.escape(f) for f in
                   sorted({source_name, display(source_name)}, key=len, reverse=True))
    return re.sub(rf'\b(?:to|of|near|beside|around|from|borders?|bordering)\s+(?:the\s+)?(?:{alt})'
                  rf'(?:\s+constellation)?\b', ' ', text, flags=re.I)


def build_lookup(catalog):
    """(surface form -> id, compiled alternation), longest form first."""
    forms = {}
    for c in catalog:
        if c['type'] != 'constellation':
            continue
        for f in {c['name'], display(c['name'])}:
            forms[fold(f).casefold()] = c['id']
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r')\b', re.I)
    return forms, pattern


def mentioned(text, forms, pattern):
    """Catalogue constellations named in `text`, in order, without repeats."""
    seen = []
    for m in pattern.finditer(text):
        i = forms[m.group(1).casefold()]
        if i not in seen:
            seen.append(i)
    return seen


def candidates(text, source, catalog, forms, pattern):
    """Return (ids, kind) with kind in {'options', 'single', 'ramble', 'none'}."""
    name = catalog[source]['name']
    body, restated = drop_question_restatement(normalize(text), name)
    ids, kind = read(drop_question_object(strip_lead(body), name), catalog, forms, pattern)
    # A completion that restates the question and then describes the source has
    # named nothing else: it is rambling about A rather than answering with it.
    if restated and ids in ([], [source]):
        return ids, 'ramble'
    return ids, kind


def read(body, catalog, forms, pattern):
    """Read candidates from a completion already stripped of the question."""
    # 1. An explicit list. A constellation named before the first marker is an
    # answer already given ("Taurus, which is 1. 1 degrees away. Pegasus is the
    # second closest"), so what follows the marker is dropped rather than
    # allowed to displace it; otherwise the list itself is read.
    marks = list(MARKER.finditer(' ' + body))
    if marks:
        cut = max(marks[0].start() - 1, 0)
        if mentioned(body[:cut], forms, pattern):
            body = body[:cut]
        elif len(marks) == 1:
            # A lone marker is a Q/A label opening the answer ("A: The constellation
            # that borders ... is Bootes. Antlia is small..."), not a list: read what
            # follows it by the clause rules below, so later sentences stay out.
            body = body[max(marks[0].end() - 1, 0):]
        else:
            ids = mentioned(body[cut:], forms, pattern)
            # A 24-token completion often shows only the first item of its list;
            # that single item is the answer, not an inconclusive set.
            if ids:
                return ids, 'options' if len(ids) >= 2 else 'single'

    # 2. Two or more names in the first clause: a set, however punctuated.
    # Only a full stop ends the clause. Models often offer alternatives as a run
    # of questions ("Pisces? Cetus? Andromeda?"), which is a list, not an answer,
    # so "?" must not terminate it.
    first = re.split(r'(?<=\.)\s', body, maxsplit=1)[0]
    series = mentioned(first, forms, pattern)
    if len(series) >= 2:
        return series, 'options'

    # 3. A single leading name, by the frozen parser on the cleaned body.
    i = parse_answer(body, catalog)
    if i is not None and catalog[i]['type'] == 'constellation':
        return [i], 'single'

    # 4. Otherwise the first name in the first sentence, if any.
    if series:
        return series[:1], 'single'
    return [], 'none'


def classify(text, source, target, catalog, forms, pattern, threshold=30.0):
    """Five-way category plus diagnostics, under the forgiving rules."""
    ids, kind = candidates(text, source, catalog, forms, pattern)
    out = dict(rich_kind=kind, rich_n=len(ids),
               rich_answer='|'.join(display(catalog[i]['name']) for i in ids),
               rich_deg_from_A=None, rich_deg_from_B=None)
    if not ids:
        out['rich_category'] = 'ramble_A' if kind == 'ramble' else 'no_constellation'
        return out
    da = float(np.mean([angle(catalog[i]['unit'], catalog[source]['unit']) for i in ids]))
    db = float(np.mean([angle(catalog[i]['unit'], catalog[target]['unit']) for i in ids]))
    out['rich_deg_from_A'] = round(da, 4)
    out['rich_deg_from_B'] = round(db, 4)
    if kind == 'ramble':
        out['rich_category'] = 'ramble_A'
    elif ids == [source]:
        out['rich_category'] = 'equals_A'
    elif ids == [target]:
        out['rich_category'] = 'equals_B'
    elif db <= threshold and db <= da:
        out['rich_category'] = 'within30_B_not_B'
    elif da <= threshold:
        out['rich_category'] = 'within30_A_not_A'
    else:
        out['rich_category'] = 'other'
    return out
