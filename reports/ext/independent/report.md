# Independent run of protocol steps 2 and 3 (Milton vs. classics, book-level source)

Everything was done only from PROTOCOL_steps_2_3.md and the files in data/. Scripts: work/common.py, work/step2.py, work/step3.py. Outputs: results_step2.tsv, results_step3.tsv.

## Step 2. Check of the gold standard

### Decisions I had to make myself
1. **Paragraph numbering.** The target file has odd paragraph ids 1,3,5,... per book (this is stated as normal). The gold place `PL.<book>.p<NNN>` is the rank of the paragraph in the book. I verified that in all 12 books the ids are exactly 1,3,5,...,2n-1, so rank = (id+1)/2. The protocol text does not say how to match these, and nothing says what the odd ids mean. I took the "place = the paragraph containing line poem_line" rule from the task.
2. **"The cited fragment lies in the paragraph"** (item 1 of the protocol). There is no quote of the poem in the gold except the note's lemma (the part of the note before the first full stop). I tested that the lemma stems occur (a) in the poem_line itself, (b) in the paragraph. Threshold: at least 50% of the lemma stems present. The protocol does not define what "the fragment" is, so this operationalization is mine. I also did not check line numbers against the target file beyond existence of (book, line).
3. **Source reference check** (items 2 and 3). I ran these checks:
   - the unit exists in the source file;
   - the author and book number in ref_text equal the unit;
   - the ref_text string is found in the full text of the note (independent route, as the protocol suggests);
   - a range is not reversed;
   - the line number is not beyond the book, using canonical line counts of the original poems from my own memory (the source is a translation: Dryden, Pope, Riley prose, so the translation itself cannot confirm verse numbers; only the book level can be checked in the source texts);
   - the note id's line number agrees with poem_line;
   - duplicates by (place, unit);
   - several records from the same note;
   - citations of a Homer/Virgil/Ovid unit in a note that are absent from the gold.
   Weak "name check": proper names from the lemma (a fixed list I wrote by hand: Orion, Oechalia, Oeta, Medusa, Raphael, Maia, Laertes, Chimera, Pythian, Olympian, Hesperian, Cerberian, Hyacinthin) are looked up in the cleaned source unit. If absent there, I note whether they are in other units of the same work. This is noisy: the source is a translation, so Zeus/Jove, Orion dropped by Dryden and spelling variants (Oechalia, Chimera/Chimaera) give hits that are not errors. I label them `weak_name_*`.
4. **Source text defects found (not in the protocol).** Iliad.24 (150k characters, 3 times the others) contains Pope's postscript, a long essay and 300+ footnotes. It contains 44 mentions of Milton/Paradise Lost, including quotations. Metamorphoses units include Riley's footnotes. Iliad units have [n] footnote markers. For step 3, I removed [n] markers and cut Iliad.24 at "END OF THE ILIAD". Footnote text in the other units remains (the Iliad footnotes have no Milton hits, I checked). If this contamination is not cleaned, any lexical feature gets a spurious Milton overlap for Iliad.24. No gold record points to Iliad.24, so the effect on the ceiling is nil, but it matters for ranking of candidates.
5. **Severity tiers.** Flags are put in the TSV as they are; I treat as informational: `shared_note_multi_ref`, `dup_place_unit`, `book_level_ref`, `weak_name_*`. Hard problems: `note_id_line_mismatch`, `ref_range_reversed`, `ref_line_doubtful_external`.

### Findings (98 records)
- Place check: 98/98 places correspond to the paragraph that contains poem_line. 0 mismatches. No line is missing in the target.
- Lemma check: 96 records tested (R085 has the generic lemma "Lines 1037-1045" and R077 the one-word lemma "Go"; both skipped). All 96 pass at the 50% threshold both for the paragraph and for the poem line (lowest: R065, 67%, spelling difference "balanc't" vs "ballanc't").
- Source unit exists: 98/98. Author and book of ref_text agree with unit: 98/98. ref_text found in full note text: 98/98 (after normalizing spaces; my first version had a bug with the word "book", fixed).
- Records "confirmed in attachment and reference" (no hard flag): 93/98 = 95%. Hard flags (5 records):
  - R085: note id says line 1037, poem_line is 1045. The note covers lines 1037-1045, so the line is the end of the range; same paragraph (checked).
  - R086: note id says line 533, poem_line is 514. The lemma "monstrous Serpent" is in line 514, so only the note id is mislabelled.
  - R096: range "1.264-226" reversed (typo, probably 264-266 or 264-326); also the line number is doubtful.
  - R011: "Metamorphoses 10. 318". The Phlegraean plains in the Orpheus song of book 10 are about line 151 (the cleaned source text of Metamorphoses.10 does mention Phlegraean plains; line 318 is the Myrrha story). The unit is right, the line number is doubtful.
  - R066: "Metamorphoses 1. 150": the line about "prolific humour" matches the spontaneous generation passage at about 1.416-437, not 150 (giants). The unit is right.
  R011, R066 and R096 are from my own knowledge of the poems (external; flag `ref_line_doubtful_external`). At the unit level these do not change the evaluation.
- Informational counts: shared_note_multi_ref 20 records (several records from one note); dup_place_unit 9 records (same paragraph and same unit from different notes or the same note: R021/R022, R025/R026/R027, R029/R031, R067/R068); book_level_ref 17 records (the reference is only a book, no verse); weak_name_* 15 records (9 absent in this unit but present in others, 6 absent in the whole work, mostly translation effects; the most notable: Orion absent in Aeneid.1 Dryden, so R009 cannot be supported by any word). 46 records have no flag at all ("ok").
- "No record unreachable by construction": every record has a valid place and a valid unit, so none is unreachable at the book level. At the verse level I cannot say anything because the source has no verse numbering (translation prose/couplets). Edition differences are another issue: Riley's prose has book boundaries equal to Ovid's, Pope's Iliad and Odyssey and Dryden's Aeneid have the same book split as the originals. No numbering mismatch found at book level.
- Notes in notes_full_text.tsv (87 notes) vs gold: I checked the notes cited units not in the gold (`note_cites_unit_not_in_gold`): zero flags, meaning each note's Homer/Virgil/Ovid citations are all in the gold. The note file has more notes (87) than distinct notes in gold; I did not test the others.

### Manual labels (40 sampled records, full note text read)
Result: **source 26 (65%), background 14 (35%), not_by_text 0 (0%)**.
- source: R001 R012 R014 R017 R018 R023 R029 R032 R033 R038 R041 R042 R045 R049 R062 R065 R067 R068 R075 R077 R080 R084 R085 R089 R090 R094
- background: R003 R005 R006 R008 R009 R021 R024 R034 R040 R056 R070 R079 R091 R093
- not_by_text: none. I looked for notes whose cited place contradicts the text; R011/R066 (not in sample) are the only ones with doubtful numbers. I additionally spot-checked words in the cleaned source units (Charybdis in Odyssey.12, Thamyris in Iliad.2, Hyacinth in Metamorphoses.10, hyacinth in Odyssey.6, chain/scales in Iliad.8, Alcinous in Odyssey.7, Python/Pyrrha/Argus in Metamorphoses.1, Cadmus in Metamorphoses.4, etc.). All found. Not found: Orion in Aeneid.1 and the name Oechalia in Metamorphoses.9 (translation drops them, but Hercules/Nessus/Lichas are there).
- Rules I applied (the three labels are not sharply defined; this is where I made judgment calls):
  - background: contrast ("Unlike Hermes ...", Achilles's opposite attitude, "Th' ascent is easie" type), "see also"/"for example" lists, a common simile or formula described as tradition (autumnal leaves, rosy dawn, eye-cleansing), glosses of a name or creature (harpies, Charybdis, Daphne of Syria, Ceres), generic genre remark (reported narrative).
  - source: the note says Milton echoes / alludes / recalls / compares with this passage, or narrates the same story or scene that the line evokes (Hephaistos's fall, Orpheus's death, Deucalion and Pyrrha, Argus, Zeus and Hera, golden chain, Hesperian fables).
- **Doubtful cases** (labels depend on the interpretation): R003 (the "see also" next to the Iliad invocation), R006 (Aeneas's shield is a comparison without "echo"), R008 vs R014 (leaves = tradition = background, but bees = specific passages = source; this is a thin distinction), R023 (a critic's analogy, labelled source), R032 (bare "See Ovid 1.5-20", labelled source because the topic is clear), R062, R075, R080, R090 (name/episode gloss treated as source because the line evokes the episode, but R024, R040, R079 treated as background; the boundary "reference to a figure" vs "reference to an episode" is not given in the protocol), R094 (parallel scene "also behold", labelled source).
- Remark: the label scheme conflates two things: (a) is the cited passage what the note says the line refers to (source vs not_by_text), and (b) is the reference an echo or a context. Contrast items ("Unlike ...") are literally a reference to the passage but are not a source in the usual sense; I used background for them. I suggest splitting `background` into `contrast` and `tradition_or_gloss`.

### Where the protocol text was insufficient (step 2)
- "Цитируемый фрагмент": the gold file here has no explicit quote; I had to use the note lemma.
- Item 3 (numbering of source) is not checkable when the source is a translation at book level; the protocol does not say what to do (I only validated book-level).
- Item 4 (corrections as records with `evidence`): I made no corrections, only flags; I did not create manual_corrections.tsv.
- The criterion "доля записей с подтверждённой привязкой и ссылкой": the criterion does not define "confirmed", the lemma threshold, or whether informational flags count. I report 98/98 for alignment+unit+ref in note, and 93/98 without hard flags.
- The protocol says nothing about inspecting the source corpus for contamination (Iliad.24 footnotes with Milton quotations).

## Step 3. Ceiling and denominator

### Decisions made myself
- Text units: the paragraph (all lines of the gold paragraph) vs. the whole cleaned source book (75 units).
- Normalization: lowercase, u/v and i/j merged, a crude suffix-stripping stemmer (no lemmatizer available offline), own stop-word list (about 190 words incl. thou/thee/hath). Content token = not stop-word, length >= 3.
- **Rare lemma** = stem of length >= 4, found in 1..5 of the 75 source units (about 7% of them), and found in at most 15 of the 368 Paradise Lost paragraphs (so frequent poem words like "heaven" are excluded even if rare in the classics). It must occur in the gold paragraph and in the gold source unit.
- **Two words in a row** = two adjacent tokens in the paragraph (line breaks ignored) that are both content stems (non-stop-words) and also occur adjacent in the source unit (stemmed, stop-words break the pair).
- **has_bridge (literal, as the protocol says)** = paragraph and correct unit share a rare lemma OR a two-word sequence.
- Because "paragraph-wide" evidence looked weak, I also computed an **anchored** variant: the shared rare lemma / both words of the bigram must also occur in the note text or in the poem line. This is my addition, not in the protocol.
- Features for the ceiling (the correct unit gets a nonzero score, no thresholds, no training): (1) number of shared content stems, (2) number of shared rare stems, (3) number of shared two-word sequences, (4) idf-weighted cosine (bag of words, weights by log(75/df)).

### Ceiling (nonzero score for the correct unit among 75)
| feature | nonzero score | correct unit rank 1 | in top 5 | in top 15 | median rank |
|---|---|---|---|---|---|
| shared content stems | 98/98 (100%) | 11 | 25 | 47 | 17.5 |
| idf cosine | 98/98 (100%) | 8 | 27 | 51 | 13.0 |
| shared rare stems | 61/98 (62%) | 9 | 27 | 55 | 12.5 |
| shared two-word sequences | 41/98 (42%) | 13 | 35 | 59 | 11.0 |

The nonzero criterion is vacuous for dense features: shared content stems and cosine are 100% for any pair of English texts of this size. I added the rank statistics so that the numbers mean something. By rank the ceiling is much lower (rank 1 in 8-13 of 98).

### Denominator (literal bridge, thresholds as above)
- **With a bridge: 75 of 98 (77%); without: 23 of 98 (23%).** Types: rare lemma only 34, bigram only 14, both 27, none 23. Rare lemma in 61, bigram in 41.
- Sensitivity: rare-lemma thresholds (df_src<=3, df_PL<=10) give 63 with bridge; (10, 30) give 90 with bridge. So the denominator moves by 27 records (28 points) depending on a threshold the protocol does not set.
- **The literal bridge is weakly informative.** For a wrong source unit the same paragraph has a rare-lemma bridge in 43% of cases, a bigram in 32%, either in 57% (against 77% for the correct unit). Many shared rare lemmas are accidental (brittl, bush, antagonist, assyrian, pearl).
- Anchored variant: 20 of 98 with a bridge (78 without). Examples are the names: Thamyris, Alcinous, Pyrrha, Pomona, Python, Hyacinthin, Enna, Daphne, golden scales, hid noth.
- The 23 records with no literal bridge: R004 R006 R008 R009 R016 R024 R044 R048 R049 R050 R051 R052 R054 R056 R057 R060 R069 R070 R084 R085 R091 R092 R094. Reasons by reading: formula epithets in Homer (rosy dawn: R050 R057 R091); a word of Milton rendered by the translator with a different word (Orion absent in Dryden: R009; Adamantine: R004; ambrosia: R016); an echo of a plot or a scene (Narcissus R044, Philomela R051, Hera and Zeus R085, Hector/Achilles scales R048/R049, Aeneas's vision R094); a contrast with a figure (Raphael/Hermes R054 R069 R070); a book or genre reference (R056 R060); a single-word verbal echo (R084 certus, R092/R093 eye-cleansing); an archetype (R052 morning star).

### Where the protocol text was insufficient or ambiguous (step 3)
1. "Общая редкая лемма" has no definition (rare in what: the source, the target, both? df cut-off?), and the denominator depends strongly on it (63 to 90).
2. "Два слова подряд": adjacent after or before stop-words? In the original text or after removal of stop-words? Are lemmas or exact forms required? For translations of a different language this almost never matches except for names; I took stems.
3. "Общая" between what? The whole paragraph vs the whole book is nearly always satisfied by chance (43% for wrong units). The protocol should say whether the shared word should be tied to the cited fragment (my anchored variant: 20 records) or to the note.
4. "Ненулевой балл" is vacuous for dense features (100% for any bag of words), so the ceiling needs a rank-based definition (top-k).
5. The rule does not say how to treat records whose evidence is only a book reference or a contrast ("not a source"): do they stay in the denominator for step 3 (they have a bridge by accident) or should step 2 labels (background) be used to remove them? In this run 14 of 40 sampled records are background; no bridge test is conditional on the label.
6. Several records share the same paragraph and unit (dup_place_unit, 9); it is not said whether the denominator counts records or unique (place, unit) pairs.
7. With a translation as the source, the bridge test measures the translator's vocabulary, not the poet's. Pope and Dryden in couplets replace names (Jove, Cytherea). The protocol does not discuss this.

## Time and effort
About 30 tool calls in total: reading the data (~5), the alignment and reference checks (~6, including three small fixes), manual reading of all 98 notes (about 35k characters of text, one pass, labels for 40), the step 3 scripts (~5, with 3 threshold runs). Roughly 1.5 to 2 hours of equivalent human work; the longest items were the manual labelling and thinking about the definitions of the labels and the bridge. Without the author I could finish both steps; the protocol text alone was enough to know what to do, but not enough to make numbers reproducible (thresholds, definition of "fragment", of "rare lemma", of "two words in a row").
