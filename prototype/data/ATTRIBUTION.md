# Corpus provenance and licensing

`documents.csv` holds 695 agricultural advisory documents. This file records
where they came from and under what terms they are redistributed, because the
corpus is **not** a single uniform body of material — it is two kinds of document
with different provenance, and the difference matters when reading a result.

The code in this repository is MIT licensed ([../LICENSE](../LICENSE)). That
licence does not cover this data.

## 1. Where the corpus came from

The corpus was provided by the organisers of the *Agricultural Extension RAG:
Smart Retrieval for Farmers* competition (Cohort 10, Tri-AI; access by
invitation — `cohort-10@tri-ai.org`). It was not scraped or assembled by this
team.

The organisers' stated licence for the competition dataset is **CC BY-NC-SA 4.0**
(attribution, non-commercial, share-alike). Separately, the corpus carries a
per-document `license` column recording upstream terms for individual documents.
Read together, the stricter of the two applies.

This demo is **non-commercial**, carries no advertising or sponsorship, and
attributes its sources below — which satisfies both readings. **Anyone intending
to use this corpus commercially should confirm terms with the organisers rather
than relying on this note.**

## 2. The two kinds of document

The `origin` column distinguishes them.

### Synthetic documents — 637 of 695

| | |
|---|---|
| `origin` | `synthetic` |
| `license` | `synthetic (CC0)` |
| Source URL | none |

Written for the competition to give the retrieval task reliable coverage across
crops and countries, and released under CC0.

**Important:** these rows still carry a `source` value — CGIAR, FAO, Plantwise,
ICRISAT, IITA, AGRA or "National Extension Service". That value names the
organisation whose published material the text was **modelled on**. It does not
mean the organisation published, reviewed or endorsed that document. The app
labels these cards "Synthetic document" for exactly this reason.

Distribution of synthetic documents by the `source` they were modelled on:

| Source label | Documents |
|---|---|
| FAO | 98 |
| Plantwise | 97 |
| ICRISAT | 96 |
| CGIAR | 94 |
| National Extension Service | 94 |
| AGRA | 80 |
| IITA | 78 |

### Source-grounded documents — 58 of 695

| | |
|---|---|
| `origin` | `llm_grounded` |
| `license` | CC-BY-4.0 (55), CC-BY-SA-4.0 (1), CC0-1.0 (1), CC-BY-3.0 (1) |
| Source URL | present on all 58 |

Grounded in published CGIAR material and resolvable through CGIAR's Handle
service. These are the documents whose claims can be followed back to an
original, and they are the only ones the app renders a link for.

All 58 resolve to `hdl.handle.net` persistent identifiers. Three carry
non-default terms and are listed individually below so an attribution notice can
name them.

## 3. Individual attributions

### CC-BY-SA-4.0 — share-alike applies

Document `650` — *Effective Management of Fall Armyworm in Lebanese Maize Fields*
(CGIAR) — <https://hdl.handle.net/10568/163102>

This is the only share-alike document in the corpus. Adaptations of **this
document** must be released under CC BY-SA 4.0. It sits among 694 documents
under other terms, so treat it individually rather than as representative.

### CC-BY-3.0

Document `694` — *Integrated Soil Fertility Management for Smallholder
Agriculture* (CGIAR)

### CC0-1.0

Document `652` — *Integrated Pest Management Strategies for Cowpea* (CGIAR)

### CC-BY-4.0 — the remaining 55

Document IDs `21` to `695`, all CGIAR, each with its `source_url` recorded in the
CSV. Attribution for these is satisfied by naming CGIAR and linking to the
recorded identifier.

All 58 grounded rows fall in ID range `21`–`695`; the synthetic rows occupy
`1`–`685`, so the two ranges overlap. Select on the `origin` column, not on ID.

## 4. Coverage

- **Crops (13):** Maize, Tomato, Rice, Cassava, Common bean, Cowpea, Groundnut,
  Sorghum, Plantain, Yam, Cocoa, Pearl millet, and a general category.
- **Countries (21):** Nigeria (133), Ghana (103), Tanzania (92), Kenya (75),
  Mali (53), Uganda (33), Ethiopia (27), Senegal (24), Zambia (24), DR Congo
  (20), Egypt (17), Côte d'Ivoire (16), Madagascar (14), Cameroon (12), Sudan
  (12), Burkina Faso (10), Malawi (9), Niger (8), Rwanda (6), Benin (4), Togo (3).

Coverage is broader than the Nigeria-focused framing used when this project
began, and any public description should reflect the actual spread above.

## 5. How the corpus is used

Documents are flattened into a single searchable string before indexing:

```
Crop {crop} | Country {country} | Title {title} | Text {text} | Source {source}
```

This template is identical in the training and inference notebooks, which is why
it must stay byte-for-byte stable — changing it silently invalidates the
fine-tuned models' learned representation.

The corpus is redistributed here **unmodified**, as received from the
competition, so that the demo is reproducible. The app retrieves documents; it
does not verify, summarise or endorse their contents.

## 6. Questions

Provenance questions about the corpus itself should go to the competition
organisers rather than to this team — we redistributed it, we did not compile it.
