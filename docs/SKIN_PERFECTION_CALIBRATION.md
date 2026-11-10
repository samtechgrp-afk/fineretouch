# Calibrating Skin Perfection on real photographs (Windows)

This guide is for the person who calibrates the engine on their own photos: preparing the
photos, running the quality gallery, reviewing what the engine found and missed, adjusting
thresholds, recording the results, and then validating independently.

**What this does and does not prove.** The synthetic benchmark (`docs/SKIN_PERFECTION_QA.md`)
uses generated skin. It shows whether a change helps or hurts. It does **not** show how the
engine performs on real photographs. Only the real-photo **test** split, scored once after the
thresholds are frozen, can support a claim about real-world accuracy. No real-photo results
exist yet: the photos used so far during development were already retouched, so they only
show false alarms, not missed imperfections.

## 0. One-time setup (PowerShell)

You need Python 3.12 and a clone of the repository (for example `C:\dev\fineretouch`).

```powershell
cd C:\dev\fineretouch
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r services\ai\requirements-dev.txt
cd services\ai
python -m scripts.perfection_benchmark --quick
```

The last command runs the 4-scene synthetic check (about 25 s on a fast PC, longer on the
Ryzen 3 7330U). If it prints a summary line and no `REGRESSION`, the setup works.

If PowerShell blocks `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use CMD with
`.venv\Scripts\activate.bat`.

Open a new PowerShell window later? Run `cd C:\dev\fineretouch` and
`.\.venv\Scripts\Activate.ps1` again, then `cd services\ai`.

## 1. Prepare the photos

1. **Use unretouched originals.** Use camera JPEGs or RAW files (CR3, NEF, ARW, DNG) that
   have not been retouched. Retouched exports have had their blemishes removed already, so
   they cannot show missed imperfections.
2. **Consent.** Use only photos the subject has cleared for internal QA. Write
   `"consent": "internal-qa"` in the manifest. The loader refuses any other value. The folder
   stays on your PC and is never committed to the repository.
3. **Folder layout.** Create one sub-folder per person (subject):

   ```
   D:\qa-photos\
     manifest.json
     S01\IMG_0012.CR3
     S02\IMG_0040.jpg
   ```

4. **Manifest.** Write `D:\qa-photos\manifest.json`, for example in Notepad. Add one entry
   per photo:

   ```json
   {"photos": [
     {"file": "S01/IMG_0012.CR3", "subject": "S01", "tone": "monk-8", "lighting": "hard",
      "region": "face", "conditions": ["whitehead", "post-acne"], "consent": "internal-qa"},
     {"file": "S02/IMG_0040.jpg", "subject": "S02", "tone": "monk-3", "lighting": "soft",
      "region": "face+body", "conditions": ["dense-acne", "body-blemish"], "consent": "internal-qa"}
   ]}
   ```

   - `tone` is `monk-1` … `monk-10`.
   - `lighting` is `soft`, `hard` or `mixed`.
   - `conditions` come from this list: `dense-acne`, `whitehead`, `blackhead`,
     `hyperpigmentation`, `post-acne`, `freckles-moles`, `stubble`, `body-blemish`.
   - Use forward slashes in `file`.

5. **Splits.** Each subject is assigned to *validation* (about 40%, used for calibration) or
   *test* (about 60%, scored once at the end) from a hash of the subject id. All of one
   person's photos land in the same split. To choose the split yourself, add
   `"split": "validation"` or `"split": "test"` to every photo of that subject.

6. **Check coverage.** See what is still missing:

   ```powershell
   python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --coverage
   ```

   About 39 validation and 64 test photos fill every cell, from at least 30 subjects, half of
   them Monk 7–10. With 10–20 photos you can start calibrating, but that is not enough for a
   release claim.

## 2. Label the photos in the app

Do this for every photo, in the FineRetouch editor:

1. Open the photo, then open **Test labels (QA)** and turn on **Label mode**.
2. Click every imperfection that should be removed: whiteheads, pimples, acne, blackheads,
   dark spots, post-acne marks and blemishes. Also label natural features that must stay:
   freckles, moles, birthmarks and follicles. Pick the kind and size before each click.
3. **Suggest from engine** (optional) adds the engine's marks as unconfirmed suggestions.
   Confirm only the ones that are right. Unconfirmed suggestions are ignored by the benchmark.
4. Enter the annotator and reviewer names, then click **Save labels**. This writes
   `<photo>.labels.json` next to the photo (for example `IMG_0012.CR3.labels.json`). The photo
   itself is never changed.

**Label every imperfection you can see at 100% zoom.** A mark you skip counts as a false
alarm when the engine finds it, and a missed imperfection cannot be measured unless it is
labelled.

## 3. Run the benchmark and the quality gallery (validation split)

```powershell
cd C:\dev\fineretouch\services\ai
python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --split validation `
  --html D:\qa-results\validation-report.html --json D:\qa-results\validation-report.json `
  --quality-gallery D:\qa-results\quality-review.html --label "before calibration"
```

Create `D:\qa-results` first (`mkdir D:\qa-results`). On the Ryzen 3 7330U, each 30 MP photo
takes about 40 s.

The command writes three files:

- `validation-report.html` / `.json`: precision, recall, removal rate, false positives,
  natural features damaged and texture retention. Each is given per photo, per tone group and
  per category.
- `quality-review.html`: every repair that failed a quality check, plus four sample repairs
  per photo that passed. Each row shows before | after, with each check's value against its
  limit.

## 4. Review detected and missed imperfections

**Detected (false alarms and wrong repairs).** Open `quality-review.html` in a browser, then:

- **Red cells:** a check failed, so the engine restored the original skin there. If the repair
  looks natural, tick **false alarm**.
- **Rows with no red cell:** the repair passed. If it does not look natural (a smudge, a flat
  patch, a repeated pattern, a colour shift or a visible edge), tick **looks wrong**.
- In `validation-report.html`, the **false positives** column per photo shows marks the engine
  flagged where you placed no label. Open those photos in the app with the marks overlay on
  and check what was flagged.

**Missed imperfections.** The reports give counts only:

- **recall:** the share of labelled imperfections the engine found.
- **removal rate:** the share it actually removed.
- **residual rate:** the share still visible afterwards.

They do not list the individual missed marks. To find them:

1. Sort the per-photo table by recall or removal rate.
2. Open the lowest photos in the app with Skin Perfection on, the marks overlay on and your
   labels visible.
3. A label with no mark beside it is a miss.

**Hands, feet and nails.** Check these specifically in full-body photos. Knuckle creases,
nail edges and toe joints can be flagged as body pimples (see the review report).

## 5. Record the results

**The tick boxes in `quality-review.html` are not saved.** Reloading or closing the page loses
them. Until the gallery can export them, record your calls in a CSV file next to the reports,
`D:\qa-results\calibration-notes.csv`:

```csv
photo,mark_id,kind,outcome,failed_checks,your_call,note
S01/IMG_0012.CR3,1834,pimple,original restored,colour,false alarm,repair looked fine at 100%
S02/IMG_0040.jpg,77,blemish,passed,,looks wrong,flat patch on cheek
```

Copy `photo`, `mark_id`, `kind`, the outcome and the failed checks from the gallery row.
Then keep:

- The report JSON from every run. Use a new `--label` and file name each time, for example
  `validation-report-v2.json`.
- The CSV file.
- A short note of which thresholds changed, from what to what, and why.

Only these numbers and notes may be shared. Never share the photos.

## 6. Adjust thresholds (with approval)

The thresholds are constants in the code. There is no settings file. Changing one changes
the output pixels, so it needs Sam's explicit approval and goes through a normal pull request.

**Repair quality checks** (`services/ai/app/engines/perfection_quality.py`):

| Constant | Now | Raise it if… | Lower it if… |
|---|---|---|---|
| `MAX_DELTA_E` | 2.0 | many colour failures look natural (false alarms) | passed repairs show colour shifts |
| `TEXTURE_RANGE` | (0.6, 1.6) | busy-looking repairs pass (lower the top) or natural ones fail as flat (lower the bottom) | |
| `MIN_PORE_SHARE` | 0.5 | repairs without pores pass | natural repairs fail on pores |
| `MAX_EDGE_RATIO` | 2.0 | natural repairs fail on edges | visible repair borders pass |
| `MAX_REPEAT_NCC` | 0.9 | natural repairs fail as repeated | repeated donor patterns pass |
| `MAX_SURROUND_DL` | 1.0 | natural repairs fail on surround | halos around repairs pass |

**Detection** (`services/ai/app/engines/perfection.py`) is more sensitive to changes. Examples:

- `WHITEHEAD_MIN_REL`
- `BLEMISH_MIN_REL`
- `MIN_REL_CONTRAST`
- `PORE_RATIO`
- `MOLE_MIN_REL`
- `BIRTHMARK_D`
- `REVIEW_BAND`
- `VERIFY_RATIO`

Change one constant at a time, on the validation split only. After each change:

```powershell
python -m pytest -q                                   # all engine tests
python -m scripts.perfection_benchmark --baseline qa\baselines\synthetic-2D.json
python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --split validation `
  --json D:\qa-results\validation-report-v2.json --baseline D:\qa-results\validation-report.json `
  --quality-gallery D:\qa-results\quality-review-v2.html --label "MAX_DELTA_E 2.0 -> 2.3"
```

The synthetic gate must stay free of regressions. On the validation split the change should:

- reduce false alarms, or improve recall or removal
- leave **natural features damaged** at 0
- leave texture retention as it was

## 7. Independent validation (test split, once)

When the thresholds are frozen and approved, score the test split **once**:

```powershell
python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --split test `
  --html D:\qa-results\test-report.html --json D:\qa-results\test-report.json --label "release candidate"
```

- Do not change thresholds after looking at the test split. If something has to change, add
  new subjects, then calibrate on validation and score a fresh test set.
- Report the test numbers per tone group, separately from the synthetic numbers. Never pool
  the two.
- This test-split report is the only evidence of real-world performance.
