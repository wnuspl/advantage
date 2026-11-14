# Maintaining the dataset

How to keep `tennis.db` current. The design behind each step is in `docs/DESIGN.md` ("Build protocol", "AcesAPI"). Run every command from the project root with the venv active (`.venv\Scripts\Activate.ps1`).

## The one rule

`tennis.db` is an output. Everything in it is rebuilt from three tracked inputs, plus the raw folders they point to:

| Input | What it holds | Who edits it |
|---|---|---|
| `data/build_manifest.json` | Which raw snapshot folders the build uses | `fetch_acesapi.py --pin`, or you by hand (MCP) |
| `data/overrides.csv` | Your decisions on names and matches the build couldn't resolve | You |
| `data/snapshot_lock.json` | sha256 of every raw file ever loaded | The build (commit it) |

Never fix data inside `tennis.db` with SQL: the next build discards it. Put the fix in `data/overrides.csv`. `models.db` is never rebuilt; its rows use stable source keys, so they survive rebuilds.

## Weekly: new results (AcesAPI)

```
python scripts/fetch_acesapi.py --pin
python scripts/build_db.py
```

1. **Fetch.** This writes a new folder `data/raw/acesapi-<UTC time>/` with every tournament that has started since the last pull, plus recent ones again (AcesAPI corrects results for a few days). It appends the folder to the manifest and logs the pull in `data/SOURCES.md` (draws, credits spent, credits left).
   - The key is `ACESAPI_KEY` in `.env` (gitignored). Never put it anywhere else.
   - Only want to see what it would fetch? Run `--plan` (costs only the list calls, 2-3 credits).
   - `--max-credits` (default 200) caps one pull. Whatever is left over is picked up by the next pull.
2. **Build** (~10 min). Close anything that has `tennis.db` open first: DB browsers, notebooks, running sweeps or tests. Otherwise the final swap fails and the new database waits at `tennis.db.building`.
3. **Check** the end of the build output and `data/build_report.json` (see "Reading the build report").
4. **Commit** `data/build_manifest.json`, `data/snapshot_lock.json` and `data/SOURCES.md`. `data/raw/` is gitignored, so back it up some other way: the pulls can't be re-downloaded as they were.

**Credits.** The free plan gives 500 a month, topped up near the monthly anniversary of the account (no rollover). A weekly pull costs about 15-30 credits (lists, the week's new tournaments, and recent ones fetched again). `credits_left` is on each pull's line in `data/SOURCES.md` and in its `pull.json`. Under 100 credits, AcesAPI slows to 10 requests a minute, and a pull takes longer.

**Skipped a few weeks?** Just run it. A pull fetches everything since the last pull's tournaments, using the folders already on disk.

## Monthly: Match Charting Project

MCP is updated upstream every few months (the last commit is in `data/SOURCES.md`).

```
python scripts/fetch_data.py tennis_MatchChartingProject
```

If it downloaded a new commit, point the manifest at it by hand. In `data/build_manifest.json`, in the snapshot whose `repo` is `JeffSackmann/tennis_MatchChartingProject`, change `folder`, `commit` and `fetched_at` to the new folder. Then run `python scripts/build_db.py`. If the build stops with "match no group and no ignore pattern", the repo added a file type: add a group or an `ignore` rule for it in that snapshot entry.

The Sackmann mirror (`tennis-sackmann-archive`) is frozen. Don't expect updates from it.

## As needed: the review queue

Names and matches the build couldn't resolve with confidence go to `review_queue`; nothing is guessed. A match whose player is in the queue stays out of `matches` until the name is resolved.

```
python scripts/export_review.py
```

This writes `data/review_open.csv` (gitignored): one row per open item, with candidates as stable keys. For each one you can confirm, add a line to `data/overrides.csv`:

```
entity_type,source,source_key,action,target_source,target_key,note
player,acesapi,wta:1234,link,sackmann_wta,211148,same player: married name
player,acesapi,atp:5678,forbid,sackmann_atp,104925,different person with the same name
```

- `link`: this source key is that player (or match). `forbid`: it is not; the resolver tries other candidates.
- Targets are stable keys of the backbone: `sackmann_atp,<id>` or `sackmann_wta,<id>` (two columns). Never a `player_id`: those change on every rebuild. Overrides load before AcesAPI rows, so a player AcesAPI created can't be a target. If an AcesAPI name is a genuinely new player that the build queued instead of creating, there is no override for that yet: leave it queued and note it.
- AcesAPI player keys are `atp:<AcesAPI player id>` / `wta:<...>`. The id is in the row's `detail` and in the raw draw files.
- An override whose target doesn't exist stops the build. That's deliberate.
- Keep notes free of commas (or quote the field): the file is plain CSV.

Rebuild after editing, then commit `data/overrides.csv`. Worth doing first: names that hold back the most matches (`player_unresolved` rows in the queue, by player).

## Reading the build report

`python scripts/build_db.py` prints table counts and open review items. `data/build_report.json` has the details:

| Field | Healthy | If not |
|---|---|---|
| `invariants` | all `ok` | The build stopped before swapping, so the old `tennis.db` is still in place. The listed rows say what broke; the half-built file is at `tennis.db.building` for inspection. |
| `acesapi.last_match_date` | within a day or two of the last pull | The pull didn't run or saved no draws: check `data/SOURCES.md`. |
| `acesapi.linked_to_backbone` | stable from build to build (819 on 2026-11-14) | A drop means AcesAPI changed names or ids for already-overlapping events. |
| `acesapi.new_players` | grows slowly (debuts) | A jump suggests a name format change. Look at the new `player` rows with `player_xref.source='acesapi'` and `method='seed'`. |
| `acesapi.revised_results` | small | Results AcesAPI corrected after first reporting them. Informational. |
| `skip_reasons` for `acesapi_draws` | `not_final`, `superseded_by_later_pull` only | `unmapped_round` or `unknown_outcome` = AcesAPI used a new label. Run `python scripts/fetch_acesapi.py --inspect data/raw/<pull folder>` to see the values, then extend `map_round` / `match_status` in `src/tennis/acesapi.py` and add a test. |
| `review_queue` | slowly growing | See "the review queue". |

## When the build stops

| Message | Cause | Fix |
|---|---|---|
| `raw files changed under a pinned commit` | A file in a pinned folder was edited or replaced | Restore it from backup. Raw folders are never edited; fetch a new snapshot instead. |
| `pinned folder not found` | A folder in the manifest isn't in `data/raw/` | Restore it, or remove its entry from the manifest (its data then disappears from the build). |
| `match no group and no ignore pattern` / `patterns matched no file` | The folder's layout differs from its manifest entry | Fix that snapshot's `groups`/`ignore` in the manifest. |
| `invariant checks failed` | Data that breaks a rule (e.g. a player in two events on one day) | Read the rows printed. Usually a bad link: fix it with a `forbid` override. |
| `could not replace tennis.db` | Something has `tennis.db` open | Close it and rerun, or move `tennis.db.building` into place yourself. |
| `saved but NOT pinned` (fetcher) | The manifest layout was unexpected | Add the entry the fetcher prints to the `snapshots` list by hand. |

## Don'ts

- Don't edit, rename or delete files inside `data/raw/<snapshot>/`. A new download is a new folder.
- Don't delete old AcesAPI pull folders that are still pinned. Older pulls are what `match_seen.first_seen_at` (when a result was first known) is computed from.
- Don't commit `.env`, `tennis.db`, `models.db` or `data/raw/`.
- Don't redistribute AcesAPI data (their terms) or Sackmann-derived data commercially (CC BY-NC-SA 4.0).
