# Lessons carried into the raw quote engine

This engine starts with the user's transaction and quote inputs. It computes new diagnostics and features, and reports the resulting run. Historical BondCliQ findings motivate the safeguards below; they are not new evidence for another dataset. Synthetic runs are demonstrations only. No historical result numbers, private records or fitted models are bundled with these lessons.

## Population, identifiers and metadata

Use the supplied transaction universe as the coverage denominator. A quote feed can contain many bonds irrelevant to the evaluated trades. Conversely, a traded bond with no quote must remain visible. The engine distinguishes quotes anywhere in the supplied file from usable information at a transaction's prediction time. Neither measure is a complete-market coverage estimate.

Keep stable transaction IDs and join by identity. Two records at the same timestamp can be different trades; group order is not a join key. Duplicate IDs must be resolved upstream, not paired positionally. Diagnostic sector and issuer mappings retain unknown or conflicting labels instead of choosing an arbitrary first label. A descriptive mapping across the supplied period is not automatically a causal historical mapping for model training.

Sparse history is a proxy. A trailing 30-day count is different from a prior-calendar-month count, and both depend on the supplied feed's coverage and filters. The engine reports its actual history field and available-history indicators. Zero observed trades cannot establish zero market activity. A loader's liquidity flag must not become a real-time label without a verified definition and timing contract.

## Time, late messages and resets

The explicit known/received timestamp determines quote availability. A quote with an earlier original timestamp that arrives later cannot be backdated for prediction. Original timestamps are retained as context when provided. Timezone conversion must preserve instants; a naive New York clock reading is not UTC. Ambiguous or invalid timestamps need explicit upstream handling.

As-of states use the latest eligible message, with exact-time inclusion controlled by configuration. Selecting an older good message after the latest message became incomplete creates a false clean history. State resets at the configured local-day boundary; there is no overnight forward fill. Movement and Path continuity also break at incomplete endpoints, long gaps and incompatible conditions. Append-only future messages must not rewrite earlier features.

The shared implementation uses legacy column names such as `quote_timestamp_ET`. Their spelling does not override the configured timezone. Calendar/time assumptions remain explicit in run provenance.

## Repetition, simultaneous candidates and quantities

A raw-row duplicate is not a separate price vote. Exact source duplicates are marked before generated IDs or wide bid/ask expansion. A duplicated wide source message can yield two duplicate side rows; source-message and expanded-side counts have different denominators.

One dealer/bond/side/known-time event can contain several spread–quantity candidates. The engine keeps distinct candidate sets and does not arbitrarily select one row. Repeated candidates do not receive extra weight. An even-cardinality median can fall between observed candidates; it is a descriptive center, not proof that the dealer offered that price.

Positive, zero, missing and invalid quote quantity have separate meanings. Zero is not a confirmed small trade or a default size. Negative, malformed or nonfinite quantities are not silently relabeled ordinary positive sizes. Wide inputs use each configured side's size when supplied; a missing side size does not silently borrow a common size.

Transaction quantity, previous transaction quantity and quote quantity are different variables. Transaction `quantity_scale` and `quote_quantity_scale` are independent. Scaling can normalize units only when their source meanings are known. Same positive quantity matching uses equality of supplied labels. Multiple prices at one size disprove a unique observed size-to-price mapping; different sizes co-occurring with different prices do not by themselves establish a size effect or an executable curve.

## Incomplete values, crossing and value units

Finite zero and negative spread observations are retained. They are not inherently corrupt. Nonfinite candidates keep an event incomplete so an apparently clean subset cannot replace the full observed state. The engine exposes support and availability instead of forcing unavailable values to zero.

This engine accepts spreads only and rejects price input. A price cannot be converted to a spread by renaming a column; perform a justified conversion upstream. Quote values and target/anchor levels must share canonical units after their explicit scales. `error_scale` then converts prediction error into the displayed unit. The legacy compatibility column `spread` and feature suffix `_bps` do not independently establish economic units.

Signed candidate bounds distinguish no crossing, some crossing and all crossing. A target-level mean gap cannot identify which individual dealer pair crossed. Stale messages, asynchronous sides, different quantities, simultaneous candidates and conflicting source conventions can coexist. The report provides distributions and explanatory events; it does not certify a causal explanation for every negative gap. It also reconstructs at most eighteen fixed transaction queries with the shared paired-slot calculator. Policy A uses all fresh slots, B selects matching-eligible slots with original candidates, and C matches candidates on exactly the B slots. The report separately shows A-to-B selection changes and B-to-C same-slot matching changes.

Compare size/time policies on identical eligible slots before attributing a change to matching. Otherwise changed coverage can masquerade as better prices. Unknown size, no positive size match and no fresh pair are separate states.

## Age, support and movement

Message age measures time since the latest observed message. Change age measures time since an observed genuine state change and can be unknown at the first observation. Median message and change ages can use different assessable dealer subsets, so one summary being smaller does not violate chronology. A recent unchanged refresh does not necessarily mean a recent economic move.

One dealer with zero dispersion is not consensus. Dealer counts need not represent independent information sources. The labels “no supported side” and “no fresh pair” describe a rule's eligibility; they do not imply that no quote or peer observations exist.

Matched-dealer direction separates temporal movement from dealer composition changes. Aggregate levels across different bonds are not comparable movements. Other-bond issuer features exclude the target bond and require sufficient donor support. Changing donor membership or quote conditions remains a composition problem, not evidence of a latent common factor. The engine does not estimate a dealer-by-bond latent-factor model.

Path descriptors distinguish unchanged quote/quantity refreshes from movement. Equality of center alone is insufficient: different candidate sets can have the same center. Changes in quantity sets or candidate counts reset comparability. The last signed move may precede the trailing activity window; directional activity shares use the declared lookback. Missing comparable history stays unknown rather than becoming zero movement.

## Baseline, anchor and evaluation choices

The historical research retained its target and reconstruction anchor, but a reusable engine must use the caller's explicit contracts. When the caller supplies `target = actual - anchor + adjustment`, reconstruction subtracts the same supplied adjustment; an explicit actual column remains the scoring truth. This engine does not silently transplant BASE14, reinterpret transaction direction, or infer buy/sell perspective from an unreliable trade-type code. Existing baseline features may already contain issuer, prior quantity and previous-trade context; quote-family gains must be measured beyond that baseline.

A previous-trade anchor can be far from the current target, including when the previous trade was much smaller. That is a diagnostic question, not justification for replacing the anchor after observing the answer. Pretrade CPP–anchor discrepancy and hindsight target–anchor error are different quantities. The engine's optional fixed 5/10-bps diagnostic groups compare a user-supplied proxy with the anchor after subtracting an explicitly mapped rollover adjustment, then apply `error_scale`. No proxy or historical yield-roll offset is inferred. Missing unit contracts do not support these bps groups. The scalar adjustment does not repair earlier quote benchmarks or cross-time benchmark shifts. Discrepancy does not certify a bad anchor.

Keep short maturity, long maturity and unknown maturity distinguishable. Short instruments can dominate absolute error even when they are a small fraction of trades; do not delete them to make gains look larger. Report any chosen business focus while preserving the complete evaluation population and support.

Both models must be scored on the same finite target/prediction rows, with coverage reported separately. MAE is record weighted unless explicitly changed. P95 cannot be reconstructed by averaging subgroup P95s. A fixed-threshold excess contribution can improve while pooled P95 worsens; these answer different questions. Small groups and tiny percentage differences are not sufficient evidence of practical or statistical benefit.

Select candidates on a declared Validation stage, freeze the choice and configuration, and then evaluate Test. If earlier Test exposure is unknown, preserve that uncertainty instead of claiming independence. More exploratory slices do not create more independent evidence. A descriptive 5% business threshold is not a significance test. Leave-one-date-out removes saved error rows without refitting, uses remaining-record weights, and is not a confidence interval or proof that every date improved.

## What the report establishes, and what it does not

The generated report reads only the current run's computed tables, comparison objects, configuration and provenance. It includes population coverage, exact-event ambiguity, quantity categories, target-time side/pair availability, age/gap distributions, feature availability, optional fixed priority groups and deterministic random/typical/high-range cases. High candidate range is an ambiguity ranking, not proven predictive harm. Cases explain mechanisms and never estimate prevalence.

Generic comparison exports contain aggregates by default. Raw diagnostics include local bond/dealer identities and a few exact event examples; review them before sharing. Run outputs, private inputs and local tests should remain outside source control.

The engine cannot certify feed completeness, quote executability, source quantity units, original timestamp correctness, metadata availability at every historical instant, or causal reasons for every outlier. Its history-coverage flags establish only what is visible in supplied data. Reproducible output makes these assumptions inspectable; it does not remove them.

## Walk-forward evidence and limits

Repeated chronological validation tests sensitivity to which dates train and evaluate a model. Each fold needs fresh fits, training-only category vocabularies and nonoverlapping held-out records. Expanding and rolling history answer different deployment questions; declare the policy before examining fold results. The raw engine reserves the outer Test separately, selects on pooled out-of-fold validation records and refits only Base and the selected family on development before an explicit final test.

Pooled loss gives every scored record its stated weight. Averaging fold MAEs without considering fold size answers a different question. Correlated securities and overlapping training windows prevent treating fold count as the number of independent experiments; descriptive fold stability is not a confidence interval. Reusing causal quote features avoids redundant aggregation, but supplied baseline features can still leak if their upstream rolling calculations used future information.
