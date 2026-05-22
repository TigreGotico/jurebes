# Text featurization

Classical-ML text classifiers operate on numeric vectors. The mapping from a raw utterance to a vector is *featurization*, and the choices made there often matter more than the choice of classifier.

## Bag of words

Treat a document as a multiset of words; ignore order. With vocabulary $V = \{w_1, \ldots, w_d\}$, the feature vector is $\mathbf{x} \in \mathbb{R}^d$ with $x_j$ = some statistic of $w_j$'s occurrence in the document.

- **Binary**: $x_j = \mathbb{1}[w_j \in \text{doc}]$. `CountVectorizer(binary=True)` in sklearn.
- **Term frequency (TF)**: $x_j = \text{count}(w_j, \text{doc})$. `CountVectorizer()`.
- **TF-IDF**: TF scaled by inverse document frequency. The default in jurebes.

## TF-IDF

For term $t$ and document $d$ in a corpus $D$:

$$\text{tf}(t, d) = \text{count}(t, d)$$

$$\text{idf}(t, D) = \log \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1$$

$$\text{tfidf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D)$$

The `+1` in the numerator and denominator is *smoothing* — it prevents division by zero and downweights extremely rare or universal terms. sklearn's `TfidfVectorizer` applies this by default and L2-normalises each document vector.

### Sublinear TF

Raw counts grow linearly with document length. *Sublinear* TF damps this:

$$\text{tf}_{\text{sub}}(t, d) = 1 + \log(\text{tf}(t, d))$$

`jurebes.featurizers.tfidf_word_sublinear()` enables `sublinear_tf=True`. Helpful when documents vary widely in length.

## n-gram features

Single words discard local order. n-grams of length $n \geq 2$ retrieve some of it:

- **Word n-grams.** `TfidfVectorizer(ngram_range=(1, 2))` extracts unigrams and bigrams. Captures phrases like `"good morning"` as a single feature.
- **Character n-grams.** `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))` extracts 3–5 character substrings within word boundaries. Captures sub-word morphology and is robust to typos and OOV words.

`jurebes.featurizers.tfidf_char()` returns the canonical char_wb 3–5 vectorizer. `tfidf_word()` returns the canonical word vectorizer.

## char_word_union

Sometimes both word and character signals matter. `FeatureUnion` concatenates two featurizer outputs:

```python
from jurebes.featurizers import char_word_union
char_word_union()    # FeatureUnion of tfidf_word() and tfidf_char()
```

The baseline `union_logreg` uses this exact union with a `LogisticRegression` classifier.

## Hashing trick (Weinberger et al. 2009)

A `HashingVectorizer` hashes each token into a fixed-size feature space, skipping the vocabulary-building step. Pros:

- Constant memory regardless of corpus size.
- Online-friendly: new documents do not enlarge the vocabulary.
- No state at fit time.

Cons:

- Hash collisions conflate distinct tokens; with $2^{18}$ features collisions are rare for typical vocabularies but possible.
- The inverse mapping (which token hashed where) is not retained.

`jurebes.featurizers.hashing_word(n_features=2**18)` and `hashing_char(...)` are the bundled hashing featurizers. The baselines `hashing_sgd_log` and `hashing_sgd_hinge` chain them with an `SGDClassifier`.

## Skip-grams

A skip-gram generalises the contiguous $n$-gram: an $n$-token tuple drawn from the document where up to $k$ token positions may be skipped between any two consecutive members (Guthrie et al. 2006). With $k = 0$ a skip-gram is an ordinary $n$-gram; larger $k$ captures longer-range co-occurrence while keeping the feature an $n$-tuple.

For the sentence `a b c d`, the $2$-skip-bigrams with $k = 1$ are `a b`, `a c`, `b c`, `b d`, `c d` — the pair `a d` is excluded because its members are three positions apart.

`jurebes.featurizers.skipgram_word(n=2, k=2)` returns a `TfidfVectorizer` with a hand-rolled skip-gram analyzer; the baseline `skipgram_logreg` chains it with a `LogisticRegression`.

## BM25 weighting

Okapi BM25 (Robertson & Zaragoza 2009) replaces TF-IDF's linear term-frequency with a saturating one and adds document-length normalisation:

$$\text{score}(t, d) = \text{idf}(t) \cdot \frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot (1 - b + b \cdot dl / \text{avgdl})}$$

The $k_1$ parameter (default $1.5$) controls how quickly term frequency saturates; $b$ (default $0.75$) controls how strongly long documents are penalised. The idf term uses the BM25 smoothing $\log(1 + (N - df + 0.5)/(df + 0.5))$, which stays non-negative.

`jurebes.featurizers.BM25Transformer` sits after a `CountVectorizer`; `bm25_word()` builds the two-step pipeline. The baselines `bm25_logreg` and `bm25_linear_svc` use it.

## Reduced-dimensionality

TF-IDF produces sparse high-dimensional vectors (one column per vocabulary entry). For some downstream classifiers — kernel SVM, MLP, QDA — a dense low-dimensional representation works better:

- **LSA (Latent Semantic Analysis).** TruncatedSVD on the TF-IDF matrix. Recovers latent topics as the top-$k$ singular vectors. `jurebes.featurizers.lsa(n_components)`.
- **NMF (Non-negative Matrix Factorisation).** Constrains factors to be non-negative, yielding additive parts-based components. `jurebes.featurizers.nmf(n_components)`.
- **LDA (Latent Dirichlet Allocation).** Probabilistic topic model with Dirichlet priors. `jurebes.featurizers.lda_topics(n_topics)`.
- **Autoencoder.** Neural bottleneck via `SklearnAutoencoder`; learns a non-linear compressed representation. `jurebes.featurizers.autoencoder()`.
- **Random projection.** Project the TF-IDF matrix onto a lower-dimensional random subspace. The Johnson-Lindenstrauss lemma guarantees pairwise distances are preserved within a bounded distortion, and `SparseRandomProjection` accepts sparse input directly. `jurebes.featurizers.random_projection(n_components, base)`; the baseline `random_projection_logreg` chains it with a `LogisticRegression`.

See [dimensionality-reduction.md](dimensionality-reduction.md) for when each helps.

## Linguistic features

The featurizers above are lexical — they key on surface tokens. Three optional-dependency featurizers add morphological and syntactic signal:

- **POS sequence.** Tag each utterance with `brill_postagger` and keep only the part-of-speech tags, dropping all lexical content: `"como está"` becomes `"SCONJ VERB"`. A purely syntactic representation. `jurebes.featurizers.pos_sequence(lang)` (`[postag]` extra).
- **Word + POS.** Emit hybrid `token__POS` tokens (`"esta__VERB"`); the double underscore survives sklearn's default `\w+` token pattern, so lexical and grammatical signal share one channel. `jurebes.featurizers.word_pos(lang)` (`[postag]` extra).
- **Stemming.** Collapse morphological variants with `nltk`'s Snowball stemmer (`"running"` → `"run"`) so a TF-IDF tail shares mass across inflections. `jurebes.featurizers.stemmed_tfidf(lang)` (`[stem]` extra).
- **Lemmatisation.** Reduce tokens to their dictionary form with `simplemma` (`"better"` → `"good"`). `jurebes.featurizers.lemmatized_tfidf(lang)` (`[lemma]` extra).

`brill_postagger` ships pretrained models for 11 languages (ca/da/de/en/es/eu/fr/gl/it/nl/pt); `pos_sequence` and `word_pos` raise `ValueError` outside that set. The Snowball stemmer covers 15 languages and `simplemma` covers many more. The baselines `pos_sequence_logreg`, `word_pos_logreg`, `stemmed_logreg` and `lemmatized_logreg` default to English.

## Text statistics

Beyond bag-of-words, simple shape features sometimes help: number of characters, number of words, mean word length, ratio of digits, ratio of upper-case characters, ratio of punctuation, OOV rate vs training vocabulary. `jurebes.featurizers.text_stats()` returns these as a seven-column dense vector. The baseline `text_stats_logreg` uses them alone; `union_text_stats_logreg` concatenates them with TF-IDF.

## Picking a featurizer

| signal | featurizer |
| --- | --- |
| general default | `tfidf_word()` |
| typo-prone, short utterances, OOV | `tfidf_char()` or `union(tfidf_word, tfidf_char)` |
| long-range co-occurrence | `skipgram_word(2, 2)` |
| document-length normalisation | `bm25_word()` |
| extremely large corpora | `hashing_word(2**18)` |
| dense input required | `lsa(50)`, `autoencoder()` or `random_projection(200)` |
| topic structure | `nmf(50)` or `lda_topics(20)` |
| metadata-style features | `text_stats()` |
| morphology / syntax | `stemmed_tfidf()`, `lemmatized_tfidf()`, `pos_sequence()`, `word_pos()` |

The baselines in `jurebes.baselines` cover the most-used combinations of featurizer and classifier; build custom ones by combining `Pipeline` steps directly.

---
- Back to [docs index](../index.md)
