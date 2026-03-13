# Bias Evaluation — Theory & Methodology

## Analysis and Operationalization of Bias Metrics and Measurement Methods for Large Language Models

## PART I: FUNDAMENTAL CONCEPTS AND TAXONOMY OF BIAS MEASUREMENT

### 1. Introduction: The Need for Continuous, Automated Bias Evaluation

Large Language Models (LLMs) developed by leading AI laboratories (e.g., OpenAI, Deepseek, xAI) are increasingly used in real-world applications. Their rapid evolution—characterized by frequent model updates, new releases, and diverse deployment configurations—poses major challenges for research. Static analyses of bias performed at single points in time quickly become outdated. This exposes a critical gap: the absence of a systematic, continuous evaluation methodology that enables researchers and practitioners to track bias evolution across model generations.

This report aims to establish a comprehensive scientific and methodological foundation for an automated evaluation platform. This requires rigorous extraction, systematization and analysis of metrics, mathematical methods and analytical procedures as documented in the state of the art. A central challenge is variability in model access. Therefore, the evaluation must be modular to cover fundamentally different scenarios:

1. **Black-box evaluation:** Applicable to proprietary models accessible only via APIs (e.g., OpenAI). Analysis is limited to generated text outputs.
2. **White-/grey-box evaluation:** Applicable to open-source models (e.g., Llama variants). Full access to internal model states, including embeddings and logits, enables deeper diagnostic analysis.

### 2. Formal definitions: What is measured?

To make bias measurable, precise definitions are essential.

**Bias vs. Fairness.** In this context, *bias* is defined as "systematic and unfair discrimination against specific individuals or groups," or, more formally, "disparate treatment or outcomes between social groups arising from historical and structural power asymmetries." *Fairness* is the normative objective of minimizing these undesired distortions.

**Taxonomy of Harms.** Harms caused by bias can be classified into two main categories:

1. **Allocational harms:** Relate to unequal distribution of resources or opportunities (typical for downstream classification tasks such as credit scoring).
2. **Representational harms:** Relate to how social groups are represented. This is the primary focus for generative LLM evaluation. Representative harms include:
   * **Stereotyping:** Associating groups with stereotypical attributes.
   * **Denigration/Toxicity:** Generating toxic, hateful, or insulting content about a group.
   * **Exclusionary norms:** Failing to acknowledge or adequately represent certain groups.

**Taxonomy of Bias Dimensions.** A multi-dimensional evaluation platform must measure bias along multiple axes. Common bias categories used as test suites include:
* Gender
* Race / Ethnicity
* Religion
* Age
* Nationality
* Sexual orientation
* Disability
* Physical appearance
* Socioeconomic status

These categories may be insufficient for specialized domains; the platform must be extensible to support domain-specific bias profiles (e.g., **Health Equity Bias** in medical contexts — the failure to account for structural explanations of inequality or disproportionate withholding of opportunities by demographic group in medical advice).

### 3. A methodological taxonomy of bias measurement procedures

Research methods for quantifying bias are most usefully classified by the required level of model access (white/grey/black box).

**Approach 1 — Embedding-based metrics (White-box).** These methods analyze internal model representations (embeddings) under the assumption that semantic associations are reflected by geometric proximity in vector space.
* **Principle:** Measure association strength between vectors for social groups (e.g., "man", "woman") and vectors for stereotype concepts (e.g., "career", "family").
* **Requires:** Full access to word or sentence embedding matrices.
* **Examples:** Word Embedding Association Test (WEAT), Sentence Encoder Association Test (SEAT), categorical association measures (e.g., Cramér's V).

**Approach 2 — Probability-based metrics (Grey-box).** These methods analyze the model's probability distribution (logits) for the next token. They are called grey-box because they require more than the final text output but not full access to model weights.
* **Principle:** Compare the probabilities (commonly pseudo-log-likelihoods) that the model assigns to stereotypical vs. anti-stereotypical sentences.
* **Requires:** Access to logit outputs.
* **Examples:** CrowS-Pairs Score, Context Association Test (CAT), All Unmasked Likelihood (AUL), Log-Probability Bias Score (LPBS).

**Approach 3 — Generated text based metrics (Black-box).** These methods analyze the final text produced by the model in response to prompts and are applicable to all API-based models.
* **Principle:** Evaluate generated text for quality, toxicity, sentiment or stereotypical content.
* **Requires:** Only API access (text in / text out).
* **Examples:** Classifier-based metrics (toxicity, sentiment), distributional metrics (Wasserstein distance).

A key empirical finding is the weak or inconsistent correlation between intrinsic metrics (Approach 1 & 2) and extrinsic, real-world harms (Approach 3). Models trained with RLHF can pass many intrinsic tests by masking explicit bias while still exhibiting implicit or generative bias. For the planned platform, this implies prioritizing a *generation-first* (black-box) approach to obtain robust and actionable results.

## PART II: MATHEMATICAL METHODS AND METRICS IN DETAIL

This part extracts the specific measurement techniques and mathematics required to compute bias metrics.

### 4. White-box: Embedding-level metrics

These metrics apply to open-source models where embedding vectors can be extracted.

#### 4.1. Word / Sentence Embedding Association Test (WEAT / SEAT)

**Concept.** WEAT (and SEAT for sentence encoders) measures differential association of two sets of target concepts (e.g., A = occupations, B = household objects) with two sets of attribute words (e.g., X = male names, Y = female names).

**Mathematical basis (sketch).**

1. Define a similarity function s(w, A, B) for a word w and concept sets A and B, typically using cosine similarity between embedding vectors.
2. Define the test statistic that sums differential associations across words in the attribute sets.
3. The final metric is an effect size d (difference of means normalized by pooled standard deviation). d > 0 indicates stronger association of X with A (and Y with B), d < 0 the opposite; d ≈ 0 indicates neutrality.

#### 4.2. Categorical association (Cramér's V)

**Concept.** Use an auxiliary classifier to quantify statistical correlation between protected attribute space and stereotypical attribute space in embedding space.

**Procedure (sketch).**

1. **Training:** Train a simple linear classifier (e.g., linear SVM) on embeddings of words clearly defining protected attributes (e.g., sets for religion labels).
2. **Inference:** Use the trained classifier to predict protected-attribute labels for embeddings of stereotype-descriptive words.
3. **Measurement:** Build a contingency matrix C of predicted protected labels vs. actual stereotype classes (e.g., positive/negative sentiment).
4. Compute observed vs expected cell frequencies under independence, derive a chi-square-like statistic (MSE-like), and normalize to obtain Cramér's V: V ∈ [0,1] where 0 indicates independence (no bias) and 1 indicates perfect association (maximal bias).

#### 4.3. Layer-wise analysis (Logit Lens)

**Concept.** Diagnose *which Transformer layer* creates or amplifies bias using the Logit Lens approach: project hidden states from each layer back to token logits and compute bias metrics per layer.

**Procedure (sketch).**

1. Propagate an input (e.g., a CrowS-Pairs sentence pair) through the model.
2. For each Transformer layer i, extract the hidden state h_i at the final token position.
3. Project h_i through the model's unembedding matrix to obtain layer-specific logits.
4. Compute a grey-box metric (e.g., PPL score) on the layer-specific logits for each layer.
5. The output is a vector of bias scores per layer, revealing how bias evolves across depth.

### 5. Grey-box: Probability-based metrics

These metrics require access to model logits and underpin benchmarks such as CrowS-Pairs and StereoSet.

#### 5.1. Pseudo-Log-Likelihood (PPL) for counterfactual pairs

**Concept.** Widely used method to measure stereotyping: for a counterfactual sentence pair, check whether the model considers the stereotypical sentence S_st more likely than the anti-stereotypical S_at.

**Application.** CrowS-Pairs, StereoSet.

**Mathematical basis (sketch).**

1. Provide a sentence pair:
   * S_st: "The man is an engineer."
   * S_at: "The woman is an engineer."
2. Compute the pseudo-log-likelihood (sum of conditional log-probabilities of tokens) for each sentence. The All-Unmasked-Likelihood (AUL) formalism separates unchanged context U and the attribute tokens M.
3. The CrowS-Pairs score (bias score) is the proportion of pairs for which the model assigns higher likelihood to S_st than to S_at. Formally: score = (1/N) Σ I( PPL(S_st) > PPL(S_at) ).
4. Interpretation: 0.5 (50%) is ideal (no preference); 1.0 is maximally biased.

#### 5.2. Context Association Test (CAT)

**Concept.** CAT measures association in the reverse direction of PPL: it measures the probability of the protected attribute M given a neutral context U (used in StereoSet). Example: for context U = "The person who works as an engineer is...", compare P(M_st = 'he' | U) versus P(M_at = 'she' | U).

### 6. Black-box: Generated text–based metrics

Black-box methods are the most universal for the planned platform because they apply to any model, including commercial API models.

#### 6.1. Classifier-based metrics

**Concept.** Run prompts that mention various social groups through the LLM; feed generated text into one or more auxiliary classifiers (toxicity, sentiment, regard) to evaluate properties and compute differential scores.

**Examples and measures.**
1. **Toxicity & Sentiment:** Use a toxicity classifier (e.g., Detoxify or Perspective API) and a sentiment classifier (e.g., TextBlob) to score outputs. Metric: difference in mean scores between outputs for group G_i and group G_j.
2. **Demographic Parity Difference (DPD):** Formalize parity measures across group-conditioned outputs using classifier scores.
3. **Regard score:** Use a classifier trained specifically to measure social connotation (positive/negative/neutral) toward groups.

#### 6.2. Distributional metrics

**Concept.** Measure bias as deviation of word distributions in generated text from a reference distribution assumed to be unbiased (Fang et al. methodology).

**Procedure (sketch).**

1. Define a reference corpus o (e.g., Reuters, NYT) as an "unbiased" baseline.
2. Generate a text corpus h from the tested LLM L using the same headlines or themes as o.
3. Compute group-specific word distribution f_o over the reference corpus and f_h^L over the model-generated corpus.
4. Metric: average Wasserstein distance (Earth Mover's Distance) between distributions W(f_h^L, f_o). A higher mean Wasserstein score indicates greater deviation from the reference and hence stronger bias.

#### 6.3. Semantic evaluation

Not implemented in the current prototype.

## PART III: REQUIRED DATA SOURCES (THE "DATABASES")

Metrics described above require specific question sets and lexica. An automated platform must integrate three fundamental types of databases.

### 7.1. Database type 1: Word lists (for white-box metrics)

**Purpose:** Provide attribute and target sets (X, Y, A, B) for geometric metrics (WEAT/SEAT) and training/testing data for the classifier in Cramér's V.
**Content:** Lists of words representing protected attributes (e.g., male_names_de, female_names_de, christian_terms_de, muslim_terms_de) and stereotype attributes (career_words_de, family_words_de, science_words_de).
**Source:** Carefully extracted, translated and validated from literature and prior benchmarks.

### 7.2. Database type 2: Counterfactual benchmarks (for grey-box metrics)

**Purpose:** Input for PPL and CAT metrics.
**Format:** Pairs (S_st, S_at) or tuples differing only by replacement of the protected attribute token.
**Catalog of benchmarks:** CrowS-Pairs (1,508 pairs across nine bias categories), StereoSet (stereotype/anti-stereotype/meaningless options), WinoBias/Winogender (gender coreference), Bias-STS-B (semantic similarity perturbations), PANDA (large perturbation dataset across gender, ethnicity, age).

### 7.3. Database type 3: Prompt collections (for black-box metrics)

**Purpose:** Inputs for generative tests.
**Catalog:** BOLD (23,679 prompts), RealToxicityPrompts (100k prompts), BBQ (Bias Benchmark for QA).

**Table 1: Essential benchmarks for the automated evaluation platform** (summary)
- CrowS-Pairs — measures stereotyping in 9 categories — PPL-based — requires logits.
- StereoSet — measures stereotypes intra-/inter-sententially — PPL / CAT — requires logits.
- BOLD — open-ended demographic prompts — classifier-based — API (text-out).
- RealToxicityPrompts — toxicity generation — classifier-based — API (text-out).

## PART IV: OPERATIONALIZING AN AUTOMATED EVALUATION PLATFORM

This section synthesizes methods into a coherent design for a prototypical implementation.

### 8. Architecture concept and database schema

The heterogeneous access levels require a modular architecture:

* **Module 1 — Black-Box API Evaluator:** Connects provider APIs and implements black-box metrics (classifier-based, distributional). Uses prompt databases (type 3).
* **Module 2 — White/Grey-Box Local Evaluator:** Connects locally loaded open-source models (e.g., via Hugging Face Transformers) and implements metrics from Parts II (embeddings, logits, generated text) using all database types.

Given fragility in bias metrics (small implementation details can yield large differences), the platform should serve as an **orchestration framework** that wraps validated open-source libraries (e.g., LangFair, Hugging Face evaluate-bias), automates their execution, and standardizes outputs. For continuous evaluation, a relational schema is recommended to store temporal results:

* Models: (Model_ID, Name, Provider, Version_Hash, Access_Level)
* Benchmarks: (Benchmark_ID, Name, Type, Bias_Domain)
* Metrics: (Metric_ID, Name, Formula_Reference, Access_Level)
* Evaluations: (Eval_ID, Timestamp, Model_ID, Benchmark_ID)
* Results: (Result_ID, Eval_ID, Metric_ID, Score, Raw_Output_Sample)

### 9. Multi-dimensional scoring and aggregation methods

Aggregating disparate bias metrics into a single scalar ("overall bias") is scientifically unsound. Instead, produce multi-dimensional *bias profiles*:

1. **Profile by bias type:** Vector of scores across bias dimensions (gender, race, etc.).
2. **Profile by metric type:** Vector comparing robustness across measurement methods (embedding-based, probability-based, generation-based).
3. **Profile by use case:** Use-case templates (e.g., "medical assistant", "customer service bot") that weight metrics differently according to domain importance.

### 10. Visualization concepts for an automated app

Essential dashboard visualizations:

1. **Multi-axis bias profile (radar chart):** Each axis a bias type; area visualizes bias signature; multiple models overlayed for comparison.
2. **Bias evolution (time series):** X-axis date of evaluation; Y-axis chosen bias score (e.g., CrowS-Pairs); line per model to track changes across versions.
3. **Layer-wise bias analysis (line chart):** For local models: X-axis transformer layer; Y-axis bias score (e.g., PPL) to visualize where bias emerges.
4. **Distribution comparison (overlaid histograms / density plots):** Reference corpus vs generated corpus; visual discrepancy corresponds to Wasserstein distance.
5. **Qualitative results table:** Interactive table with Prompt and Model response for manual review.

## PART V: RESEARCH HORIZONS AND META-ANALYSIS (ADVANCED TOPICS)

A robust platform must implement current methods and account for their weaknesses and limits.

### 12. Challenge 2: Measuring unanticipated bias

Most benchmarks focus on known bias axes. The platform should also detect *unanticipated* bias against groups not predefined in benchmarks.

**Approach: Uncertainty Quantification (UQ).**

* **Hypothesis:** LLMs exhibit higher uncertainty when queried about underrepresented or poorly covered social groups.
* **Operationalization (conceptual):**
  1. Build a database of niche-group entities.
  2. Query the LLM about these groups.
  3. **Grey-box metric:** Measure entropy of next-token logit distribution.
  4. **Black-box metric:** Measure variance of generated responses across multiple samples at temperature T > 0.
  5. Significant increases in uncertainty signal representation gaps and potential unanticipated bias.

### 13. Summary and scientific framing

The literature supports a multi-metric, multi-method approach that respects access-level differences. Key takeaways:

1. **Robust metrics:** PPL (grey-box) for stereotyping; Wasserstein distance (black-box) for distributional deviations.
2. **Method comparison:** Black-box methods are most universal and detect implicit bias that RLHF can hide. White/grey-box tests provide diagnostic insight into *where* bias arises but require local access.
3. **Operationalization:** The platform should orchestrate validated benchmarks (CrowS-Pairs, BOLD) across model interfaces and output multi-dimensional bias profiles visualized in radar and time-series charts.

