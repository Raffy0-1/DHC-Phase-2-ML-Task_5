# Auto Tagging Support Tickets Using LLM

Automatically classify customer support tickets into meaningful categories using three different LLM-based approaches — zero-shot, few-shot, and fine-tuning — then compare their performance side by side.

---

## Objective

Support teams receive hundreds of tickets daily across billing, delivery, account issues, and more. Manually tagging each one is slow and inconsistent. The goal of this project is to build a pipeline that reads a raw customer support message and automatically assigns the top 3 most relevant tags to it — using large language models at different levels of complexity and training effort.

The three approaches are compared head-to-head so you can understand the real trade-off between speed of setup and classification accuracy.

---

## Dataset

**Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (Hugging Face)  
**Size:** 26,872 real customer support utterances  
**Original labels:** ~20 specific intents (e.g. `cancel_order`, `payment_issue`, `track_order`)  
**Remapped to 7 tag categories:**

| Tag | Covered intents |
|-----|-----------------|
| `billing` | check_invoice, get_refund, payment_issue, track_refund |
| `delivery` | cancel_order, track_order, delivery_options, change_shipping_address |
| `account_access` | create_account, delete_account, recover_password, registration_problems |
| `technical_issue` | check_cancellation_fee, newsletter_subscription |
| `customer_service` | contact_human_agent, complaint |
| `positive_feedback` | review, feedback |
| `general_complaint` | fallback for unmatched intents |

The dataset contains real linguistic variation — typos, colloquial language, placeholder tokens like `{{Order Number}}` — making it a realistic benchmark for support ticket classification.

---

## Methodology / Approach

The project is structured as a single end-to-end notebook covering five sequential phases.

### Phase 1 — Data loading and exploration
Loading the Bitext dataset from Hugging Face, mapping the original intents to the 7 tag categories, and exploring text length distributions and class balance before any modeling begins.

### Phase 2 — Zero-shot classification
Using `facebook/bart-large-mnli` via Hugging Face's `pipeline()` API. This model was trained on Natural Language Inference (NLI) and repurposes that capability for classification — each candidate tag becomes a hypothesis, and the model scores how likely it is to follow from the ticket text. No training examples are provided. Outputs top 3 tags ranked by confidence score.

### Phase 3 — Few-shot classification
Using `google/flan-t5-base`, an instruction-tuned text-to-text model. A structured prompt is built that includes one hand-picked example per category, followed by the unseen ticket. The model learns the expected format from the examples in the prompt itself — no weight updates occur. The output is parsed and matched back to the known tag vocabulary.

### Phase 4 — Fine-tuning BERT
Using `bert-base-uncased` with a custom classification head added on top. The model is fine-tuned end-to-end on the full 26,872-ticket dataset using an 80/20 train/validation split. Multi-label classification is handled with `BCEWithLogitsLoss` instead of the default `CrossEntropyLoss`. Training runs for 3 epochs on a T4 GPU. The trained model and tokenizer are saved directly to Google Drive so any Colab session restart loads the saved model instead of retraining.

### Phase 5 — Evaluation and comparison
All three methods are evaluated on a stratified 196-ticket sample (drawn proportionally across all 7 categories). Metrics used: **precision**, **recall**, and **F1 score** with `average='samples'` — the standard choice for multi-label classification. A grouped bar chart and per-class F1 breakdown are saved to the `visualizations/` folder.

---

## Key Results

Evaluated on 196 tickets (stratified sample across all 7 categories):

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Zero-shot (`bart-large-mnli`) | 0.245 | 0.735 | 0.367 |
| Few-shot (`flan-t5-base`) | 0.143 | 0.143 | 0.143 |
| Fine-tuned (`bert-base-uncased`) | **0.333** | **1.000** | **0.500** |

### Observations

**Fine-tuned BERT achieved the highest F1 (0.50) and perfect recall (1.0).** The recall of 1.0 means it successfully identified the correct tag in every single ticket's top-3 predictions. The precision of 0.33 reflects that when predicting 3 tags per ticket, roughly one of them is always right — a strong result given the multi-label setup where only one true tag exists per ticket.

**Zero-shot performed surprisingly well with an F1 of 0.37** despite having no training data at all. Its recall of 0.73 shows it catches the correct tag the majority of the time, though it struggles with precision — it returns top 3 labels, several of which are often wrong. Using natural language label descriptions (`'billing and payment'` instead of just `'billing'`) helped significantly here.

**Few-shot underperformed at F1 0.14.** The `flan-t5-base` model consistently predicted `billing` regardless of input, indicating that the 6-example prompt was not enough to override the model's default tendency for this particular label. A larger instruction-tuned model (e.g. `flan-t5-xl` or an API-based model like GPT-4) would likely close this gap substantially.

**The dataset contains placeholder tokens** like `{{Order Number}}` and `{{Account Type}}` from the data generation process. These do not affect BERT's fine-tuning significantly — BERT learns to recognize surrounding context — but they reduce the realism of zero-shot and few-shot prompts where the model expects natural text.

---

## Project Structure

```
auto-tagging-support-tickets/
│
├── auto_tagging_support_tickets.ipynb   # main notebook — all 5 phases
├── requirements.txt                     # Python dependencies
├── .gitignore                           # files excluded from version control
├── README.md                            # this file
│
└── visualizations/                      # saved charts from the notebook
    ├── data_exploration.png             # word count + tag distribution (Phase 1)
    ├── comparison_chart.png             # zero-shot vs few-shot vs fine-tuned (Phase 5)
    └── per_class_f1.png                 # per-category F1 for fine-tuned BERT (Phase 5)
```

---

## Models Used

| Model | Used for | Parameters |
|-------|----------|------------|
| `facebook/bart-large-mnli` | Zero-shot classification | 406M |
| `google/flan-t5-base` | Few-shot classification | 250M |
| `bert-base-uncased` | Fine-tuning | 110M |

---

## Environment

- **Platform:** Google Colab (T4 GPU)
- **Python:** 3.10
- **Key libraries:** `transformers`, `datasets`, `torch`, `scikit-learn`
- **Drive integration:** Fine-tuned model checkpoint saved to `/MyDrive/auto_tagging_project/bert_finetuned/` — Colab restarts load from Drive, never retrain

---

## Skills Demonstrated

- Prompt engineering for classification tasks
- Zero-shot and few-shot learning with pre-trained LLMs
- Multi-label text classification with BERT
- Custom PyTorch Dataset and Trainer classes
- Evaluation with F1, precision, and recall (sample-averaged, multi-label)
- Google Drive checkpoint management for interrupted GPU sessions
