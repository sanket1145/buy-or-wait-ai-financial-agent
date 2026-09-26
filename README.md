# Buy or Wait — AI Financial Decision Agent

A financial decision-making agent built for the **HackerRank Orchestrate — Buy or Wait** challenge. The system analyzes a user's financial history, recurring commitments, upcoming payments, income, minimum-balance requirements, and available payment options to determine whether a purchase is safe to make now, should be planned, delayed, or avoided.

The project combines **financial data processing, 90-day cash-flow forecasting, rule-based decision logic, payment-plan generation, OCR-based data recovery, and automated validation** into an end-to-end financial decision pipeline.

## 🚀 Key Features

* **Purchase affordability analysis** based on current and projected financial position
* **90-day cash-flow forecasting** using historical financial events
* **Recurring expense and commitment analysis**
* **Minimum-balance protection** to prevent unsafe spending
* **Payment-plan generation** for purchases that cannot be paid safely in full immediately
* **Future safe-payment date calculation**
* **Partial-payment and installment evaluation**
* **Spending-adjustment recommendations** when applicable
* **OCR-based recovery** of missing transaction amounts from supporting images
* **Financial boundary handling** using monetary tolerance for floating-point calculations
* **Automated validation** of the final 250-request evaluation output

## 🧠 Decision Framework

For every purchase request, the decision engine follows a structured evaluation process:

```text
                    Purchase Request
                           │
                           ▼
                Financial Data Validation
                           │
                           ▼
                 90-Day Cash-Flow Forecast
                           │
                           ▼
              Can the purchase be paid safely?
                           │
              ┌────────────┼────────────┐
              │            │            │
             YES           NO           │
              │            │            │
              ▼            ▼            ▼
       Affordable Now   Valid Plan?   Safe Later?
                            │            │
                         YES│         YES│
                            ▼            ▼
                    Affordable With   Affordable
                         Plan           Later
                                         │
                                         │ NO
                                         ▼
                                  Not Affordable
```

### Affordability Outcomes

| Status                 | Meaning                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `affordable_now`       | The purchase can be paid safely in full immediately.                                      |
| `affordable_with_plan` | The purchase requires a valid payment plan or partial payment.                            |
| `affordable_later`     | The purchase is not safe now but becomes affordable on a future date within the forecast. |
| `not_affordable`       | No safe payment path exists within the available constraints and forecast period.         |

## 💳 Payment Methods

The system evaluates available payment options and can recommend:

* `full_payment`
* `partial_payment`
* `installments`
* `wait`
* `not_recommended`

Payment plans consider factors such as payment deadlines, available balance, minimum required balance, user preferences, and supplied payment options.

## 📊 Dataset & Evaluation

The complete evaluation pipeline processes:

| Data                  |  Volume |
| --------------------- | ------: |
| Evaluation requests   |     250 |
| Financial events      |  25,342 |
| User profiles         |     275 |
| Payment options       |     790 |
| Messages              |     215 |
| Exchange-rate records |     134 |
| Supporting images     |      16 |
| Forecast horizon      | 90 days |

### Final Evaluation Results

| Affordability Status   | Requests |
| ---------------------- | -------: |
| `affordable_now`       |       74 |
| `affordable_with_plan` |       64 |
| `affordable_later`     |       48 |
| `not_affordable`       |       64 |
| **Total**              |  **250** |

## 🔍 Example Decision

Example evaluation for `request_46`:

```text
Amount safe to pay:       ₹48,133.81
Affordability status:     affordable_with_plan
Payment method:           partial_payment

Payment plan:
2024-12-03 → ₹48,133.81
2024-12-15 → ₹1,316.19

Earliest full-payment date:
2024-12-15

Spending changes required:
None
```

The decision engine determines that the requested purchase cannot be safely completed in a single payment while satisfying the user's financial constraints, but it can be completed using a valid partial-payment plan.

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Financial Data Analysis**
* **Cash-Flow Forecasting**
* **Rule-Based Decision Engine**
* **OCR / Image-Based Data Recovery**
* **Data Validation**
* **CSV / Structured Dataset Processing**
* **Git & GitHub**

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Input Datasets    │
                    │                     │
                    │ Requests            │
                    │ Profiles            │
                    │ Transactions        │
                    │ Payment Options     │
                    │ Messages / Images   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   main.py           │
                    │ Data Loading &      │
                    │ Validation          │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
       ┌─────────────────────┐   ┌─────────────────────┐
       │ financial_engine.py │   │ decision_engine.py  │
       │                     │   │                     │
       │ 90-Day Forecast     │──▶│ Affordability       │
       │ Cash Flow           │   │ Payment Planning    │
       │ Recurring Expenses  │   │ Safe Dates          │
       └─────────────────────┘   └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Final Validation    │
                                  │ & output.csv        │
                                  └─────────────────────┘
```

## 📁 Project Structure

```text
buy-or-wait-ai-financial-agent/
│
├── code/
│   ├── main.py
│   ├── financial_engine.py
│   ├── decision_engine.py
│   │
│   └── evaluation/
│       └── usage_report.md
│
├── dataset/
│   ├── requests.csv
│   ├── profiles.csv
│   ├── events.csv
│   ├── payment_options.csv
│   ├── messages.csv
│   ├── image_links.csv
│   └── exchange_rates.csv
│
├── problem_statement.md
├── AGENTS.md
├── CLAUDE.md
├── chat_transcript.txt
├── .gitignore
└── README.md
```

## ⚙️ How It Works

### 1. Data Processing

The system loads and validates financial datasets, including transactions, user profiles, purchase requests, payment options, messages, and exchange-rate information.

### 2. Financial Forecasting

Historical and scheduled financial events are processed to build a **90-day daily cash-flow forecast** for the relevant user.

The forecast accounts for:

* Income
* Expenses
* Recurring commitments
* Upcoming payments
* Minimum balance requirements
* Transaction timing
* Financial constraints

### 3. Decision Engine

The decision engine evaluates whether the requested purchase can be completed safely.

It considers:

* Current available balance
* Future projected balance
* Purchase amount
* Minimum protected balance
* Payment deadlines
* Partial-payment rules
* Installment options
* User preferences
* Recurring commitments

### 4. Payment Planning

When full payment is not immediately possible, the engine evaluates valid alternatives such as partial payment or installments and generates a structured payment plan.

### 5. Validation

The final output is validated for:

* Required columns
* Request IDs
* Data types
* Valid affordability statuses
* Valid payment methods
* Payment-plan consistency
* Complete 250-request coverage
* Financial edge cases

## 🧮 Numerical Stability

Financial calculations can be affected by floating-point precision.

For monetary boundary comparisons, the project uses a small tolerance to prevent insignificant floating-point differences from incorrectly changing an affordability decision.

For example, a calculated balance such as:

```text
₹83,999.995
```

can represent an expected value of approximately:

```text
₹84,000.00
```

A ₹0.01 tolerance is therefore used when evaluating critical monetary boundaries.

## 🤖 AI-Assisted Development

AI tools were used during development for implementation support, debugging, refactoring, and reasoning about edge cases.

The final application itself uses a **deterministic Python-based financial forecasting and decision engine** and does not require an external LLM API to generate the final decisions.

## 📌 Challenge Context

This project was developed as part of the **HackerRank Orchestrate — Buy or Wait** financial agent challenge.

The original challenge documentation and project instructions are preserved in this repository:

* `problem_statement.md`
* `AGENTS.md`
* `CLAUDE.md`

## 📈 What I Learned

Through this project, I worked on:

* Designing rule-based financial decision systems
* Processing large structured datasets
* Building cash-flow forecasting logic
* Handling recurring financial commitments
* Designing payment-plan algorithms
* Working with incomplete financial data
* Using OCR to recover missing information
* Handling floating-point issues in financial calculations
* Building automated validation pipelines
* Structuring Python projects for reproducible evaluation

## 👨‍💻 Author

**Sanket Patil**

Computer Engineering | Software Engineering & AI/Data Analytics

GitHub: [sanket1145](https://github.com/sanket1145)
