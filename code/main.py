from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

# Make code/ imports reliable when running:
# python code\main.py
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ============================================================
# IMPORT ENGINES
# ============================================================

from financial_engine import build_forecast
from decision_engine import evaluate_request, ExchangeRateEngine


# ============================================================
# DATA FILES
# ============================================================

DATA_FILES = {
    "requests": DATASET / "requests.csv",
    "profiles": DATASET / "financial_profiles.csv",
    "events": DATASET / "financial_events.csv",
    "exchange_rates": DATASET / "exchange_rates.csv",
    "payment_options": DATASET / "request_payment_options.csv",
    "messages": DATASET / "messages.csv",
    "images": DATASET / "images.csv",
    "output_template": DATASET / "output.csv",
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "requests": [
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    ],
    "profiles": [
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    ],
    "events": [
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    ],
    "exchange_rates": [
        "rate_date",
        "from_currency",
        "to_currency",
        "rate",
    ],
    "payment_options": [
        "payment_option_id",
        "request_id",
        "payment_method",
        "number_of_payments",
        "payment_amount",
    ],
    "messages": [
        "message_id",
        "user_id",
        "request_id",
        "related_event_id",
        "sent_at",
        "source_type",
        "message_text",
    ],
    "images": [
        "image_id",
        "user_id",
        "request_id",
        "related_event_id",
    ],
    "output_template": [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ],
}


# ============================================================
# OCR RECONSTRUCTION
# ============================================================

OCR_AMOUNTS = {
    "event_253": 4365000.00,
    "event_1442": 100000.00,
    "event_1545": 41272.00,
    "event_1700": 854.00,
    "event_1786": 704.05,
    "event_3051": 1995.00,
    "event_3231": 8528.10,
    "event_4535": 15339.00,
    "event_5170": 723.00,
    "event_6033": 79679.26,
    "event_6859": 3650.00,
    "event_7307": 33.50,
    "event_7941": 2298.00,
    "event_9806": 9968.00,
    "event_10521": 393.22,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    data = {}

    for name, path in DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

        data[name] = pd.read_csv(path)

    return data


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(data):
    for name, required in REQUIRED_COLUMNS.items():
        actual = set(data[name].columns)
        missing = set(required) - actual

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )


# ============================================================
# PREPARE DATES
# ============================================================

def prepare_dates(data):
    date_columns = {
        "requests": [
            "request_date",
            "desired_completion_date",
        ],
        "profiles": [],
        "events": [
            "event_date",
            "settlement_date",
        ],
        "exchange_rates": [
            "rate_date",
        ],
        "payment_options": [],
        "messages": [
            "sent_at",
        ],
        "images": [],
        "output_template": [],
    }

    for dataset_name, columns in date_columns.items():
        for column in columns:
            data[dataset_name][column] = pd.to_datetime(
                data[dataset_name][column],
                errors="coerce",
            )


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_data(data):
    requests = data["requests"]
    profiles = data["profiles"]
    events = data["events"]
    messages = data["messages"]
    images = data["images"]
    output_template = data["output_template"]

    if requests["request_id"].duplicated().any():
        raise ValueError(
            "Duplicate request_id found in requests.csv"
        )

    if requests["user_id"].isna().any():
        raise ValueError(
            "Missing user_id in requests.csv"
        )

    if len(requests) != 250:
        raise ValueError(
            f"Expected 250 evaluation requests, "
            f"found {len(requests)}"
        )

    if profiles["user_id"].duplicated().any():
        raise ValueError(
            "Duplicate user_id found in financial_profiles.csv"
        )

    missing_profiles = (
        set(requests["user_id"])
        - set(profiles["user_id"])
    )

    if missing_profiles:
        raise ValueError(
            f"Missing financial profiles for users: "
            f"{missing_profiles}"
        )

    invalid_event_users = (
        set(events["user_id"])
        - set(profiles["user_id"])
    )

    if invalid_event_users:
        raise ValueError(
            f"Financial events reference unknown users: "
            f"{invalid_event_users}"
        )

    invalid_message_users = (
        set(messages["user_id"])
        - set(profiles["user_id"])
    )

    if invalid_message_users:
        raise ValueError(
            f"Messages reference unknown users: "
            f"{invalid_message_users}"
        )

    invalid_image_users = (
        set(images["user_id"])
        - set(profiles["user_id"])
    )

    if invalid_image_users:
        raise ValueError(
            f"Images reference unknown users: "
            f"{invalid_image_users}"
        )

    if set(output_template["request_id"]) != set(
        requests["request_id"]
    ):
        raise ValueError(
            "output.csv does not match evaluation request IDs"
        )


# ============================================================
# RECONSTRUCT MISSING AMOUNTS
# ============================================================

def reconstruct_missing_amounts(financial_events):
    events = financial_events.copy()

    for event_id, amount in OCR_AMOUNTS.items():
        mask = events["event_id"] == event_id

        if mask.any():
            events.loc[mask, "amount"] = amount

    print(
        f"Reconstructed OCR amounts : "
        f"{len(OCR_AMOUNTS)}"
    )

    print(
        f"Remaining missing amounts : "
        f"{events['amount'].isna().sum()}"
    )

    return events


# ============================================================
# PRINT DATA SUMMARY
# ============================================================

def print_summary(data):
    print()
    print("=" * 60)
    print("BUY OR WAIT? - DATA LOADER")
    print("=" * 60)

    print(
        f"requests            : "
        f"{len(data['requests']):,} rows"
    )

    print(
        f"profiles            : "
        f"{len(data['profiles']):,} rows"
    )

    print(
        f"events              : "
        f"{len(data['events']):,} rows"
    )

    print(
        f"exchange_rates      : "
        f"{len(data['exchange_rates']):,} rows"
    )

    print(
        f"payment_options     : "
        f"{len(data['payment_options']):,} rows"
    )

    print(
        f"messages            : "
        f"{len(data['messages']):,} rows"
    )

    print(
        f"images              : "
        f"{len(data['images']):,} rows"
    )

    print(
        f"output_template     : "
        f"{len(data['output_template']):,} rows"
    )

    print("-" * 60)

    print(
        f"Evaluation requests : "
        f"{data['requests']['request_id'].nunique():,}"
    )

    print(
        f"Unique users        : "
        f"{data['requests']['user_id'].nunique():,}"
    )

    print(
        f"Financial events    : "
        f"{len(data['events']):,}"
    )

    print(
        f"Payment options     : "
        f"{len(data['payment_options']):,}"
    )

    print("=" * 60)


# ============================================================
# NORMALIZE OUTPUT ROW
# ============================================================

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def normalize_output_row(request_id, result):
    """
    Convert the decision-engine result into exactly the
    submission schema required by output.csv.
    """

    row = {
        column: result.get(column, "")
        for column in OUTPUT_COLUMNS
    }

    row["request_id"] = request_id

    # Numeric formatting.
    try:
        row["amount_safe_to_pay"] = round(
            float(row["amount_safe_to_pay"]),
            2,
        )
    except (TypeError, ValueError):
        row["amount_safe_to_pay"] = 0.0

    # Convert timestamps/dates to YYYY-MM-DD.
    value = row["earliest_date_for_full_payment"]

    if pd.notna(value) and value != "":
        try:
            row["earliest_date_for_full_payment"] = (
                pd.Timestamp(value).strftime("%Y-%m-%d")
            )
        except Exception:
            row["earliest_date_for_full_payment"] = str(value)

    else:
        row["earliest_date_for_full_payment"] = ""

    # Convert None/NaN to empty strings.
    for column in OUTPUT_COLUMNS:
        if column == "amount_safe_to_pay":
            continue

        value = row[column]

        if value is None:
            row[column] = ""
        elif isinstance(value, float) and pd.isna(value):
            row[column] = ""
        else:
            row[column] = str(value)

    return row


# ============================================================
# EVALUATE ALL REQUESTS
# ============================================================

def evaluate_all_requests(data):
    requests = data["requests"]
    profiles = data["profiles"]
    events = data["events"]
    exchange_rates = data["exchange_rates"]
    exchange_engine = ExchangeRateEngine(exchange_rates)
    payment_options = data["payment_options"]
    messages = data["messages"]
    images = data["images"]

    results = []

    total = len(requests)

    print()
    print("=" * 60)
    print("BUY OR WAIT? - DECISION ENGINE")
    print("=" * 60)
    print(f"Evaluating {total} requests...")
    print()

    for index, (_, request) in enumerate(
        requests.iterrows(),
        start=1,
    ):
        request_id = request["request_id"]
        user_id = request["user_id"]

        profile_match = profiles[
            profiles["user_id"] == user_id
        ]

        if profile_match.empty:
            raise ValueError(
                f"No profile found for {user_id}"
            )

        profile = profile_match.iloc[0]

        user_events = events[
            events["user_id"] == user_id
        ].copy()

        user_messages = messages[
            messages["user_id"] == user_id
        ].copy()

        user_images = images[
            images["user_id"] == user_id
        ].copy()

        request_options = payment_options[
            payment_options["request_id"] == request_id
        ].copy()

        # ----------------------------------------------------
        # Build the 90-day financial forecast.
        # ----------------------------------------------------

        forecast = build_forecast(
        user_events,
        user_id,
        request["request_date"],
        90,
        )
        # ----------------------------------------------------
        # Evaluate request.
        #
        # The decision engine handles:
        # - full payment
        # - installment options
        # - partial payment
        # - spending changes
        # - affordability timing
        # - messages/images context
        # ----------------------------------------------------

        result = evaluate_request(
    request,
    profile,
    forecast,
    request_options,
    exchange_engine,
)

        if result is None:
            raise ValueError(
                f"Decision engine returned None for "
                f"{request_id}"
            )

        result = normalize_output_row(
            request_id,
            result,
        )

        results.append(result)

        # Progress indicator.
        if index % 25 == 0 or index == total:
            print(
                f"Evaluated {index}/{total} requests"
            )

    print()
    print("All requests evaluated successfully.")

    return pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )


# ============================================================
# VALIDATE FINAL OUTPUT
# ============================================================

def validate_final_output(output, requests):
    print()
    print("=" * 60)
    print("VALIDATING FINAL OUTPUT")
    print("=" * 60)

    # Exact row count.
    if len(output) != 250:
        raise ValueError(
            f"Final output must contain 250 rows, "
            f"found {len(output)}"
        )

    # Exact columns and order.
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Final output columns do not match required schema"
        )

    # Unique request IDs.
    if output["request_id"].duplicated().any():
        raise ValueError(
            "Duplicate request_id found in final output"
        )

    # Same request IDs as evaluation requests.
    expected_ids = set(requests["request_id"])
    actual_ids = set(output["request_id"])

    if expected_ids != actual_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        raise ValueError(
            f"Final output request IDs do not match.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    # Allowed statuses.
    allowed_statuses = {
        "affordable_now",
        "affordable_with_plan",
        "affordable_later",
        "not_affordable",
    }

    invalid_statuses = set(
        output["affordability_status"]
    ) - allowed_statuses

    if invalid_statuses:
        raise ValueError(
            f"Invalid affordability_status values: "
            f"{invalid_statuses}"
        )

    # Allowed payment methods.
    allowed_methods = {
        "full_payment",
        "partial_payment",
        "installments",
        "wait",
        "not_recommended",
    }

    invalid_methods = set(
        output["recommended_payment_method"]
    ) - allowed_methods

    if invalid_methods:
        raise ValueError(
            f"Invalid recommended_payment_method values: "
            f"{invalid_methods}"
        )

    # Safe-to-pay must be numeric.
    numeric_safe = pd.to_numeric(
        output["amount_safe_to_pay"],
        errors="coerce",
    )

    if numeric_safe.isna().any():
        bad_rows = output.loc[
            numeric_safe.isna(),
            "request_id",
        ].tolist()

        raise ValueError(
            f"Non-numeric amount_safe_to_pay "
            f"for requests: {bad_rows}"
        )

    # Safe-to-pay cannot be negative.
    if (numeric_safe < -0.01).any():
        raise ValueError(
            "Negative amount_safe_to_pay detected"
        )

    print("Rows                  : 250")
    print("Columns               : 8")
    print("Unique request IDs    : 250")
    print("Statuses              :")
    print(
        output["affordability_status"]
        .value_counts()
        .to_string()
    )
    print()
    print("Payment methods       :")
    print(
        output["recommended_payment_method"]
        .value_counts()
        .to_string()
    )

    print()
    print("Final output validation passed.")
    print("=" * 60)


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(output):
    output_path = ROOT / "output.csv"

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("OUTPUT FILE CREATED")
    print("=" * 60)
    print(f"Path : {output_path}")
    print(f"Rows : {len(output)}")
    print("=" * 60)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading datasets...")

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # 2. Validate structure
    # --------------------------------------------------------

    validate_columns(data)

    # --------------------------------------------------------
    # 3. Prepare dates
    # --------------------------------------------------------

    prepare_dates(data)

    # --------------------------------------------------------
    # 4. Validate relationships
    # --------------------------------------------------------

    validate_data(data)

    # --------------------------------------------------------
    # 5. Reconstruct image/OCR amounts
    # --------------------------------------------------------

    data["events"] = reconstruct_missing_amounts(
        data["events"]
    )

    # --------------------------------------------------------
    # 6. Print loader summary
    # --------------------------------------------------------

    print_summary(data)

    print()
    print(
        "All datasets loaded and validated successfully."
    )
    print("=" * 60)

    # --------------------------------------------------------
    # 7. Evaluate all 250 requests
    # --------------------------------------------------------

    output = evaluate_all_requests(data)

    # --------------------------------------------------------
    # 8. Validate final submission
    # --------------------------------------------------------

    validate_final_output(
        output,
        data["requests"],
    )

    # --------------------------------------------------------
    # 9. Save root output.csv
    # --------------------------------------------------------

    output_path = save_output(output)

    # --------------------------------------------------------
    # 10. Show a few important rows
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAMPLE FINAL DECISIONS")
    print("=" * 60)

    print(
        output[
            [
                "request_id",
                "amount_safe_to_pay",
                "affordability_status",
                "recommended_payment_method",
                "earliest_date_for_full_payment",
            ]
        ].head(10).to_string(index=False)
    )

    # Request 46 sanity check.
    if "request_46" in set(output["request_id"]):
        print()
        print("REQUEST 46 SANITY CHECK")
        print("-" * 60)

        row_46 = output[
            output["request_id"] == "request_46"
        ].iloc[0]

        print(
            row_46[
                OUTPUT_COLUMNS
            ].to_string()
        )

    print()
    print("=" * 60)
    print("SUCCESS - OUTPUT READY")
    print("=" * 60)
    print(f"Final file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()