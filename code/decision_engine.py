import pandas as pd
import numpy as np
from datetime import timedelta


# ============================================================
# CONSTANTS
# ============================================================

IGNORED_STATUSES = {
    "cancelled",
    "failed",
}

DIRECTION_CREDIT = "credit"
DIRECTION_DEBIT = "debit"


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_amount(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = (
            value.replace(",", "")
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .strip()
        )

        if value == "":
            return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def normalize_date(value):
    if pd.isna(value):
        return pd.NaT

    return pd.Timestamp(value).normalize()


# ============================================================
# EXCHANGE RATE ENGINE
# ============================================================

class ExchangeRateEngine:

    def __init__(self, exchange_rates):
        self.rates = exchange_rates.copy()

        self.rates["rate_date"] = pd.to_datetime(
            self.rates["rate_date"],
            errors="coerce",
        ).dt.normalize()

        self.rates["rate"] = pd.to_numeric(
            self.rates["rate"],
            errors="coerce",
        )

        self.rates = self.rates.dropna(
            subset=[
                "rate_date",
                "from_currency",
                "to_currency",
                "rate",
            ]
        )

    def get_rate(
        self,
        rate_date,
        from_currency,
        to_currency,
    ):
        """
        Convert from_currency -> to_currency using the supplied
        exchange-rate dataset.

        Exact-date rate is preferred.

        If the reverse pair exists, its inverse is used.

        No external exchange-rate data is used.
        """

        from_currency = str(
            from_currency
        ).upper()

        to_currency = str(
            to_currency
        ).upper()

        rate_date = normalize_date(rate_date)

        if (
            pd.isna(rate_date)
            or from_currency == to_currency
        ):
            return 1.0

        # ----------------------------------------------------
        # Exact-date direct rate
        # ----------------------------------------------------

        direct = self.rates[
            (self.rates["rate_date"] == rate_date)
            & (
                self.rates["from_currency"]
                .astype(str)
                .str.upper()
                == from_currency
            )
            & (
                self.rates["to_currency"]
                .astype(str)
                .str.upper()
                == to_currency
            )
        ]

        if not direct.empty:
            return float(
                direct.iloc[0]["rate"]
            )

        # ----------------------------------------------------
        # Exact-date reverse rate
        # ----------------------------------------------------

        reverse = self.rates[
            (self.rates["rate_date"] == rate_date)
            & (
                self.rates["from_currency"]
                .astype(str)
                .str.upper()
                == to_currency
            )
            & (
                self.rates["to_currency"]
                .astype(str)
                .str.upper()
                == from_currency
            )
        ]

        if not reverse.empty:
            reverse_rate = float(
                reverse.iloc[0]["rate"]
            )

            if reverse_rate != 0:
                return 1.0 / reverse_rate

        # ----------------------------------------------------
        # Nearest-date direct rate
        # ----------------------------------------------------

        candidates = self.rates[
            (
                self.rates["from_currency"]
                .astype(str)
                .str.upper()
                == from_currency
            )
            & (
                self.rates["to_currency"]
                .astype(str)
                .str.upper()
                == to_currency
            )
        ].copy()

        if not candidates.empty:

            candidates["_distance"] = (
                candidates["rate_date"] - rate_date
            ).abs()

            candidates = candidates.sort_values(
                "_distance"
            )

            return float(
                candidates.iloc[0]["rate"]
            )

        # ----------------------------------------------------
        # Nearest-date reverse rate
        # ----------------------------------------------------

        candidates = self.rates[
            (
                self.rates["from_currency"]
                .astype(str)
                .str.upper()
                == to_currency
            )
            & (
                self.rates["to_currency"]
                .astype(str)
                .str.upper()
                == from_currency
            )
        ].copy()

        if not candidates.empty:

            candidates["_distance"] = (
                candidates["rate_date"] - rate_date
            ).abs()

            candidates = candidates.sort_values(
                "_distance"
            )

            reverse_rate = float(
                candidates.iloc[0]["rate"]
            )

            if reverse_rate != 0:
                return 1.0 / reverse_rate

        raise ValueError(
            f"No exchange rate available for "
            f"{from_currency}->{to_currency} "
            f"near {rate_date.date()}"
        )

    def convert(
        self,
        amount,
        rate_date,
        from_currency,
        to_currency,
    ):
        amount = clean_amount(amount)

        if pd.isna(amount):
            return np.nan

        rate = self.get_rate(
            rate_date,
            from_currency,
            to_currency,
        )

        return amount * rate


# ============================================================
# FORECAST NORMALIZATION
# ============================================================

def normalize_forecast(
    forecast,
    home_currency,
    exchange_engine,
):
    """
    Convert all forecast amounts into the user's home currency.
    """

    if forecast is None or forecast.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "user_id",
                "description",
                "category",
                "direction",
                "amount",
                "currency",
                "effective_date",
                "source_event_id",
                "flexibility",
                "minimum_allowed_amount",
                "amount_home_currency",
            ]
        )

    df = forecast.copy()

    df["effective_date"] = pd.to_datetime(
        df["effective_date"],
        errors="coerce",
    ).dt.normalize()

    df["amount"] = df["amount"].apply(
        clean_amount
    )

    converted_amounts = []

    for _, row in df.iterrows():

        amount = row["amount"]

        currency = row.get(
            "currency",
            home_currency,
        )

        if pd.isna(currency):
            currency = home_currency

        converted = exchange_engine.convert(
            amount,
            row["effective_date"],
            currency,
            home_currency,
        )

        converted_amounts.append(
            converted
        )

    df["amount_home_currency"] = (
        converted_amounts
    )

    return df


# ============================================================
# PAYMENT OPTION SCHEDULE
# ============================================================

def build_payment_schedule(
    payment_option,
):
    """
    Convert one supplied payment option into exact
    payment dates and amounts.
    """

    payment_method = str(
        payment_option["payment_method"]
    ).strip().lower()

    payment_amount = clean_amount(
        payment_option["payment_amount"]
    )

    number_of_payments = int(
        payment_option["number_of_payments"]
    )

    first_payment_date = normalize_date(
        payment_option["first_payment_date"]
    )

    frequency = clean_amount(
        payment_option["payment_frequency_days"]
    )

    if pd.isna(frequency):
        frequency = 0

    frequency = int(frequency)

    rows = []

    for i in range(number_of_payments):

        payment_date = (
            first_payment_date
            + timedelta(
                days=i * frequency
            )
        )

        rows.append(
            {
                "payment_option_id":
                    payment_option[
                        "payment_option_id"
                    ],
                "payment_number":
                    i + 1,
                "payment_date":
                    payment_date,
                "amount":
                    payment_amount,
                "payment_method":
                    payment_method,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DEADLINE CHECK
# ============================================================

def schedule_meets_deadline(
    schedule,
    desired_completion_date,
):
    if schedule.empty:
        return False

    deadline = normalize_date(
        desired_completion_date
    )

    final_date = schedule[
        "payment_date"
    ].max()

    return final_date <= deadline


# ============================================================
# PAYMENT METHOD ELIGIBILITY
# ============================================================

def payment_method_allowed(
    payment_method,
    profile,
):
    allowed = str(
        profile.get(
            "payment_methods_user_will_consider",
            "",
        )
    )

    allowed_methods = {
        x.strip().lower()
        for x in allowed.split("|")
        if x.strip()
    }

    return (
        payment_method.lower()
        in allowed_methods
    )


# ============================================================
# INSTALLMENT LIMIT
# ============================================================

def installment_count_allowed(
    schedule,
    profile,
):
    max_months = clean_amount(
        profile.get(
            "max_installment_months",
            np.nan,
        )
    )

    if pd.isna(max_months):
        return True

    number_of_payments = len(
        schedule
    )

    return (
        number_of_payments
        <= max_months
    )


# ============================================================
# CASH-FLOW SIMULATION
# ============================================================

def simulate_with_request(
    forecast,
    starting_balance,
    minimum_balance,
    payment_schedule,
    spending_changes=None,
):
    """
    Simulate the forecast plus the request's payment schedule.

    Returns:
        safe
        minimum_balance
        ending_balance
        failure_date
    """

    df = forecast.copy()

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "effective_date",
                "direction",
                "amount_home_currency",
            ]
        )

    # --------------------------------------------------------
    # Apply spending changes
    # --------------------------------------------------------

    if spending_changes:

        for change in spending_changes:

            parts = str(change).split(":")

            if not parts:
                continue

            action = parts[0].strip().lower()

            # ------------------------------------------------
            # STOP EVENT
            # ------------------------------------------------

            if action == "stop" and len(parts) == 2:

                event_id = parts[1]

                mask = (
                    df["source_event_id"]
                    .astype(str)
                    == str(event_id)
                )

                mask = (
                    mask
                    | (
                        df["event_id"]
                        .astype(str)
                        == str(event_id)
                    )
                )

                df.loc[
                    mask
                    & (
                        df["direction"]
                        .astype(str)
                        .str.lower()
                        == "debit"
                    ),
                    "amount_home_currency",
                ] = 0.0

            # ------------------------------------------------
            # REDUCE EVENT
            # ------------------------------------------------

            elif (
                action == "reduce_to"
                and len(parts) == 3
            ):

                event_id = parts[1]

                try:
                    new_amount = float(
                        parts[2]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    continue

                mask = (
                    df["source_event_id"]
                    .astype(str)
                    == str(event_id)
                )

                mask = (
                    mask
                    | (
                        df["event_id"]
                        .astype(str)
                        == str(event_id)
                    )
                )

                df.loc[
                    mask
                    & (
                        df["direction"]
                        .astype(str)
                        .str.lower()
                        == "debit"
                    ),
                    "amount_home_currency",
                ] = new_amount

    # --------------------------------------------------------
    # Forecast events
    # --------------------------------------------------------

    df["effective_date"] = pd.to_datetime(
        df["effective_date"],
        errors="coerce",
    ).dt.normalize()

    df["amount_home_currency"] = pd.to_numeric(
        df["amount_home_currency"],
        errors="coerce",
    )

    df = df[
        df["effective_date"].notna()
        & df[
            "amount_home_currency"
        ].notna()
    ].copy()

    # --------------------------------------------------------
    # Add request payments
    # --------------------------------------------------------

    payment_rows = []

    if payment_schedule is not None:

        for _, payment in payment_schedule.iterrows():

            payment_rows.append(
                {
                    "effective_date":
                        normalize_date(
                            payment[
                                "payment_date"
                            ]
                        ),

                    "direction":
                        DIRECTION_DEBIT,

                    "amount_home_currency":
                        float(
                            payment["amount"]
                        ),

                    "event_id":
                        "request_payment",

                    "source_event_id":
                        "request_payment",
                }
            )

    if payment_rows:

        request_df = pd.DataFrame(
            payment_rows
        )

        df = pd.concat(
            [
                df,
                request_df,
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "effective_date",
            "event_id",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    balance = float(
        starting_balance
    )

    minimum_observed = balance

    failure_date = None

    for date, day_events in df.groupby(
        "effective_date",
        sort=True,
    ):

        for _, event in day_events.iterrows():

            amount = float(
                event[
                    "amount_home_currency"
                ]
            )

            direction = str(
                event["direction"]
            ).lower()

            if direction == DIRECTION_CREDIT:

                balance += amount

            elif direction == DIRECTION_DEBIT:

                balance -= amount

        minimum_observed = min(
            minimum_observed,
            balance,
        )

        # ----------------------------------------------------
        # Monetary values are effectively 2-decimal values.
        # Avoid false failures caused by floating-point
        # representation such as 83999.995 vs 84000.
        # ----------------------------------------------------

        if balance < (
            minimum_balance - 0.01
        ):

            failure_date = date

            return {
                "safe": False,

                "minimum_balance":
                    minimum_observed,

                "ending_balance":
                    balance,

                "failure_date":
                    failure_date,
            }

    return {
        "safe": True,

        "minimum_balance":
            minimum_observed,

        "ending_balance":
            balance,

        "failure_date":
            None,
    }


# ============================================================
# AMOUNT SAFE TO PAY
# ============================================================

def calculate_amount_safe_to_pay(
    forecast,
    request_date,
    starting_balance,
    minimum_balance,
    requested_amount,
    home_currency,
    exchange_engine,
    spending_changes=None,
):
    """
    Maximum amount that can be paid on request_date
    without optional spending changes.

    Uses binary search because affordability is monotonic
    with respect to the immediate payment amount.
    """

    request_date = normalize_date(
        request_date
    )

    requested_amount = float(
        requested_amount
    )

    if requested_amount <= 0:
        return 0.0

    # --------------------------------------------------------
    # Baseline forecast without request
    # --------------------------------------------------------

    baseline_result = simulate_with_request(
        forecast,
        starting_balance,
        minimum_balance,
        pd.DataFrame(
            columns=[
                "payment_date",
                "amount",
            ]
        ),
        spending_changes=spending_changes,
    )

    if not baseline_result["safe"]:
        return 0.0

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def is_safe(amount):

        schedule = pd.DataFrame(
            [
                {
                    "payment_date":
                        request_date,

                    "amount":
                        amount,
                }
            ]
        )

        result = simulate_with_request(
            forecast,
            starting_balance,
            minimum_balance,
            schedule,
            spending_changes=spending_changes,
        )

        return result["safe"]

    # --------------------------------------------------------
    # Full request already safe
    # --------------------------------------------------------

    if is_safe(requested_amount):
        return requested_amount

    # --------------------------------------------------------
    # Binary search
    # --------------------------------------------------------

    low = 0.0
    high = requested_amount

    for _ in range(60):

        mid = (
            low + high
        ) / 2.0

        if is_safe(mid):
            low = mid
        else:
            high = mid

    return round(
        low,
        2,
    )


# ============================================================
# EARLIEST SAFE DATE
# ============================================================

def find_earliest_safe_date(
    forecast,
    request_date,
    desired_completion_date,
    starting_balance,
    minimum_balance,
    requested_amount,
):
    """
    Find the first date between request_date and the
    desired completion date where a full immediate payment
    is safe without spending changes.
    """

    request_date = normalize_date(
        request_date
    )

    desired_completion_date = normalize_date(
        desired_completion_date
    )

    if desired_completion_date < request_date:
        return None

    date = request_date

    while date <= desired_completion_date:

        schedule = pd.DataFrame(
            [
                {
                    "payment_date":
                        date,

                    "amount":
                        requested_amount,
                }
            ]
        )

        result = simulate_with_request(
            forecast,
            starting_balance,
            minimum_balance,
            schedule,
            spending_changes=None,
        )

        if result["safe"]:
            return date

        date += timedelta(days=1)

    return None


# ============================================================
# EARLIEST SAFE PARTIAL PAYMENT DATE
# ============================================================

def find_earliest_partial_payment_date(
    forecast,
    request_date,
    desired_completion_date,
    starting_balance,
    minimum_balance,
    first_payment_amount,
    remaining_amount,
):
    """
    Find the earliest date on which the remaining amount
    can be paid after paying the safe partial amount today.

    IMPORTANT:
    The simulation includes BOTH payments together.
    """

    request_date = normalize_date(
        request_date
    )

    desired_completion_date = normalize_date(
        desired_completion_date
    )

    date = (
        request_date
        + timedelta(days=1)
    )

    while date <= desired_completion_date:

        schedule = pd.DataFrame(
            [
                {
                    "payment_date":
                        request_date,

                    "amount":
                        first_payment_amount,
                },
                {
                    "payment_date":
                        date,

                    "amount":
                        remaining_amount,
                },
            ]
        )

        result = simulate_with_request(
            forecast,
            starting_balance,
            minimum_balance,
            schedule,
            spending_changes=None,
        )

        if result["safe"]:
            return date

        date += timedelta(days=1)

    return None


# ============================================================
# SPENDING-CHANGE CANDIDATES
# ============================================================

def get_spending_change_candidates(
    forecast,
    profile,
):
    """
    Identify recurring flexible debit events that the user
    explicitly allows to reduce or stop.
    """

    if forecast.empty:
        return []

    reduce_categories = {
        x.strip().lower()
        for x in str(
            profile.get(
                "expense_categories_user_is_willing_to_reduce",
                "",
            )
        ).split("|")
        if x.strip()
    }

    # FIXED:
    # Actual profile column is:
    # expense_categories_user_is_willing_to_stop
    stop_categories = {
        x.strip().lower()
        for x in str(
            profile.get(
                "expense_categories_user_is_willing_to_stop",
                "",
            )
        ).split("|")
        if x.strip()
    }

    candidates = []

    for _, row in forecast.iterrows():

        direction = str(
            row.get(
                "direction",
                "",
            )
        ).lower()

        if direction != DIRECTION_DEBIT:
            continue

        category = str(
            row.get(
                "category",
                "",
            )
        ).lower()

        flexibility = str(
            row.get(
                "flexibility",
                "",
            )
        ).lower()

        amount = clean_amount(
            row.get(
                "amount_home_currency",
                np.nan,
            )
        )

        if pd.isna(amount):
            continue

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if (
            category in stop_categories
            and flexibility
            not in {"", "fixed"}
        ):

            candidates.append(
                {
                    "event_id":
                        row.get(
                            "source_event_id",
                            row.get(
                                "event_id"
                            ),
                        ),

                    "category":
                        category,

                    "action":
                        "stop",

                    "amount":
                        float(amount),
                }
            )

        # ----------------------------------------------------
        # REDUCE
        # ----------------------------------------------------

        elif (
            category in reduce_categories
            and flexibility
            not in {"", "fixed"}
        ):

            minimum = clean_amount(
                row.get(
                    "minimum_allowed_amount",
                    np.nan,
                )
            )

            current = float(
                amount
            )

            if pd.notna(minimum):

                new_amount = min(
                    current,
                    float(minimum),
                )

            else:

                new_amount = (
                    current * 0.5
                )

            candidates.append(
                {
                    "event_id":
                        row.get(
                            "source_event_id",
                            row.get(
                                "event_id"
                            ),
                        ),

                    "category":
                        category,

                    "action":
                        "reduce_to",

                    "amount":
                        current,

                    "new_amount":
                        new_amount,
                }
            )

    return candidates


# ============================================================
# CHANGE FORMATTER
# ============================================================

def format_spending_change(
    candidate,
):
    if candidate["action"] == "stop":

        return (
            f"stop:{candidate['event_id']}"
        )

    return (
        f"reduce_to:"
        f"{candidate['event_id']}:"
        f"{candidate['new_amount']:.2f}"
    )


# ============================================================
# FIND USEFUL SPENDING CHANGES
# ============================================================

def find_spending_changes(
    forecast,
    request_date,
    starting_balance,
    minimum_balance,
    requested_amount,
    profile,
):
    """
    Search permitted spending changes.

    Maximum three changes are allowed.

    The search starts with one change, then combinations
    of two and three changes.

    The request payment is correctly tested on request_date.
    """

    candidates = get_spending_change_candidates(
        forecast,
        profile,
    )

    if not candidates:
        return None

    # --------------------------------------------------------
    # Keep only unique event/action candidates.
    # --------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = (
            candidate["event_id"],
            candidate["action"],
        )

        unique[key] = candidate

    candidates = list(
        unique.values()
    )

    # --------------------------------------------------------
    # Test combinations
    # --------------------------------------------------------

    from itertools import combinations

    request_date = normalize_date(
        request_date
    )

    for count in range(
        1,
        min(
            3,
            len(candidates)
        ) + 1,
    ):

        for combo in combinations(
            candidates,
            count,
        ):

            changes = [
                format_spending_change(
                    candidate
                )
                for candidate in combo
            ]

            schedule = pd.DataFrame(
                [
                    {
                        "payment_date":
                            request_date,

                        "amount":
                            requested_amount,
                    }
                ]
            )

            result = simulate_with_request(
                forecast,
                starting_balance,
                minimum_balance,
                schedule,
                spending_changes=changes,
            )

            if result["safe"]:
                return changes

    return None


# ============================================================
# SINGLE REQUEST EVALUATION
# ============================================================

def evaluate_request(
    request,
    profile,
    forecast,
    payment_options,
    exchange_engine,
):
    """
    Evaluate one request and return the required output fields.
    """

    request_id = request["request_id"]

    request_date = normalize_date(
        request["request_date"]
    )

    desired_completion_date = normalize_date(
        request["desired_completion_date"]
    )

    requested_amount = float(
        request["requested_amount"]
    )

    starting_balance = float(
        profile["current_available_balance"]
    )

    minimum_balance = float(
        profile["minimum_balance_to_keep"]
    )

    home_currency = str(
        profile["home_currency"]
    ).upper()

    allows_partial = str(
        request["allows_partial_payment"]
    ).lower() in {
        "true",
        "1",
        "yes",
    }

    # ========================================================
    # NORMALIZE FORECAST
    # ========================================================

    forecast_home = normalize_forecast(
        forecast,
        home_currency,
        exchange_engine,
    )

    # ========================================================
    # AMOUNT SAFE TO PAY TODAY
    # ========================================================

    amount_safe = calculate_amount_safe_to_pay(
        forecast_home,
        request_date,
        starting_balance,
        minimum_balance,
        requested_amount,
        home_currency,
        exchange_engine,
    )

    amount_safe = min(
        requested_amount,
        max(
            0.0,
            amount_safe
        ),
    )

    # ========================================================
    # EARLIEST DATE FULL AMOUNT IS SAFE
    # ========================================================

    earliest_date = find_earliest_safe_date(
        forecast_home,
        request_date,
        desired_completion_date,
        starting_balance,
        minimum_balance,
        requested_amount,
    )

    # ========================================================
    # REQUEST-SPECIFIC PAYMENT OPTIONS
    # ========================================================

    request_options = payment_options[
        payment_options["request_id"].astype(str)
        == str(request_id)
    ].copy()

    # ========================================================
    # EVALUATE SUPPLIED PAYMENT OPTIONS
    # ========================================================

    valid_options = []

    for _, option in request_options.iterrows():

        method = str(
            option["payment_method"]
        ).lower()

        # ----------------------------------------------------
        # Payment method must be allowed by profile
        # ----------------------------------------------------

        if not payment_method_allowed(
            method,
            profile,
        ):
            continue

        schedule = build_payment_schedule(
            option
        )

        if schedule.empty:
            continue

        # ----------------------------------------------------
        # Must finish by deadline
        # ----------------------------------------------------

        if not schedule_meets_deadline(
            schedule,
            desired_completion_date,
        ):
            continue

        # ----------------------------------------------------
        # Installment limit
        # ----------------------------------------------------

        if (
            method == "installments"
            and not installment_count_allowed(
                schedule,
                profile,
            )
        ):
            continue

        # ----------------------------------------------------
        # Simulate supplied option
        # ----------------------------------------------------

        result = simulate_with_request(
            forecast_home,
            starting_balance,
            minimum_balance,
            schedule,
            spending_changes=None,
        )

        if result["safe"]:

            valid_options.append(
                {
                    "payment_option_id":
                        option["payment_option_id"],

                    "payment_method":
                        method,

                    "schedule":
                        schedule,

                    "total_payable":
                        float(
                            option[
                                "total_payable_amount"
                            ]
                        ),

                    "number_of_payments":
                        len(schedule),
                }
            )

    # ========================================================
    # SELECT BEST SUPPLIED OPTION
    # ========================================================

    best_option = None

    if valid_options:

        def option_sort_key(option):

            schedule = option[
                "schedule"
            ]

            first_date = schedule[
                "payment_date"
            ].min()

            return (
                option["total_payable"],
                first_date,
                option["number_of_payments"],
                str(
                    option[
                        "payment_option_id"
                    ]
                ),
            )

        valid_options.sort(
            key=option_sort_key
        )

        best_option = valid_options[0]

    # ========================================================
    # SPENDING CHANGES
    # ========================================================

    spending_changes = None

    if best_option is None:

        spending_changes = find_spending_changes(
            forecast_home,
            request_date,
            starting_balance,
            minimum_balance,
            requested_amount,
            profile,
        )

        if spending_changes:

            changed_options = []

            for _, option in request_options.iterrows():

                method = str(
                    option["payment_method"]
                ).lower()

                if not payment_method_allowed(
                    method,
                    profile,
                ):
                    continue

                schedule = build_payment_schedule(
                    option
                )

                if schedule.empty:
                    continue

                if not schedule_meets_deadline(
                    schedule,
                    desired_completion_date,
                ):
                    continue

                if (
                    method == "installments"
                    and not installment_count_allowed(
                        schedule,
                        profile,
                    )
                ):
                    continue

                result = simulate_with_request(
                    forecast_home,
                    starting_balance,
                    minimum_balance,
                    schedule,
                    spending_changes=spending_changes,
                )

                if result["safe"]:

                    changed_options.append(
                        {
                            "payment_option_id":
                                option[
                                    "payment_option_id"
                                ],

                            "payment_method":
                                method,

                            "schedule":
                                schedule,

                            "total_payable":
                                float(
                                    option[
                                        "total_payable_amount"
                                    ]
                                ),

                            "number_of_payments":
                                len(schedule),
                        }
                    )

            if changed_options:

                changed_options.sort(
                    key=lambda x: (
                        x["total_payable"],
                        x["schedule"][
                            "payment_date"
                        ].min(),
                        x["number_of_payments"],
                        str(
                            x[
                                "payment_option_id"
                            ]
                        ),
                    )
                )

                best_option = changed_options[0]

    # ========================================================
    # PARTIAL PAYMENT CANDIDATE
    # ========================================================
    #
    # Partial payment must be evaluated BEFORE returning
    # best_option.
    #
    # Requirements:
    #   - request allows partial payment
    #   - profile allows partial_payment
    #   - some amount is safe today
    #   - remaining amount > 0
    #   - remaining amount can be paid later
    #   - completion is within deadline
    #
    # The partial plan consists of EXACTLY two payments:
    #
    #   today       -> safe amount
    #   later date  -> remaining amount
    #
    # It competes with supplied installment options.
    # ========================================================

    partial_allowed_by_profile = (
        payment_method_allowed(
            "partial_payment",
            profile,
        )
    )

    partial_remaining_amount = round(
        requested_amount - amount_safe,
        2,
    )

    partial_remaining_date = None

    if partial_remaining_amount > 0:

        partial_remaining_date = (
            find_earliest_partial_payment_date(
                forecast_home,
                request_date,
                desired_completion_date,
                starting_balance,
                minimum_balance,
                amount_safe,
                partial_remaining_amount,
            )
        )

    if (
        allows_partial
        and partial_allowed_by_profile
        and amount_safe > 0
        and amount_safe < requested_amount
        and partial_remaining_date is not None
        and partial_remaining_date > request_date
        and partial_remaining_date <= desired_completion_date
    ):

        partial_schedule = pd.DataFrame(
            [
                {
                    "payment_date":
                        request_date,

                    "amount":
                        amount_safe,
                },

                {
                    "payment_date":
                        partial_remaining_date,

                    "amount":
                        partial_remaining_amount,
                },
            ]
        )

        partial_candidate = {
            "payment_option_id":
                "partial_payment",

            "payment_method":
                "partial_payment",

            "schedule":
                partial_schedule,

            "total_payable":
                requested_amount,

            "number_of_payments":
                2,
        }

        # ----------------------------------------------------
        # Compare partial plan with existing best option
        # ----------------------------------------------------

        if best_option is None:

            best_option = partial_candidate

            # Partial payment does not require spending changes.
            spending_changes = None

        else:

            existing_schedule = (
                best_option["schedule"]
            )

            existing_key = (
                best_option["total_payable"],

                existing_schedule[
                    "payment_date"
                ].min(),

                best_option[
                    "number_of_payments"
                ],

                str(
                    best_option[
                        "payment_option_id"
                    ]
                ),
            )

            partial_key = (
                partial_candidate[
                    "total_payable"
                ],

                partial_candidate[
                    "schedule"
                ]["payment_date"].min(),

                partial_candidate[
                    "number_of_payments"
                ],

                "partial_payment",
            )

            if partial_key < existing_key:

                best_option = (
                    partial_candidate
                )

                spending_changes = None

    # ========================================================
    # IF A VALID PLAN EXISTS, RETURN IT
    # ========================================================

    if best_option is not None:

        method = best_option[
            "payment_method"
        ]

        schedule = best_option[
            "schedule"
        ]

        first_date = schedule[
            "payment_date"
        ].min()

        last_date = schedule[
            "payment_date"
        ].max()

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            method == "full_payment"
            and first_date == request_date
        ):

            status = (
                "affordable_now"
            )

        else:

            status = (
                "affordable_with_plan"
            )

        # ----------------------------------------------------
        # Payment plan
        # ----------------------------------------------------

        payment_plan = "|".join(
            [
                (
                    f"{row['payment_date'].strftime('%Y-%m-%d')}:"
                    f"{row['amount']:.2f}"
                )

                for _, row
                in schedule.iterrows()
            ]
        )

        # ----------------------------------------------------
        # Spending changes
        # ----------------------------------------------------

        changes_text = (
            "|".join(
                spending_changes
            )
            if spending_changes
            else "none"
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        if method == "partial_payment":

            explanation = (
                f"{amount_safe:.2f} "
                f"{home_currency} is safe to pay "
                f"on the request date while maintaining "
                f"the required minimum balance. "
                f"The remaining "
                f"{partial_remaining_amount:.2f} "
                f"{home_currency} can be paid on "
                f"{last_date.strftime('%Y-%m-%d')}, "
                f"completing the full request by the deadline."
            )

        elif spending_changes:

            explanation = (
                f"The request can be completed safely "
                f"using the recommended payment plan "
                f"with permitted spending changes while "
                f"maintaining the required minimum balance."
            )

        else:

            explanation = (
                f"Full request of "
                f"{requested_amount:.2f} "
                f"{home_currency} can be completed "
                f"safely by "
                f"{last_date.strftime('%Y-%m-%d')} "
                f"while maintaining the required "
                f"minimum balance."
            )

        return {
            "request_id":
                request_id,

            "amount_safe_to_pay":
                round(
                    amount_safe,
                    2,
                ),

            "affordability_status":
                status,

            "recommended_payment_method":
                method,

            "payment_plan":
                payment_plan,

            "earliest_date_for_full_payment":
                last_date.strftime(
                    "%Y-%m-%d"
                ),

            "spending_changes_needed":
                changes_text,

            "decision_explanation":
                explanation,
        }

    # ========================================================
    # AFFORDABLE LATER
    #
    # Only recommend wait when:
    #   - no valid payment plan exists
    #   - full_payment is allowed
    #   - full request becomes safe before deadline
    # ========================================================

    full_payment_allowed = payment_method_allowed(
        "full_payment",
        profile,
    )

    if (
        earliest_date is not None
        and earliest_date <= desired_completion_date
        and full_payment_allowed
    ):

        payment_plan = (
            f"{earliest_date.strftime('%Y-%m-%d')}:"
            f"{requested_amount:.2f}"
        )

        return {
            "request_id":
                request_id,

            "amount_safe_to_pay":
                round(
                    amount_safe,
                    2,
                ),

            "affordability_status":
                "affordable_later",

            "recommended_payment_method":
                "wait",

            "payment_plan":
                payment_plan,

            "earliest_date_for_full_payment":
                earliest_date.strftime(
                    "%Y-%m-%d"
                ),

            "spending_changes_needed":
                "none",

            "decision_explanation":
                (
                    f"The full amount is not safe "
                    f"on the request date, but the "
                    f"request becomes affordable on "
                    f"{earliest_date.strftime('%Y-%m-%d')}."
                ),
        }

    # ========================================================
    # NOT AFFORDABLE
    # ========================================================

    return {
        "request_id":
            request_id,

        "amount_safe_to_pay":
            round(
                amount_safe,
                2,
            ),

        "affordability_status":
            "not_affordable",

        "recommended_payment_method":
            "not_recommended",

        "payment_plan":
            "none",

        "earliest_date_for_full_payment":
            (
                earliest_date.strftime(
                    "%Y-%m-%d"
                )
                if earliest_date is not None
                else ""
            ),

        "spending_changes_needed":
            "none",

        "decision_explanation":
            (
                f"The requested amount of "
                f"{requested_amount:.2f} "
                f"{home_currency} cannot be completed "
                f"safely within the allowed forecast "
                f"and payment constraints."
            ),
    }