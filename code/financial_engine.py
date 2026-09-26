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


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_amount(value):
    """
    Convert an amount to numeric.
    Blank/invalid values remain NaN.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = (
            value.replace(",", "")
            .replace("Γé╣", "")
            .replace("$", "")
            .replace("Γé¼", "")
            .strip()
        )

        if value == "":
            return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _event_date(row):
    """
    Determine the effective financial date.

    Settlement date takes precedence when available.
    Otherwise event date is used.
    """

    settlement = row["settlement_date"]
    event_date = row["event_date"]

    if pd.notna(settlement):
        return pd.Timestamp(settlement).normalize()

    if pd.notna(event_date):
        return pd.Timestamp(event_date).normalize()

    return pd.NaT


# ============================================================
# EVENT PREPARATION
# ============================================================

def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare financial events.

    Adds:
        effective_date
    """

    df = events.copy()

    # --------------------------------------------------------
    # Clean amount
    # --------------------------------------------------------

    if "amount" in df.columns:
        df["amount"] = df["amount"].apply(
            _clean_amount
        )

    # --------------------------------------------------------
    # Normalize status
    # --------------------------------------------------------

    if "status" in df.columns:
        df["status"] = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        df["status"] = ""

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(
            df["event_date"],
            errors="coerce",
        )

    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(
            df["settlement_date"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Effective date
    # --------------------------------------------------------

    df["effective_date"] = df.apply(
        _event_date,
        axis=1,
    )

    return df


# ============================================================
# RECURRING EVENT DETECTION
# ============================================================

def historical_recurring_events(
    events: pd.DataFrame,
    user_id: str,
    request_date,
) -> pd.DataFrame:
    """
    Identify recurring historical income/expense patterns.

    A recurring pattern requires at least two historical
    occurrences and a reasonably consistent interval.
    """

    request_date = pd.Timestamp(
        request_date
    ).normalize()

    user_events = events[
        (events["user_id"] == user_id)
        & (events["effective_date"].notna())
        & (events["effective_date"] < request_date)
        & (~events["status"].isin(IGNORED_STATUSES))
        & (events["amount"].notna())
    ].copy()

    if user_events.empty:
        return pd.DataFrame()

    group_columns = [
        "event_type",
        "description",
        "category",
        "direction",
    ]

    recurring_rows = []

    for group_key, group in user_events.groupby(
        group_columns,
        dropna=False,
    ):
        group = group.sort_values(
            "effective_date"
        ).copy()

        if len(group) < 2:
            continue

        dates = (
            group["effective_date"]
            .dropna()
            .sort_values()
        )

        if len(dates) < 2:
            continue

        gaps = (
            dates.diff()
            .dt.days
            .dropna()
        )

        if gaps.empty:
            continue

        median_gap = float(
            gaps.median()
        )

        valid_interval = (
            5 <= median_gap <= 10
            or 25 <= median_gap <= 35
            or 50 <= median_gap <= 65
            or 80 <= median_gap <= 100
            or 350 <= median_gap <= 380
        )

        if not valid_interval:
            continue

        gap_deviation = np.abs(
            gaps - median_gap
        )

        if len(gap_deviation) >= 2:
            if gap_deviation.median() > max(
                3,
                median_gap * 0.20,
            ):
                continue

        recent = group.tail(
            min(3, len(group))
        )

        estimated_amount = float(
            recent["amount"].median()
        )

        last_row = group.iloc[-1]

        recurring_rows.append(
            {
                "event_type": last_row[
                    "event_type"
                ],
                "description": last_row[
                    "description"
                ],
                "category": last_row[
                    "category"
                ],
                "direction": last_row[
                    "direction"
                ],
                "median_gap": median_gap,
                "estimated_amount": estimated_amount,
                "last_date": last_row[
                    "effective_date"
                ],
                "source_event_id": last_row[
                    "event_id"
                ],
                "currency": last_row.get(
                    "currency",
                    None,
                ),
                "flexibility": last_row.get(
                    "flexibility",
                    None,
                ),
                "minimum_allowed_amount": last_row.get(
                    "minimum_allowed_amount",
                    np.nan,
                ),
            }
        )

    if not recurring_rows:
        return pd.DataFrame()

    return pd.DataFrame(
        recurring_rows
    )


# ============================================================
# PROJECT RECURRING EVENTS
# ============================================================

def project_recurring_events(
    events: pd.DataFrame,
    request_date,
    forecast_end,
) -> pd.DataFrame:
    """
    Project recurring historical events into the forecast window.
    """

    request_date = pd.Timestamp(
        request_date
    ).normalize()

    forecast_end = pd.Timestamp(
        forecast_end
    ).normalize()

    if "user_id" not in events.columns:
        return pd.DataFrame()

    user_ids = (
        events["user_id"]
        .dropna()
        .unique()
    )

    projected_rows = []

    for user_id in user_ids:

        recurring = historical_recurring_events(
            events,
            user_id,
            request_date,
        )

        if recurring.empty:
            continue

        for _, pattern in recurring.iterrows():

            gap = int(
                round(
                    pattern[
                        "median_gap"
                    ]
                )
            )

            if gap <= 0:
                continue

            next_date = (
                pd.Timestamp(
                    pattern["last_date"]
                )
                + timedelta(days=gap)
            )

            while next_date < request_date:
                next_date += timedelta(
                    days=gap
                )

            while next_date <= forecast_end:

                projected_rows.append(
                    {
                        "event_id": (
                            f"projected_"
                            f"{pattern['source_event_id']}_"
                            f"{next_date.strftime('%Y%m%d')}"
                        ),
                        "user_id": user_id,
                        "event_type": pattern[
                            "event_type"
                        ],
                        "description": pattern[
                            "description"
                        ],
                        "category": pattern[
                            "category"
                        ],
                        "direction": pattern[
                            "direction"
                        ],
                        "amount": pattern[
                            "estimated_amount"
                        ],
                        "currency": pattern[
                            "currency"
                        ],
                        "event_date": next_date,
                        "settlement_date": next_date,
                        "status": "projected",
                        "linked_event_id": None,
                        "flexibility": pattern[
                            "flexibility"
                        ],
                        "minimum_allowed_amount": pattern[
                            "minimum_allowed_amount"
                        ],
                        "effective_date": next_date,
                        "source_event_id": pattern[
                            "source_event_id"
                        ],
                    }
                )

                next_date += timedelta(
                    days=gap
                )

    if not projected_rows:
        return pd.DataFrame()

    return pd.DataFrame(
        projected_rows
    )


# ============================================================
# BUILD 90-DAY FORECAST
# ============================================================

def build_forecast(
    events: pd.DataFrame,
    user_id: str,
    request_date,
    days: int = 90,
):
    """
    Build the user's baseline 90-day cash-flow forecast.

    Combines:

        1. Confirmed/known future events
        2. Projected recurring events
    """

    request_date = pd.Timestamp(
        request_date
    ).normalize()

    forecast_end = (
        request_date
        + timedelta(days=days)
    )

    user_events = events[
        events["user_id"] == user_id
    ].copy()

    if user_events.empty:
        return pd.DataFrame()

    user_events = prepare_events(
        user_events
    )

    # --------------------------------------------------------
    # Known future events
    # --------------------------------------------------------

    future_known = user_events[
        (user_events["effective_date"] >= request_date)
        & (user_events["effective_date"] <= forecast_end)
        & (
            ~user_events["status"]
            .astype(str)
            .str.lower()
            .isin(IGNORED_STATUSES)
        )
    ].copy()

    # --------------------------------------------------------
    # Project recurring events
    # --------------------------------------------------------

    projected = project_recurring_events(
        user_events,
        request_date,
        forecast_end,
    )

    # --------------------------------------------------------
    # Mark source
    # --------------------------------------------------------

    if not future_known.empty:
        future_known["_is_known"] = True

    if not projected.empty:
        projected["_is_known"] = False

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    frames = []

    if not future_known.empty:
        frames.append(
            future_known
        )

    if not projected.empty:
        frames.append(
            projected
        )

    if not frames:
        return pd.DataFrame()

    forecast = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    forecast["effective_date"] = pd.to_datetime(
        forecast["effective_date"],
        errors="coerce",
    ).dt.normalize()

    forecast = forecast[
        forecast["effective_date"].notna()
    ].copy()

    if forecast.empty:
        return forecast

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    forecast["_duplicate_key"] = (
        forecast["description"]
        .fillna("")
        .astype(str)
        + "|"
        + forecast["category"]
        .fillna("")
        .astype(str)
        + "|"
        + forecast["direction"]
        .fillna("")
        .astype(str)
        + "|"
        + forecast["effective_date"]
        .astype(str)
    )

    forecast = (
        forecast
        .sort_values(
            [
                "_duplicate_key",
                "_is_known",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset="_duplicate_key",
            keep="first",
        )
    )

    # --------------------------------------------------------
    # Final sorting
    # --------------------------------------------------------

    forecast = (
        forecast
        .drop(
            columns=[
                "_duplicate_key",
                "_is_known",
            ],
            errors="ignore",
        )
        .sort_values(
            [
                "effective_date",
                "event_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Compatibility column
    # --------------------------------------------------------
    #
    # decision_engine.py uses amount_home_currency.
    #
    # The forecast amount is the amount used by the financial
    # engine. This compatibility column prevents the cash-flow
    # simulation from failing when the normalized forecast is
    # passed directly to it.
    #
    # normalize_forecast() in decision_engine.py remains
    # responsible for currency normalization where required.
    # --------------------------------------------------------

    forecast[
        "amount_home_currency"
    ] = pd.to_numeric(
        forecast["amount"],
        errors="coerce",
    )

    return forecast


# ============================================================
# SPENDING CHANGE APPLICATION
# ============================================================

def apply_spending_change(
    forecast: pd.DataFrame,
    change: str,
) -> pd.DataFrame:
    """
    Apply one permitted spending change.

    Supported formats:

        stop:<event_id>

        reduce_to:<event_id>:<new_amount>
    """

    result = forecast.copy()

    if not change or change == "none":
        return result

    parts = change.split(":")

    if not parts:
        return result

    action = parts[0].strip().lower()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if action == "stop" and len(parts) == 2:

        event_id = parts[1]

        mask = (
            result["source_event_id"]
            .astype(str)
            == str(event_id)
        )

        mask = (
            mask
            | (
                result["event_id"]
                .astype(str)
                == str(event_id)
            )
        )

        result.loc[
            mask
            & (
                result["direction"]
                .astype(str)
                .str.lower()
                == "debit"
            ),
            "amount",
        ] = 0.0

        # Keep compatibility column synchronized.
        if "amount_home_currency" in result.columns:
            result.loc[
                mask
                & (
                    result["direction"]
                    .astype(str)
                    .str.lower()
                    == "debit"
                ),
                "amount_home_currency",
            ] = 0.0

        return result

    # --------------------------------------------------------
    # REDUCE
    # --------------------------------------------------------

    if (
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
            return result

        mask = (
            result["source_event_id"]
            .astype(str)
            == str(event_id)
        )

        mask = (
            mask
            | (
                result["event_id"]
                .astype(str)
                == str(event_id)
            )
        )

        debit_mask = (
            mask
            & (
                result["direction"]
                .astype(str)
                .str.lower()
                == "debit"
            )
        )

        result.loc[
            debit_mask,
            "amount",
        ] = new_amount

        if "amount_home_currency" in result.columns:
            result.loc[
                debit_mask,
                "amount_home_currency",
            ] = new_amount

        return result

    return result


# ============================================================
# CASH-FLOW SIMULATION
# ============================================================

def simulate_cashflow(
    forecast: pd.DataFrame,
    starting_balance: float,
    minimum_balance: float,
    request_payments=None,
    spending_changes=None,
):
    """
    Simulate cash flow over the forecast period.

    Positive balance change:
        credit

    Negative balance change:
        debit

    Returns:
        safe
        minimum_observed_balance
        ending_balance
        daily_balances
    """

    df = forecast.copy()

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "effective_date",
                "direction",
                "amount",
            ]
        )

    # --------------------------------------------------------
    # Apply spending changes
    # --------------------------------------------------------

    if spending_changes:

        for change in spending_changes:

            df = apply_spending_change(
                df,
                change,
            )

    # --------------------------------------------------------
    # Prepare events
    # --------------------------------------------------------

    df["effective_date"] = pd.to_datetime(
        df["effective_date"],
        errors="coerce",
    ).dt.normalize()

    df["amount"] = df[
        "amount"
    ].apply(
        _clean_amount
    )

    df = df[
        df["effective_date"].notna()
        & df["amount"].notna()
    ].copy()

    # --------------------------------------------------------
    # Request payments
    # --------------------------------------------------------

    payment_rows = []

    if request_payments:

        for payment in request_payments:

            if isinstance(
                payment,
                dict,
            ):

                payment_rows.append(
                    {
                        "effective_date": pd.Timestamp(
                            payment["date"]
                        ).normalize(),
                        "direction": "debit",
                        "amount": float(
                            payment["amount"]
                        ),
                        "event_id": payment.get(
                            "event_id",
                            "request_payment",
                        ),
                    }
                )

            elif isinstance(
                payment,
                (tuple, list),
            ):

                payment_rows.append(
                    {
                        "effective_date": pd.Timestamp(
                            payment[0]
                        ).normalize(),
                        "direction": "debit",
                        "amount": float(
                            payment[1]
                        ),
                        "event_id": "request_payment",
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
    # Sort events
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "effective_date",
            "event_id",
        ],
        na_position="last",
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

    daily_balances = []

    for date, day_events in df.groupby(
        "effective_date",
        sort=True,
    ):

        for _, event in day_events.iterrows():

            amount = float(
                event["amount"]
            )

            direction = str(
                event["direction"]
            ).lower()

            if direction == "credit":

                balance += amount

            elif direction == "debit":

                balance -= amount

        minimum_observed = min(
            minimum_observed,
            balance,
        )

        daily_balances.append(
            {
                "date": date,
                "balance": balance,
            }
        )

        # ----------------------------------------------------
        # Safety condition
        # ----------------------------------------------------

        if balance < minimum_balance:

            return {
                "safe": False,
                "minimum_observed_balance": (
                    minimum_observed
                ),
                "ending_balance": balance,
                "daily_balances": (
                    daily_balances
                ),
            }

    return {
        "safe": (
            minimum_observed
            >= minimum_balance
        ),
        "minimum_observed_balance": (
            minimum_observed
        ),
        "ending_balance": balance,
        "daily_balances": (
            daily_balances
        ),
    }