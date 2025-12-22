import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os


# =========================
# Colored Output (optional)
# =========================
def print_blue(msg):
    """
    Print a message in blue color (ANSI escape code).
    
    Args:
        msg: Any printable object (str/int/etc.)
    """
    print("\033[94m" + str(msg) + "\033[0m")


def print_red(msg):
    """
    Print a message in red color (ANSI escape code).
    
    Args:
        msg: Any printable object (str/int/etc.)
    """
    print("\033[91m" + str(msg) + "\033[0m")


# =========================
# Validation + Parsing
# =========================
def ask_nonempty(prompt):
    """
    Ask the user for input and ensure it is not empty.

    Args:
        prompt (str): Prompt text shown to the user.

    Returns:
        str: Trimmed, non-empty input.

    Raises:
        ValueError: If input is empty.
    """
    v = input(prompt).strip()
    if not v:
        raise ValueError("This field cannot be empty.")
    return v


def ask_float_positive(prompt):
    """
    Ask the user for a positive float.

    Args:
        prompt (str): Prompt text shown to the user.

    Returns:
        float: A positive float value.

    Raises:
        ValueError: If input is not a valid positive float.
    """
    try:
        v = float(input(prompt).strip())
        if v <= 0:
            raise ValueError
        return v
    except:
        raise ValueError("Enter a valid positive number.")


def ask_int_min(prompt, min_value=1):
    """
    Ask the user for an integer >= a minimum value.

    Args:
        prompt (str): Prompt text shown to the user.
        min_value (int): Minimum allowed integer value.

    Returns:
        int: Valid integer >= min_value.

    Raises:
        ValueError: If input is not valid or less than min_value.
    """
    try:
        v = int(input(prompt).strip())
        if v < min_value:
            raise ValueError
        return v
    except:
        raise ValueError(f"Enter an integer >= {min_value}")


def parse_time_hhmm(value):
    """
    Parse a HH:MM string into a datetime.time object.

    Args:
        value (str): Time string in HH:MM format.

    Returns:
        datetime.time: Parsed time.

    Raises:
        ValueError: If format is invalid.
    """
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except:
        raise ValueError("Invalid time format. Use HH:MM (e.g., 08:30).")


def parse_date_flexible(value):
    """
    Parse a date string using multiple formats into a datetime.date object.

    Supported formats:
        - YYYY-MM-DD
        - YYYY/MM/DD
        - DD/MM/YYYY
        - DD-MM-YYYY

    Args:
        value (str): Date string.

    Returns:
        datetime.date: Parsed date.

    Raises:
        ValueError: If no supported format matches.
    """
    patterns = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]
    for p in patterns:
        try:
            return datetime.strptime(value.strip(), p).date()
        except:
            pass
    raise ValueError("Invalid date format.")


# =========================
# Files (CSV)
# =========================
MEDS_FILE = "medications.csv"
LOG_FILE  = "dose_log.csv"

MEDS_COLUMNS = [
    "patient", "medication", "dosage_mg",
    "times_per_day", "first_time", "duration_days", "start_date"
]

LOG_COLUMNS = [
    "date_time", "patient", "medication", "scheduled_date", "scheduled_time", "status"
]


def ensure_csv(filename, columns):
    """
    Ensure a CSV file exists; if not, create it with the given columns.

    Args:
        filename (str): Target CSV filename.
        columns (list[str]): Column names for the CSV.
    """
    if not os.path.exists(filename):
        pd.DataFrame(columns=columns).to_csv(filename, index=False)


def _normalize_text_cols(df, cols):
    """
    Normalize specified columns as clean strings: fill NaN, cast to str, strip whitespace.
    This prevents pandas .str accessor errors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list[str]): Columns to normalize.

    Returns:
        pd.DataFrame: Updated DataFrame with normalized text columns.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def load_meds():
    """
    Load medications CSV into a DataFrame and enforce expected schema.

    Returns:
        pd.DataFrame: Medications DataFrame with normalized text columns and numeric conversions.
    """
    ensure_csv(MEDS_FILE, MEDS_COLUMNS)
    df = pd.read_csv(MEDS_FILE)

    # ensure all expected columns exist even if file is weird
    for c in MEDS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[MEDS_COLUMNS]

    df = _normalize_text_cols(df, ["patient", "medication", "first_time", "start_date"])

    # safe numeric conversions
    df["dosage_mg"] = pd.to_numeric(df["dosage_mg"], errors="coerce")
    df["times_per_day"] = pd.to_numeric(df["times_per_day"], errors="coerce")
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")

    return df


def save_meds(df):
    """
    Save medications DataFrame to CSV, enforcing the expected column order.

    Args:
        df (pd.DataFrame): Medications DataFrame to save.
    """
    df = df.copy()
    for c in MEDS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[MEDS_COLUMNS]
    df.to_csv(MEDS_FILE, index=False)


def load_log():
    """
    Load dose log CSV into a DataFrame and enforce expected schema.

    Returns:
        pd.DataFrame: Log DataFrame with normalized text columns.
    """
    ensure_csv(LOG_FILE, LOG_COLUMNS)
    df = pd.read_csv(LOG_FILE)

    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LOG_COLUMNS]

    df = _normalize_text_cols(df, ["date_time", "patient", "medication", "scheduled_date", "scheduled_time", "status"])
    return df


def append_log(row_dict):
    """
    Append a single log record (row) to the log CSV.

    Args:
        row_dict (dict): A dictionary matching LOG_COLUMNS keys/fields.
    """
    ensure_csv(LOG_FILE, LOG_COLUMNS)
    df = pd.read_csv(LOG_FILE)

    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LOG_COLUMNS]

    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


# =========================
# OOP
# =========================
class Medication:
    """
    Represents one medication course for a patient.
    Includes schedule parameters and helper methods to compute dose times.
    """

    def __init__(self, patient, name, dosage_mg, times_per_day, first_time, duration_days, start_date):
        """
        Initialize a Medication object and validate inputs.

        Args:
            patient (str): Patient name.
            name (str): Medication name.
            dosage_mg (float): Dosage in mg.
            times_per_day (int): Number of doses per day.
            first_time (datetime.time): First dose time of day.
            duration_days (int): Treatment duration in days.
            start_date (datetime.date): Start date of course.
        """
        self.patient = patient
        self.name = name
        self.dosage_mg = dosage_mg
        self.times_per_day = times_per_day
        self.first_time = first_time          # time object
        self.duration_days = duration_days
        self.start_date = start_date          # date object
        self.validate()

    def validate(self):
        """
        Validate medication fields.

        Raises:
            ValueError: If any field is invalid.
        """
        if not str(self.patient).strip():
            raise ValueError("Patient name required.")
        if not str(self.name).strip():
            raise ValueError("Medication name required.")
        if float(self.dosage_mg) <= 0:
            raise ValueError("Dosage must be positive.")
        if int(self.times_per_day) < 1:
            raise ValueError("Times per day must be >= 1.")
        if int(self.duration_days) < 1:
            raise ValueError("Duration must be >= 1.")

    def end_date(self):
        """
        Compute the end date of the medication course.

        Returns:
            datetime.date: End date (start_date + duration_days - 1).
        """
        return self.start_date + timedelta(days=int(self.duration_days) - 1)

    def total_doses(self):
        """
        Compute the total number of doses across the full course.

        Returns:
            int: times_per_day * duration_days
        """
        return int(self.times_per_day) * int(self.duration_days)

    def dose_datetimes_for_day(self, day):
        """
        Generate dose datetimes for a given day using NumPy offsets.

        Args:
            day (datetime.date): The day to compute doses for.

        Returns:
            list[datetime.datetime]: Dose datetimes for the day (empty if day out of range).
        """
        if day < self.start_date or day > self.end_date():
            return []

        base = datetime.combine(day, self.first_time)
        interval_hours = 24.0 / float(self.times_per_day)

        offsets = np.arange(int(self.times_per_day)) * interval_hours
        return [base + timedelta(hours=float(h)) for h in offsets]

    def next_dose_datetime(self, now):
        """
        Find the next scheduled dose datetime >= now, scanning forward in the course window.

        Args:
            now (datetime.datetime): Current time.

        Returns:
            datetime.datetime | None: Next dose datetime, or None if no future doses exist.
        """
        for i in range(int(self.duration_days) + 1):
            d = now.date() + timedelta(days=i)
            for t in self.dose_datetimes_for_day(d):
                if t >= now:
                    return t
        return None

    def to_row(self):
        """
        Convert the Medication object into a dict row suitable for CSV/DataFrame.

        Returns:
            dict: Row matching MEDS_COLUMNS fields.
        """
        return {
            "patient": str(self.patient).strip(),
            "medication": str(self.name).strip(),
            "dosage_mg": float(self.dosage_mg),
            "times_per_day": int(self.times_per_day),
            "first_time": self.first_time.strftime("%H:%M"),
            "duration_days": int(self.duration_days),
            "start_date": self.start_date.strftime("%Y-%m-%d")
        }


class MedicationSchedule:
    """
    Manages medications for a single patient (CRUD operations on the meds DataFrame).
    """

    def __init__(self, patient_name):
        """
        Initialize schedule for a patient.

        Args:
            patient_name (str): Patient name.
        """
        self.patient_name = str(patient_name).strip()

    def add_medication(self, med_obj, meds_df):
        """
        Add a medication for the current patient (prevent duplicates by name).

        Args:
            med_obj (Medication): Medication object to add.
            meds_df (pd.DataFrame): Medications DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame after adding and saving.

        Raises:
            ValueError: If duplicate medication exists for this patient.
        """
        meds_df = meds_df.copy()
        meds_df = _normalize_text_cols(meds_df, ["patient", "medication", "first_time", "start_date"])

        exists = meds_df[
            (meds_df["patient"].str.lower() == self.patient_name.lower()) &
            (meds_df["medication"].str.lower() == med_obj.name.lower())
        ]
        if not exists.empty:
            raise ValueError("Medication already exists for this patient.")

        meds_df = pd.concat([meds_df, pd.DataFrame([med_obj.to_row()])], ignore_index=True)
        save_meds(meds_df)
        return meds_df

    def remove_medication(self, name, meds_df):
        """
        Remove a medication by name for the current patient.

        Args:
            name (str): Medication name to remove.
            meds_df (pd.DataFrame): Medications DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame after removal and saving.

        Raises:
            ValueError: If medication not found.
        """
        meds_df = meds_df.copy()
        meds_df = _normalize_text_cols(meds_df, ["patient", "medication"])

        before = len(meds_df)
        meds_df = meds_df[
            ~(
                (meds_df["patient"].str.lower() == self.patient_name.lower()) &
                (meds_df["medication"].str.lower() == str(name).strip().lower())
            )
        ]
        after = len(meds_df)
        if before == after:
            raise ValueError("Medication not found.")
        save_meds(meds_df)
        return meds_df

    def get_patient_meds_df(self, meds_df):
        """
        Filter the meds DataFrame to only the current patient's medications.

        Args:
            meds_df (pd.DataFrame): Medications DataFrame.

        Returns:
            pd.DataFrame: Copy of rows for the current patient.
        """
        meds_df = meds_df.copy()
        meds_df = _normalize_text_cols(meds_df, ["patient"])
        return meds_df[meds_df["patient"].str.lower() == self.patient_name.lower()].copy()


class Reminder:
    """
    Generates reports and due-soon checks for a patient's medication schedule.
    Uses Medication objects reconstructed from DataFrame rows.
    """

    def __init__(self, schedule: MedicationSchedule):
        """
        Initialize reminder helper.

        Args:
            schedule (MedicationSchedule): Patient schedule manager.
        """
        self.schedule = schedule

    def _row_to_med(self, row):
        """
        Convert a DataFrame row into a Medication object.

        Args:
            row (pd.Series | dict): Medication row.

        Returns:
            Medication: Constructed Medication object.
        """
        return Medication(
            patient=row["patient"],
            name=row["medication"],
            dosage_mg=float(row["dosage_mg"]),
            times_per_day=int(row["times_per_day"]),
            first_time=parse_time_hhmm(str(row["first_time"])),
            duration_days=int(row["duration_days"]),
            start_date=parse_date_flexible(str(row["start_date"]))
        )

    def taken_count_for_med(self, patient, med_name, log_df):
        """
        Count TAKEN logs for a given patient+medication.

        Args:
            patient (str): Patient name.
            med_name (str): Medication name.
            log_df (pd.DataFrame): Log DataFrame.

        Returns:
            int: Number of TAKEN records.
        """
        log_df = log_df.copy()
        log_df = _normalize_text_cols(log_df, ["patient", "medication", "status"])

        f = (
            (log_df["patient"].str.lower() == patient.lower()) &
            (log_df["medication"].str.lower() == med_name.lower()) &
            (log_df["status"].str.upper() == "TAKEN")
        )
        return int(f.sum())

    def remaining_doses_for_med(self, med_obj, log_df):
        """
        Compute remaining doses for a medication based on total_doses - taken_count.

        Args:
            med_obj (Medication): Medication object.
            log_df (pd.DataFrame): Log DataFrame.

        Returns:
            int: Remaining dose count (>=0).
        """
        taken = self.taken_count_for_med(med_obj.patient, med_obj.name, log_df)
        return max(0, med_obj.total_doses() - taken)

    def daily_report(self, meds_df, log_df, day: date):
        """
        Build a daily schedule report as a DataFrame for the patient.

        Args:
            meds_df (pd.DataFrame): Medications DataFrame.
            log_df (pd.DataFrame): Log DataFrame.
            day (datetime.date): Day to report.

        Returns:
            pd.DataFrame: Report with columns [Patient, Medication, Dosage (mg), Dose Time, Remaining Doses]
        """
        patient_df = self.schedule.get_patient_meds_df(meds_df)
        rows = []

        for _, r in patient_df.iterrows():
            med = self._row_to_med(r)
            for dt in med.dose_datetimes_for_day(day):
                rows.append({
                    "Patient": med.patient,
                    "Medication": med.name,
                    "Dosage (mg)": med.dosage_mg,
                    "Dose Time": dt.strftime("%H:%M"),
                    "Remaining Doses": self.remaining_doses_for_med(med, log_df)
                })

        df = pd.DataFrame(rows)
        return df.sort_values(by="Dose Time") if not df.empty else df

    def due_soon(self, meds_df, now, window_minutes=15):
        """
        Check medications due within the next window_minutes.

        Args:
            meds_df (pd.DataFrame): Medications DataFrame.
            now (datetime.datetime): Current time.
            window_minutes (int): Time window in minutes.

        Returns:
            pd.DataFrame: Due soon table with [Medication, Next Dose, Minutes Left]
        """
        patient_df = self.schedule.get_patient_meds_df(meds_df)
        rows = []

        for _, r in patient_df.iterrows():
            med = self._row_to_med(r)
            nxt = med.next_dose_datetime(now)
            if nxt is None:
                continue

            minutes = int((np.datetime64(nxt) - np.datetime64(now)) / np.timedelta64(1, "m"))
            if 0 <= minutes <= int(window_minutes):
                rows.append({
                    "Medication": med.name,
                    "Next Dose": nxt.strftime("%Y-%m-%d %H:%M"),
                    "Minutes Left": minutes
                })

        return pd.DataFrame(rows)


# =========================
# Actions (Logging + Reports + Plots)
# =========================
def log_dose(patient, med_name, scheduled_time_hhmm, status, meds_df):
    """
    Log a dose event (TAKEN/MISSED) for the current patient and medication.

    This validates:
      - status is TAKEN or MISSED
      - scheduled_time is valid HH:MM
      - medication exists for that patient in meds_df

    Args:
        patient (str): Patient name.
        med_name (str): Medication name.
        scheduled_time_hhmm (str): Scheduled time in HH:MM.
        status (str): "TAKEN" or "MISSED".
        meds_df (pd.DataFrame): Medications DataFrame (for existence check).

    Raises:
        ValueError: If status/time invalid or medication not found for patient.
    """
    status = status.strip().upper()
    if status not in ["TAKEN", "MISSED"]:
        raise ValueError("Status must be TAKEN or MISSED.")

    parse_time_hhmm(scheduled_time_hhmm)

    meds_df = meds_df.copy()
    meds_df = _normalize_text_cols(meds_df, ["patient", "medication"])
    exists = meds_df[
        (meds_df["patient"].str.lower() == patient.lower()) &
        (meds_df["medication"].str.lower() == med_name.lower())
    ]
    if exists.empty:
        raise ValueError("This medication is not found for the current patient.")

    now = datetime.now()
    row = {
        "date_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "patient": patient,
        "medication": med_name,
        "scheduled_date": now.strftime("%Y-%m-%d"),
        "scheduled_time": scheduled_time_hhmm.strip(),
        "status": status
    }
    append_log(row)
    print_blue("Dose logged successfully.")


def daily_summary_file(patient, meds_df, log_df):
    """
    Print and save a daily summary report (txt) for a given patient.

    The report includes:
      - Total scheduled doses = sum(times_per_day) for patient's meds
      - TAKEN/MISSED counts from today's log entries

    Args:
        patient (str): Patient name.
        meds_df (pd.DataFrame): Medications DataFrame.
        log_df (pd.DataFrame): Log DataFrame.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    meds_df = meds_df.copy()
    meds_df = _normalize_text_cols(meds_df, ["patient"])
    patient_df = meds_df[meds_df["patient"].str.lower() == patient.lower()]

    total_scheduled = int(patient_df["times_per_day"].sum()) if not patient_df.empty else 0

    log_df = log_df.copy()
    log_df = _normalize_text_cols(log_df, ["patient", "date_time", "status"])
    today_logs = log_df[
        (log_df["patient"].str.lower() == patient.lower()) &
        (log_df["date_time"].astype(str).str.startswith(today))
    ]

    taken = int((today_logs["status"].str.upper() == "TAKEN").sum())
    missed = int((today_logs["status"].str.upper() == "MISSED").sum())

    print("\nDaily Report:")
    print("Patient:", patient)
    print("Date:", today)
    print("Total scheduled doses (sum times/day):", total_scheduled)
    print("TAKEN:", taken)
    print("MISSED:", missed)

    filename = f"report_{patient}_{today}.txt".replace(" ", "_")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Daily Report\n")
        f.write(f"Patient: {patient}\n")
        f.write(f"Date: {today}\n")
        f.write(f"Total scheduled (sum times/day): {total_scheduled}\n")
        f.write(f"Taken: {taken}\n")
        f.write(f"Missed: {missed}\n")

    print_blue("Report saved: " + filename)


def plot_remaining_doses(patient, meds_df, log_df):
    """
    Plot a bar chart showing remaining doses per medication for a patient.

    Remaining doses are computed using:
      remaining = total_doses (times_per_day * duration_days) - TAKEN_count

    Args:
        patient (str): Patient name.
        meds_df (pd.DataFrame): Medications DataFrame.
        log_df (pd.DataFrame): Log DataFrame.
    """
    meds_df = meds_df.copy()
    meds_df = _normalize_text_cols(meds_df, ["patient"])
    patient_df = meds_df[meds_df["patient"].str.lower() == patient.lower()]

    if patient_df.empty:
        print_red("No medications to visualize.")
        return

    reminder = Reminder(MedicationSchedule(patient))
    meds = []
    remaining = []
    for _, r in patient_df.iterrows():
        med = reminder._row_to_med(r)
        meds.append(med.name)
        remaining.append(reminder.remaining_doses_for_med(med, log_df))

    plt.figure(figsize=(8, 4))
    plt.bar(meds, remaining)
    plt.title("Remaining Doses - " + patient)
    plt.xlabel("Medication")
    plt.ylabel("Remaining Doses")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


# =========================
# Menu
# =========================
def main():
    """
    Run the CLI menu loop for PyMedix.

    Flow:
      - Load meds + log data
      - Ask for active patient
      - Loop menu options:
          add/delete/report/due/log/dataframes/plot/switch/report file/exit
    """
    meds_df = load_meds()
    log_df = load_log()

    current_patient = ask_nonempty("Enter patient name: ")
    schedule = MedicationSchedule(current_patient)
    reminder = Reminder(schedule)
    print_blue(f"Active patient: {current_patient}")

    while True:
        meds_df = load_meds()
        log_df = load_log()

        print("\n====================")
        print("PyMedix - Smart Medication System")
        print("====================")
        print("1) Add medication")
        print("2) Delete medication")
        print("3) Daily report (today)")
        print("4) Due soon (next 15 mins)")
        print("5) Log dose (TAKEN/MISSED)")
        print("6) DataFrame (current patient)")
        print("7) DataFrame (all patients)")
        print("8) Visualization (remaining doses)")
        print("9) Switch patient")
        print("10) Daily report file (txt)")
        print("0) Exit")

        choice = input("Choose: ").strip()

        try:
            if choice == "1":
                med = Medication(
                    patient=current_patient,
                    name=ask_nonempty("Medication name: "),
                    dosage_mg=ask_float_positive("Dosage (mg): "),
                    times_per_day=ask_int_min("Times per day: ", 1),
                    first_time=parse_time_hhmm(ask_nonempty("First time HH:MM: ")),
                    duration_days=ask_int_min("Duration days: ", 1),
                    start_date=parse_date_flexible(ask_nonempty("Start date: "))
                )
                meds_df = schedule.add_medication(med, meds_df)
                print_blue("Medication added.")

            elif choice == "2":
                name = ask_nonempty("Medication name to delete: ")
                meds_df = schedule.remove_medication(name, meds_df)
                print_blue("Medication deleted.")

            elif choice == "3":
                df = reminder.daily_report(meds_df, log_df, datetime.now().date())
                print(df if not df.empty else "No data")

            elif choice == "4":
                df = reminder.due_soon(meds_df, datetime.now(), window_minutes=15)
                print(df if not df.empty else "No due medications")

            elif choice == "5":
                med_name = ask_nonempty("Medication name: ")
                t = ask_nonempty("Scheduled time (HH:MM): ")
                s = ask_nonempty("TAKEN or MISSED: ").upper()
                log_dose(current_patient, med_name, t, s, meds_df)

            elif choice == "6":
                dfp = schedule.get_patient_meds_df(meds_df)
                print(dfp if not dfp.empty else "No medications for this patient")

            elif choice == "7":
                print(meds_df if not meds_df.empty else "No patients / medications yet")

            elif choice == "8":
                plot_remaining_doses(current_patient, meds_df, log_df)

            elif choice == "9":
                current_patient = ask_nonempty("Enter patient name: ")
                schedule = MedicationSchedule(current_patient)
                reminder = Reminder(schedule)
                print_blue(f"Switched. Active patient: {current_patient}")

            elif choice == "10":
                daily_summary_file(current_patient, meds_df, log_df)

            elif choice == "0":
                print_blue("Bye 👋")
                break

            else:
                print_red("Invalid choice.")

        except Exception as e:
            print_red(e)


if __name__ == "__main__":
    main()

