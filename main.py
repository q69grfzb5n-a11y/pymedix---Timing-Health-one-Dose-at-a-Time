import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os


# ================
# Colored Output 
# ================
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

    def _init_(self, patient, name, dosage_mg, times_per_day, first_time, duration_days, start_date):
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

    def _init_(self, patient_name):
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

    def _init_(self, schedule: MedicationSchedule):
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

    filename = f"report_{patient}{today}.txt".replace(" ", "")
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


if _name_ == "_main_":
    main()
# ==========================================
# PyMedix - Full Tkinter GUI (Multi-Patients)
# (No Visualization) + Fixed Logo + 2-Column Menu
# ==========================================

import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta, date

# Pillow for logo
from PIL import Image, ImageTk


# =========================
# Colors / Theme
# =========================
BG_ROOT = "#0a192f"
BG_MAIN = "#102a43"
BG_CONTENT = "#1c4e80"

BTN_BG = "#1976d2"
BTN_BG_ACTIVE = "#0d47a1"
TXT_HEADER = "#bbdefb"


# =========================
# Validation Helpers
# =========================
def parse_time_hhmm(value: str):
    """Parse HH:MM time string into a datetime.time object."""
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except Exception:
        raise ValueError("Invalid time format. Use HH:MM (e.g., 08:30).")


def parse_date_flexible(value: str):
    """Parse multiple date formats into a datetime.date object."""
    patterns = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]
    v = value.strip()
    for p in patterns:
        try:
            return datetime.strptime(v, p).date()
        except Exception:
            pass
    raise ValueError("Invalid date format. Use YYYY-MM-DD or DD/MM/YYYY, etc.")


def ensure_positive_number(x, name="value"):
    """Ensure numeric value is positive."""
    if x <= 0:
        raise ValueError(f"{name} must be > 0")
    return x


# =========================
# Data Storage (CSV)
# =========================
DATA_DIR = "pymedix_data"
MEDS_CSV = os.path.join(DATA_DIR, "medications.csv")
LOG_CSV = os.path.join(DATA_DIR, "dose_log.csv")

MEDS_COLUMNS = [
    "patient_name",
    "medication",
    "dosage_mg",
    "times_per_day",
    "first_time",       # HH:MM
    "duration_days",
    "start_date"        # YYYY-MM-DD
]

LOG_COLUMNS = [
    "patient_name",
    "medication",
    "dose_time",        # YYYY-MM-DD HH:MM
    "status"            # TAKEN / MISSED
]


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_meds():
    """Load medications dataframe from CSV (or empty)."""
    _ensure_data_dir()
    if os.path.exists(MEDS_CSV):
        df = pd.read_csv(MEDS_CSV)
        # ensure columns exist
        for c in MEDS_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[MEDS_COLUMNS].copy()
    return pd.DataFrame(columns=MEDS_COLUMNS)


def save_meds(df: pd.DataFrame):
    """Save medications dataframe to CSV."""
    _ensure_data_dir()
    df.to_csv(MEDS_CSV, index=False)


def load_log():
    """Load log dataframe from CSV (or empty)."""
    _ensure_data_dir()
    if os.path.exists(LOG_CSV):
        df = pd.read_csv(LOG_CSV)
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LOG_COLUMNS].copy()
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df: pd.DataFrame):
    """Save log dataframe to CSV."""
    _ensure_data_dir()
    df.to_csv(LOG_CSV, index=False)


def list_patients(meds_df: pd.DataFrame, log_df: pd.DataFrame):
    """Return sorted unique patient names from meds and log."""
    names = set()
    if not meds_df.empty:
        names |= set(meds_df["patient_name"].dropna().astype(str).str.strip())
    if not log_df.empty:
        names |= set(log_df["patient_name"].dropna().astype(str).str.strip())
    names = {n for n in names if n}
    return sorted(names, key=lambda x: x.lower())


# =========================
# Core Logic
# =========================
class Medication:
    """Medication record container."""
    def _init_(self, patient_name, medication, dosage_mg, times_per_day,
                 first_time, duration_days, start_date):
        self.patient_name = patient_name
        self.medication = medication
        self.dosage_mg = dosage_mg
        self.times_per_day = times_per_day
        self.first_time = first_time
        self.duration_days = duration_days
        self.start_date = start_date


class MedicationSchedule:
    """Schedule manager for a patient."""
    def _init_(self, patient_name: str):
        self.patient_name = patient_name.strip()

    def add_medication(self, med: Medication, meds_df: pd.DataFrame) -> pd.DataFrame:
        """Add medication row to meds_df."""
        name = med.medication.strip()
        if not name:
            raise ValueError("Medication name is required.")

        row = {
            "patient_name": self.patient_name,
            "medication": name,
            "dosage_mg": float(ensure_positive_number(med.dosage_mg, "dosage_mg")),
            "times_per_day": int(ensure_positive_number(med.times_per_day, "times_per_day")),
            "first_time": med.first_time.strftime("%H:%M"),
            "duration_days": int(ensure_positive_number(med.duration_days, "duration_days")),
            "start_date": med.start_date.isoformat(),
        }

        new_df = pd.concat([meds_df, pd.DataFrame([row])], ignore_index=True)
        save_meds(new_df)
        return new_df

    def remove_medication(self, med_name: str, meds_df: pd.DataFrame) -> pd.DataFrame:
        """Remove medication (by name) for this patient."""
        med_name = med_name.strip()
        if not med_name:
            raise ValueError("Enter medication name to delete.")

        before = len(meds_df)
        new_df = meds_df[~(
            (meds_df["patient_name"].astype(str) == self.patient_name) &
            (meds_df["medication"].astype(str).str.lower() == med_name.lower())
        )].copy()

        if len(new_df) == before:
            raise ValueError("Medication not found for this patient.")

        save_meds(new_df)
        return new_df

    def get_patient_meds_df(self, meds_df: pd.DataFrame) -> pd.DataFrame:
        """Filter meds_df for this patient."""
        if meds_df.empty:
            return meds_df.copy()
        df = meds_df[meds_df["patient_name"].astype(str) == self.patient_name].copy()
        return df.reset_index(drop=True)


class Reminder:
    """Reports helper class."""
    def _init_(self, schedule: MedicationSchedule):
        self.schedule = schedule

    def due_soon(self, meds_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """
        Simple 'due soon': meds whose next scheduled time is within the next 90 minutes.
        (Basic logic: uses first_time + evenly spaced doses per day from start_date)
        """
        patient = self.schedule.patient_name
        pm = meds_df[meds_df["patient_name"].astype(str) == patient].copy()
        if pm.empty:
            return pd.DataFrame()

        results = []
        for _, r in pm.iterrows():
            try:
                med = str(r["medication"])
                times = int(r["times_per_day"])
                first = parse_time_hhmm(str(r["first_time"]))
                start = parse_date_flexible(str(r["start_date"]))
                dur = int(r["duration_days"])
            except Exception:
                continue

            # check if still active
            end_date = start + timedelta(days=dur - 1)
            if now.date() < start or now.date() > end_date:
                continue

            # generate today dose times
            if times <= 0:
                continue
            interval_minutes = int(24 * 60 / times)
            t0 = datetime.combine(now.date(), first)
            for k in range(times):
                t = t0 + timedelta(minutes=k * interval_minutes)
                delta = (t - now).total_seconds() / 60
                if 0 <= delta <= 90:
                    results.append({
                        "patient_name": patient,
                        "medication": med,
                        "due_time": t.strftime("%Y-%m-%d %H:%M"),
                        "in_minutes": int(delta)
                    })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values(["in_minutes", "medication"]).reset_index(drop=True)

    def daily_report(self, meds_df: pd.DataFrame, log_df: pd.DataFrame, day: date) -> pd.DataFrame:
        """Daily report: all meds for patient + taken/missed count for that date."""
        patient = self.schedule.patient_name
        pm = meds_df[meds_df["patient_name"].astype(str) == patient].copy()
        if pm.empty:
            return pd.DataFrame()

        # Filter logs for this patient/day
        pl = log_df[log_df["patient_name"].astype(str) == patient].copy()
        if not pl.empty:
            pl["dose_time"] = pd.to_datetime(pl["dose_time"], errors="coerce")
            pl = pl[pl["dose_time"].dt.date == day]

        rows = []
        for _, r in pm.iterrows():
            med = str(r["medication"])
            taken = 0
            missed = 0
            if not pl.empty:
                m = pl[pl["medication"].astype(str).str.lower() == med.lower()]
                taken = int((m["status"].astype(str) == "TAKEN").sum())
                missed = int((m["status"].astype(str) == "MISSED").sum())

            rows.append({
                "patient_name": patient,
                "medication": med,
                "dosage_mg": r["dosage_mg"],
                "times_per_day": r["times_per_day"],
                "start_date": r["start_date"],
                "duration_days": r["duration_days"],
                "taken_today": taken,
                "missed_today": missed
            })

        return pd.DataFrame(rows)


def log_dose(patient_name: str, medication: str, time_hhmm: str, status: str,
             meds_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """Append a dose log entry after validation."""
    patient_name = patient_name.strip()
    medication = medication.strip()
    if not patient_name:
        raise ValueError("No active patient.")
    if not medication:
        raise ValueError("Medication name is required.")
    if status not in ("TAKEN", "MISSED"):
        raise ValueError("Status must be TAKEN or MISSED.")

    # validate time
    t = parse_time_hhmm(time_hhmm)

    # optional: ensure medication exists for that patient
    exists = False
    if not meds_df.empty:
        x = meds_df[
            (meds_df["patient_name"].astype(str) == patient_name) &
            (meds_df["medication"].astype(str).str.lower() == medication.lower())
        ]
        exists = not x.empty
    if not exists:
        raise ValueError("This medication is not found for the active patient.")

    dt = datetime.combine(datetime.now().date(), t)

    row = {
        "patient_name": patient_name,
        "medication": medication,
        "dose_time": dt.strftime("%Y-%m-%d %H:%M"),
        "status": status
    }

    new_log = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    save_log(new_log)
    return new_log


# =========================
# Tkinter App
# =========================
meds_df = load_meds()
log_df = load_log()

schedule = None
reminder = None

root = tk.Tk()
root.title("Pymedix - Medication Management System")
root.geometry("1200x700")
root.configure(bg=BG_ROOT)

# ---------- STYLE ----------
style = ttk.Style()
style.theme_use("clam")

style.configure("TButton",
    background=BTN_BG,
    foreground="white",
    font=("Segoe UI", 11),
    padding=8
)
style.map("TButton",
    background=[("active", BTN_BG_ACTIVE)]
)

style.configure("TLabel",
    background=BG_ROOT,
    foreground="white",
    font=("Segoe UI", 11)
)

style.configure("HeaderTitle.TLabel",
    font=("Segoe UI", 28, "bold"),
    foreground=TXT_HEADER,
    background=BG_ROOT
)

style.configure("HeaderSub.TLabel",
    font=("Segoe UI", 14),
    foreground="white",
    background=BG_ROOT
)

# =========================
# HEADER (Logo + Title)
# =========================
header = tk.Frame(root, bg=BG_ROOT)
header.pack(fill="x", pady=10)

logo_label = tk.Label(header, bg=BG_ROOT)
logo_label.grid(row=0, column=0, rowspan=2, padx=(20, 10), sticky="w")

title_box = tk.Frame(header, bg=BG_ROOT)
title_box.grid(row=0, column=1, sticky="w")

ttk.Label(title_box, text="Pymedix", style="HeaderTitle.TLabel").pack(anchor="w")
ttk.Label(title_box, text="A Python-Based Medication Management Project", style="HeaderSub.TLabel").pack(anchor="w")

header.grid_columnconfigure(1, weight=1)

# Load logo (fixed size; doesn't affect sidebar/menu)
def load_logo(path: str, size=(120, 120)):
    try:
        img = Image.open(path)
        img = img.resize(size)  # keep whatever you want
        logo_img = ImageTk.PhotoImage(img)
        logo_label.config(image=logo_img)
        logo_label.image = logo_img
    except Exception as e:
        print("Logo load error:", e)

# put your path here (you can change anytime)
logo_path = r"C:\Users\Jana7\Downloads\logo.png"
load_logo(logo_path, size=(120, 120))

# =========================
# MAIN LAYOUT
# =========================
main = tk.Frame(root, bg=BG_MAIN)
main.pack(fill="both", expand=True, padx=15, pady=10)

sidebar = tk.Frame(main, bg=BG_MAIN, width=360)
sidebar.pack(side="left", fill="y", padx=(10, 12))
sidebar.pack_propagate(False)

content = tk.Frame(main, bg=BG_CONTENT)
content.pack(side="right", fill="both", expand=True)
content.pack_propagate(False)

# =========================
# Helpers
# =========================
def clear_content():
    for w in content.winfo_children():
        w.destroy()

def require_patient():
    if schedule is None:
        messagebox.showerror("Error", "Please activate a patient first")
        return False
    return True

def refresh_patient_list():
    global meds_df, log_df
    patients = list_patients(meds_df, log_df)
    patient_combo["values"] = patients

def set_patient_from_ui():
    """Activate from entry OR combobox."""
    global schedule, reminder
    name = patient_entry.get().strip()

    # if entry empty, try combobox selection
    if not name:
        selected = patient_combo.get().strip()
        name = selected

    if not name:
        messagebox.showerror("Error", "Enter patient name or choose from the list.")
        return

    schedule = MedicationSchedule(name)
    reminder = Reminder(schedule)
    patient_status.config(text=f"Active Patient: {name}")

    # update combobox values (in case new patient created)
    refresh_patient_list()
    patient_combo.set(name)

def show_dataframe(df: pd.DataFrame, title: str):
    """Show DataFrame in a Treeview (clean view)."""
    clear_content()

    ttk.Label(content, text=title, font=("Segoe UI", 18), background=BG_CONTENT, foreground="white")\
        .pack(pady=(12, 8), anchor="w", padx=12)

    if df is None or df.empty:
        ttk.Label(content, text="No data to display.", background=BG_CONTENT, foreground="white")\
            .pack(padx=12, pady=10, anchor="w")
        return

    container = tk.Frame(content, bg=BG_CONTENT)
    container.pack(fill="both", expand=True, padx=12, pady=12)

    cols = list(df.columns)

    tree = ttk.Treeview(container, columns=cols, show="headings")
    tree.pack(side="left", fill="both", expand=True)

    vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    vsb.pack(side="right", fill="y")
    tree.configure(yscrollcommand=vsb.set)

    # headings
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=130, anchor="center")

    # rows
    for _, row in df.iterrows():
        values = [str(row[c]) for c in cols]
        tree.insert("", "end", values=values)


# =========================
# Sidebar - Patient Controls
# =========================
ttk.Label(sidebar, text="Patient Name").pack(pady=(14, 4))
patient_entry = ttk.Entry(sidebar)
patient_entry.pack(fill="x", padx=6)

patient_status = ttk.Label(sidebar, text="No active patient")
patient_status.pack(pady=(8, 10))

ttk.Label(sidebar, text="Existing Patients").pack(pady=(6, 4))
patient_combo = ttk.Combobox(sidebar, state="readonly")
patient_combo.pack(fill="x", padx=6)

ttk.Button(sidebar, text="Activate / Switch Patient", command=set_patient_from_ui)\
    .pack(fill="x", pady=10, padx=6)


# =========================
# Screens
# =========================
def add_med_screen():
    if not require_patient(): return
    clear_content()

    ttk.Label(content, text="Add Medication", font=("Segoe UI", 18), background=BG_CONTENT, foreground="white")\
        .grid(row=0, column=0, columnspan=2, pady=15, padx=10, sticky="w")

    fields = [
        ("Medication Name", "name"),
        ("Dosage (mg)", "dose"),
        ("Times per Day", "times"),
        ("First Time (HH:MM)", "first"),
        ("Duration (Days)", "duration"),
        ("Start Date", "start")
    ]

    entries = {}
    for i, (label, key) in enumerate(fields, start=1):
        ttk.Label(content, text=label, width=22, anchor="e", background=BG_CONTENT, foreground="white")\
            .grid(row=i, column=0, padx=10, pady=8, sticky="e")

        e = ttk.Entry(content, width=30)
        e.grid(row=i, column=1, padx=10, pady=8, sticky="w")
        entries[key] = e

    def save():
        global meds_df
        try:
            med = Medication(
                schedule.patient_name,
                entries["name"].get(),
                float(entries["dose"].get()),
                int(entries["times"].get()),
                parse_time_hhmm(entries["first"].get()),
                int(entries["duration"].get()),
                parse_date_flexible(entries["start"].get())
            )
            meds_df = schedule.add_medication(med, meds_df)
            refresh_patient_list()
            messagebox.showinfo("Success", "Medication added successfully")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(content, text="Save Medication", command=save)\
        .grid(row=len(fields)+1, column=0, columnspan=2, pady=16)


def delete_med_screen():
    if not require_patient(): return
    clear_content()

    ttk.Label(content, text="Delete Medication", font=("Segoe UI", 18), background=BG_CONTENT, foreground="white")\
        .grid(row=0, column=0, columnspan=2, pady=15, padx=10, sticky="w")

    ttk.Label(content, text="Medication Name", width=22, anchor="e", background=BG_CONTENT, foreground="white")\
        .grid(row=1, column=0, padx=10, pady=10, sticky="e")

    name_entry = ttk.Entry(content, width=30)
    name_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

    def delete():
        global meds_df
        try:
            meds_df = schedule.remove_medication(name_entry.get(), meds_df)
            messagebox.showinfo("Deleted", "Medication removed")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(content, text="Delete", command=delete)\
        .grid(row=2, column=0, columnspan=2, pady=16)


def daily_report_screen():
    if not require_patient(): return
    df = reminder.daily_report(meds_df, log_df, datetime.now().date())
    show_dataframe(df, f"Daily Report — {schedule.patient_name}")


def due_soon_screen():
    if not require_patient(): return
    df = reminder.due_soon(meds_df, datetime.now())
    show_dataframe(df, f"Due Soon — {schedule.patient_name}")


def log_dose_screen():
    if not require_patient(): return
    clear_content()

    ttk.Label(content, text="Log Dose", font=("Segoe UI", 18), background=BG_CONTENT, foreground="white")\
        .pack(pady=(14, 10), anchor="w", padx=12)

    form = tk.Frame(content, bg=BG_CONTENT)
    form.pack(padx=12, pady=10, anchor="w")

    ttk.Label(form, text="Medication Name", background=BG_CONTENT, foreground="white").grid(row=0, column=0, sticky="e", padx=8, pady=6)
    med_entry = ttk.Entry(form, width=28)
    med_entry.grid(row=0, column=1, padx=8, pady=6, sticky="w")

    ttk.Label(form, text="Time (HH:MM)", background=BG_CONTENT, foreground="white").grid(row=1, column=0, sticky="e", padx=8, pady=6)
    time_entry = ttk.Entry(form, width=28)
    time_entry.grid(row=1, column=1, padx=8, pady=6, sticky="w")

    ttk.Label(form, text="Status", background=BG_CONTENT, foreground="white").grid(row=2, column=0, sticky="e", padx=8, pady=6)
    status_combo = ttk.Combobox(form, values=["TAKEN", "MISSED"], state="readonly", width=25)
    status_combo.set("TAKEN")
    status_combo.grid(row=2, column=1, padx=8, pady=6, sticky="w")

    def save():
        global log_df
        try:
            log_df = log_dose(
                schedule.patient_name,
                med_entry.get(),
                time_entry.get(),
                status_combo.get(),
                meds_df,
                log_df
            )
            refresh_patient_list()
            messagebox.showinfo("Success", "Dose logged")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(content, text="Log", command=save).pack(pady=12, padx=12, anchor="w")


def patient_df_screen():
    if not require_patient(): return
    df = schedule.get_patient_meds_df(meds_df)
    show_dataframe(df, f"Patient Medications — {schedule.patient_name}")


def all_df_screen():
    # All patients meds, sorted nicely
    df = meds_df.copy()
    if not df.empty:
        df = df.sort_values(["patient_name", "medication"]).reset_index(drop=True)
    show_dataframe(df, "All Patients — Medications DataFrame")


# =========================
# 2-Column Menu (Fixes menu_frame error too)
# =========================
menu_frame = tk.Frame(sidebar, bg=BG_MAIN)
menu_frame.pack(fill="both", expand=False, padx=6, pady=(12, 6))

menu_frame.columnconfigure(0, weight=1)
menu_frame.columnconfigure(1, weight=1)

buttons = [
    ("➕ Add Medication", add_med_screen),
    ("🗑 Delete Medication", delete_med_screen),
    ("📅 Daily Report", daily_report_screen),
    ("⏰ Due Soon", due_soon_screen),
    ("✍ Log Dose", log_dose_screen),
    ("📋 Patient DataFrame", patient_df_screen),
    ("📊 All DataFrame", all_df_screen),
    ("❌ Exit", root.destroy),
]

for i, (text, cmd) in enumerate(buttons):
    r = i // 2
    c = i % 2
    btn = ttk.Button(menu_frame, text=text, command=cmd)
    btn.grid(row=r, column=c, padx=6, pady=6, sticky="ew")

# Initialize combobox list
refresh_patient_list()

root.mainloop()



