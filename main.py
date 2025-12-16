from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# =========================
# Colored Output
# =========================

def print_blue(msg):
    print("\033[94m" + str(msg) + "\033[0m")

def print_red(msg):
    print("\033[91m" + str(msg) + "\033[0m")

# =========================
# Parsing + Input Validation
# =========================

def parse_time_hhmm(value):
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except:
        raise ValueError("Invalid time format. Use HH:MM (e.g., 08:30).")

def parse_date_flexible(value):
    patterns = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]
    for p in patterns:
        try:
            return datetime.strptime(value.strip(), p).date()
        except:
            pass
    raise ValueError("Invalid date format.")

def ask_nonempty(prompt):
    value = input(prompt).strip()
    if not value:
        raise ValueError("This field cannot be empty.")
    return value

def ask_float_positive(prompt):
    try:
        v = float(input(prompt).strip())
        if v <= 0:
            raise ValueError
        return v
    except:
        raise ValueError("Enter a valid positive number.")

def ask_int_min(prompt, min_value=1):
    try:
        v = int(input(prompt).strip())
        if v < min_value:
            raise ValueError
        return v
    except:
        raise ValueError(f"Enter an integer >= {min_value}")

# =========================
# OOP Classes
# =========================

class Medication:
    def __init__(self, name, dosage_mg, times_per_day, first_time, duration_days, start_date):
        self.name = name
        self.dosage_mg = dosage_mg
        self.times_per_day = times_per_day
        self.first_time = first_time
        self.duration_days = duration_days
        self.start_date = start_date
        self.taken_count = 0
        self.validate()

    def validate(self):
        if not self.name:
            raise ValueError("Medication name required.")
        if self.dosage_mg <= 0:
            raise ValueError("Dosage must be positive.")
        if self.times_per_day < 1:
            raise ValueError("Times per day must be >= 1.")
        if self.duration_days < 1:
            raise ValueError("Duration must be >= 1.")

    def total_doses(self):
        return self.times_per_day * self.duration_days

    def remaining_doses(self):
        return max(0, self.total_doses() - self.taken_count)

    def mark_taken(self):
        if self.remaining_doses() <= 0:
            raise ValueError("All doses already taken.")
        self.taken_count += 1

    def dose_times_for_day(self, day):
        end = self.start_date + timedelta(days=self.duration_days - 1)
        if day < self.start_date or day > end:
            return []

        interval = 24 / self.times_per_day
        base = datetime.combine(day, self.first_time)
        return [base + timedelta(hours=i * interval) for i in range(self.times_per_day)]

    def next_dose_datetime(self, now):
        for i in range(self.duration_days + 1):
            d = now.date() + timedelta(days=i)
            for t in self.dose_times_for_day(d):
                if t >= now:
                    return t
        return None

class MedicationSchedule:
    def __init__(self, patient_name):
        self.patient_name = patient_name
        self.items = []

    def add(self, med):
        if any(m.name.lower() == med.name.lower() for m in self.items):
            raise ValueError("Medication already exists.")
        self.items.append(med)

    def find(self, name):
        for m in self.items:
            if m.name.lower() == name.lower():
                return m
        return None

    def remove(self, name):
        before = len(self.items)
        self.items = [m for m in self.items if m.name.lower() != name.lower()]
        return before != len(self.items)

class Reminder:
    def __init__(self, schedule):
        self.schedule = schedule

    def daily_report(self, day):
        rows = []
        for m in self.schedule.items:
            for t in m.dose_times_for_day(day):
                rows.append({
                    "Medication": m.name,
                    "Dosage (mg)": m.dosage_mg,
                    "Dose Time": t.strftime("%H:%M"),
                    "Remaining Doses": m.remaining_doses()
                })
        df = pd.DataFrame(rows)
        return df.sort_values(by="Dose Time") if not df.empty else df

    def check_due(self, now, window_minutes=15):
        rows = []
        for m in self.schedule.items:
            nxt = m.next_dose_datetime(now)
            if nxt:
                minutes = int((np.datetime64(nxt) - np.datetime64(now)) / np.timedelta64(1, "m"))
                if 0 <= minutes <= window_minutes:
                    rows.append({
                        "Medication": m.name,
                        "Next Dose": nxt.strftime("%Y-%m-%d %H:%M"),
                        "Minutes Left": minutes
                    })
        return pd.DataFrame(rows)

# =========================
# DataFrames
# =========================

def build_patient_dataframe(schedule):
    return pd.DataFrame([{
        "Patient": schedule.patient_name,
        "Medication": m.name,
        "Dosage (mg)": m.dosage_mg,
        "Times/Day": m.times_per_day,
        "Start Date": m.start_date.strftime("%Y-%m-%d"),
        "Duration (days)": m.duration_days,
        "Taken": m.taken_count,
        "Remaining": m.remaining_doses()
    } for m in schedule.items])

def build_all_patients_dataframe(patients):
    frames = [build_patient_dataframe(s) for s in patients.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# =========================
# Visualization
# =========================

def plot_remaining_doses(schedule):
    if not schedule.items:
        print_red("No medications to visualize.")
        return

    names = [m.name for m in schedule.items]
    remaining = [m.remaining_doses() for m in schedule.items]

    plt.figure(figsize=(8,4))
    plt.bar(names, remaining)
    plt.title("Remaining Doses - " + schedule.patient_name)
    plt.xlabel("Medication")
    plt.ylabel("Remaining Doses")
    plt.tight_layout()
    plt.show()

# =========================
# Menu System
# =========================

patients = {}
current_patient = None

def setup_patient():
    global current_patient
    name = ask_nonempty("Enter patient name: ")
    if name not in patients:
        patients[name] = MedicationSchedule(name)
        print_blue("New patient created.")
    current_patient = name

def run_menu():
    setup_patient()
    while True:
        sch = patients[current_patient]
        reminder = Reminder(sch)

        print("\n1 Add medication")
        print("2 Delete medication")
        print("3 Mark dose taken (auto-remove if finished)")
        print("4 Daily report")
        print("5 Due soon")
        print("6 DataFrame (current patient)")
        print("7 DataFrame (all patients)")
        print("8 Visualization")
        print("9 Switch patient")
        print("0 Exit")

        c = input("Choose: ").strip()

        try:
            if c == "1":
                med = Medication(
                    ask_nonempty("Medication name: "),
                    ask_float_positive("Dosage (mg): "),
                    ask_int_min("Times per day: "),
                    parse_time_hhmm(ask_nonempty("First time HH:MM: ")),
                    ask_int_min("Duration days: "),
                    parse_date_flexible(ask_nonempty("Start date: "))
                )
                sch.add(med)
                print_blue("Medication added.")

            elif c == "2":
                name = ask_nonempty("Medication name: ")
                if sch.remove(name):
                    print_blue("Medication deleted.")
                else:
                    print_red("Medication not found.")

            elif c == "3":
                name = ask_nonempty("Medication name: ")
                m = sch.find(name)
                if not m:
                    print_red("Medication not found.")
                else:
                    before = m.remaining_doses()
                    m.mark_taken()
                    after = m.remaining_doses()
                    print_blue(f"Dose recorded. Remaining: {after} (was {before})")

                    # ✅ Auto-remove medication when all doses are taken
                    if m.remaining_doses() == 0:
                        sch.remove(m.name)
                        print_blue(f"{m.name} finished ✅ Removed from schedule.")

            elif c == "4":
                df = reminder.daily_report(datetime.now().date())
                display(df if not df.empty else "No data")

            elif c == "5":
                df = reminder.check_due(datetime.now())
                display(df if not df.empty else "No due medications")

            elif c == "6":
                dfp = build_patient_dataframe(sch)
                display(dfp if not dfp.empty else "No medications for this patient")

            elif c == "7":
                dfa = build_all_patients_dataframe(patients)
                display(dfa if not dfa.empty else "No patients / medications yet")

            elif c == "8":
                plot_remaining_doses(sch)

            elif c == "9":
                setup_patient()

            elif c == "0":
                print_blue("Bye 👋")
                break

            else:
                print_red("Invalid choice.")

        except Exception as e:
            print_red(e)

run_menu()
