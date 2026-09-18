"""
Report generator: CSV export + PDF reports (daily + monthly).
Uses reportlab for PDF generation.
"""
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

from .config import REPORTS_DIR


def generate_daily_csv(records: dict, session_name: str, output_dir: str = None) -> str:
    """Export daily attendance as CSV."""
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"daily_{session_name}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Roll", "Entry Time", "Exit Time", "Dwell (min)", "Status", "Visits", "Detections", "Spoof Flags"])
        for roll, rec in sorted(records.items(), key=lambda x: x[1].entry_time):
            writer.writerow([
                rec.name, rec.roll, rec.entry_time, rec.exit_time,
                round(rec.duration_sec / 60, 1),
                rec.status, len(rec.visits),
                rec.detection_count, rec.spoof_flags,
            ])

    print(f"[REPORT] Daily CSV: {path}")
    return str(path)


def generate_daily_pdf(records: dict, alerts: list, session_name: str, output_dir: str = None) -> str:
    """Generate a PDF attendance report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"daily_{session_name}.pdf"

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Daily Attendance Report", styles["Title"]))
    elements.append(Paragraph(f"Session: {session_name}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Summary stats
    # Counted = met the dwell requirement. Someone merely glimpsed is "brief" and
    # is reported apart from attendance rather than inflating it.
    counted = sum(1 for r in records.values() if r.status != "brief")
    brief = len(records) - counted
    spoofs = sum(1 for r in records.values() if r.spoof_flags > 0)
    elements.append(Paragraph(f"Seen: {len(records)} | Attendance counted: {counted} | Too brief to count: {brief} | Spoofs Detected: {spoofs}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Attendance table
    data = [["Name", "Roll", "Entry", "Exit", "Dwell (min)", "Visits", "Status"]]
    for roll, rec in sorted(records.items(), key=lambda x: x[1].entry_time):
        data.append([
            rec.name, rec.roll, rec.entry_time, rec.exit_time,
            str(round(rec.duration_sec / 60, 1)),
            str(len(rec.visits)), rec.status,
        ])

    if len(data) > 1:
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ]))
        elements.append(table)

    # Alerts section
    if alerts:
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Anomaly Alerts", styles["Heading2"]))
        alert_data = [["Time", "Type", "Details"]]
        for a in alerts[-30:]:  # last 30
            alert_data.append([a.timestamp, a.alert_type, a.details[:60]])

        if len(alert_data) > 1:
            alert_table = Table(alert_data, repeatRows=1)
            alert_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc2626")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(alert_table)

    doc.build(elements)
    print(f"[REPORT] Daily PDF: {path}")
    return str(path)


def generate_monthly_report(log_dirs: list[str], output_dir: str = None) -> str:
    """Aggregate multiple daily logs into a monthly summary PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    import json

    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate data
    monthly_data = {}  # roll → {name, total_present, total_sessions, ...}

    for log_dir in log_dirs:
        log_file = Path(log_dir) / "attendance.json"
        if not log_file.exists():
            continue
        with open(log_file) as f:
            data = json.load(f)
        for p in data.get("persons", []):
            roll = p["roll"]
            if roll not in monthly_data:
                monthly_data[roll] = {"name": p["name"], "roll": roll, "present": 0, "total": 0, "spoofs": 0}
            monthly_data[roll]["total"] += 1
            if p.get("present"):
                monthly_data[roll]["present"] += 1
            monthly_data[roll]["spoofs"] += p.get("spoofs", 0)

    month_str = datetime.now().strftime("%Y-%m")
    path = out_dir / f"monthly_{month_str}.pdf"

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Monthly Attendance Report - {month_str}", styles["Title"]))
    elements.append(Paragraph(f"Sessions analyzed: {len(log_dirs)}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [["Name", "Roll", "Sessions Present", "Total Sessions", "Attendance %", "Spoofs"]]
    for roll, m in sorted(monthly_data.items(), key=lambda x: x[1]["name"]):
        pct = round(100 * m["present"] / max(m["total"], 1), 1)
        data.append([m["name"], m["roll"], str(m["present"]), str(m["total"]), f"{pct}%", str(m["spoofs"])])

    if len(data) > 1:
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eff6ff")]),
        ]))
        elements.append(table)

    doc.build(elements)
    print(f"[REPORT] Monthly PDF: {path}")
    return str(path)
