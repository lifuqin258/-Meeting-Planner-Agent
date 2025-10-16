from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import pickle
from datetime import datetime, timedelta
from config import CREDENTIALS_PATH, SCOPES, USE_MOCK, ORGANIZER_EMAIL

def authenticate():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return creds

# ===== MOCK MODE =====
def mock_check_availability(emails, start_time, end_time):
    # 模拟：只有 organizer 有冲突
    conflicts = {}
    for email in emails:
        if email == ORGANIZER_EMAIL and "14:00" in start_time.isoformat():
            conflicts[email] = [start_time]
    return conflicts

def mock_create_event(summary, start_time, end_time, attendees):
    return {
        "id": "mock_event_123",
        "htmlLink": "https://calendar.google.com/mock",
        "start": {"dateTime": start_time.isoformat()},
        "end": {"dateTime": end_time.isoformat()}
    }

# ===== REAL MODE =====
def real_check_availability(emails, start_time, end_time):
    creds = authenticate()
    service = build("calendar", "v3", credentials=creds)
    body = {
        "timeMin": start_time.isoformat() + "Z",
        "timeMax": end_time.isoformat() + "Z",
        "items": [{"id": email} for email in emails]
    }
    result = service.freebusy().query(body=body).execute()
    conflicts = {}
    for email, cal_info in result["calendars"].items():
        if cal_info["busy"]:
            conflicts[email] = [busy["start"] for busy in cal_info["busy"]]
    return conflicts

def real_create_event(summary, start_time, end_time, attendees):
    creds = authenticate()
    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": summary,
        "start": {"dateTime": start_time.isoformat(), "timeZone": "Asia/Shanghai"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "Asia/Shanghai"},
        "attendees": [{"email": email} for email in attendees],
        "reminders": {"useDefault": False, "overrides": [{"method": "email", "minutes": 10}]}
    }
    event = service.events().insert(calendarId="primary", body=event).execute()
    return event

# ===== PUBLIC INTERFACE =====
def check_availability(emails, start_time, end_time):
    if USE_MOCK:
        return mock_check_availability(emails, start_time, end_time)
    else:
        return real_check_availability(emails, start_time, end_time)

def create_calendar_event(summary, start_time, end_time, attendees):
    if USE_MOCK:
        return mock_create_event(summary, start_time, end_time, attendees)
    else:
        return real_create_event(summary, start_time, end_time, attendees)
