# test_email_send.py
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="my_agent/.env")

from my_agent.tools import send_email

result = send_email.invoke({
    "recipient": "racheldissertation2026@gmail.com",
    "subject": "HopeBot Test Email",
    "body": "<p>This is a test email from HopeBot.</p>"
})

print(f"Result: {result}")