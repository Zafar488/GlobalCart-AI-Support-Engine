import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ================== CONFIG (Kept Exactly as Requested) ==================
# ================== CONFIG ==================
EMAIL_CONFIG = {
    "sender_email": "zafarullah1385@gmail.com",   
    "app_password": "chwvkieycjzxebmp",  # 🔴 FIX: Spaces hata diye hain
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Executes Real SMTP Email Transmission"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender_email"]
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the AI generated text body
        msg.attach(MIMEText(body, 'plain'))

        # Connect to server
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls() # Secure connection
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["app_password"])
        server.send_message(msg)
        server.quit()

        print(f"✅ [SMTP] Email successfully relayed to {to_email} at {datetime.now()}")
        return True
    
    except Exception as e:
        print(f"❌ [SMTP] Delivery failure: {e}")
        return False

# Test Execution
if __name__ == "__main__":
    test_body = "This is an automated system check from your AI Architecture.\n\nAll systems operational.\nBest regards,\nEngineering Team"
    send_email("zafarullahbittani79@gmail.com", "AI Architecture Systems Check", test_body)