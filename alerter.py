"""
NeuroAlert — Mobile Alert System
Sends real-time seizure alerts via:
  - Telegram (recommended — instant phone notification)
  - Email (Gmail)
  - Browser sound alert
"""

import requests
import time
import threading
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR CREDENTIALS HERE
# ══════════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = "8705369362:AAE_7VaHTfmFChut3nacSffa9_7bYutM5HQ"       # from @BotFather
TELEGRAM_CHAT_ID = "6645660919"         # from getUpdates URL

# Optional email alert (Gmail)
EMAIL_SENDER   = "abcduem2@gmail.com"
EMAIL_PASSWORD = "your_app_password"            # Gmail App Password (not login password)
EMAIL_RECEIVER = "sarkar.anju2004@gmail.com"

# ══════════════════════════════════════════════════════════════════════════════


class NeuroAlerter:
    """
    Sends multi-channel alerts when seizure risk exceeds threshold.
    Uses cooldown to avoid spamming — one alert per 60 seconds max.
    """

    COOLDOWN_SEC = 60   # minimum seconds between alerts

    def __init__(self):
        self.last_alert_time = 0
        self.alert_count     = 0
        self._lock           = threading.Lock()

    def should_alert(self) -> bool:
        return (time.time() - self.last_alert_time) > self.COOLDOWN_SEC

    def send_all(self, risk: float, channel_risks: dict = None):
        """
        Call this whenever risk crosses the threshold.
        Handles cooldown internally — safe to call every window.
        """
        with self._lock:
            if not self.should_alert():
                return

            self.last_alert_time = time.time()
            self.alert_count    += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Build message
            msg = self._build_message(risk, timestamp, channel_risks)

            # Send in background threads so dashboard doesn't freeze
            threading.Thread(target=self.send_telegram, args=(msg,), daemon=True).start()
            threading.Thread(target=self.send_email,    args=(risk, msg), daemon=True).start()

            print(f"[NeuroAlert] 🚨 Alert #{self.alert_count} sent at {timestamp} — Risk: {risk:.1f}%")

    def _build_message(self, risk: float, timestamp: str, channel_risks: dict = None) -> str:
        level = "🔴 CRITICAL" if risk >= 80 else "🟠 HIGH"
        msg   = (
            f"🧠 *NeuroAlert — Seizure Warning*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{level} Risk detected\n\n"
            f"📊 Risk score: *{risk:.1f}%*\n"
            f"🕐 Time: {timestamp}\n"
            f"⚡ Pre-ictal pattern identified\n\n"
            f"⚠️ *Seizure may occur within 30 seconds.*\n"
            f"Please ensure patient safety immediately.\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_NeuroAlert Early Detection System_"
        )
        if channel_risks:
            top = sorted(channel_risks.items(), key=lambda x: x[1], reverse=True)[:3]
            ch_text = "\n".join([f"  • {ch}: {v:.0f}%" for ch, v in top])
            msg += f"\n\n🔬 Top active channels:\n{ch_text}"
        return msg

    # ── Telegram ──────────────────────────────────────────────────────────────

    def send_telegram(self, message: str) -> bool:
        """Send a Telegram message to your phone."""
        if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("  [Telegram] Skipped — token not configured")
            return False
        try:
            url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       message,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, data=data, timeout=5)
            if resp.status_code == 200:
                print("  [Telegram] ✓ Alert sent to phone")
                return True
            else:
                print(f"  [Telegram] ✗ Failed: {resp.text}")
                return False
        except Exception as e:
            print(f"  [Telegram] ✗ Error: {e}")
            return False

    def send_telegram_image(self, fig_path: str, caption: str = "EEG Snapshot"):
        """Send an EEG screenshot along with the alert."""
        if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(fig_path, "rb") as photo:
                requests.post(url, data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                }, files={"photo": photo}, timeout=10)
            print("  [Telegram] ✓ EEG image sent")
        except Exception as e:
            print(f"  [Telegram] ✗ Image error: {e}")

    # ── Email ─────────────────────────────────────────────────────────────────

    def send_email(self, risk: float, body: str) -> bool:
        """Send an email alert via Gmail."""
        if EMAIL_SENDER == "your_gmail@gmail.com":
            print("  [Email] Skipped — email not configured")
            return False
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg            = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 NeuroAlert — Seizure Risk {risk:.0f}%"
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER

            html = f"""
            <html><body style="font-family:sans-serif;background:#0a0a0f;color:#e0e0f0;padding:24px">
              <div style="max-width:480px;margin:auto;background:#12121a;border-radius:12px;
                          border:1px solid #e74c3c;padding:24px">
                <h2 style="color:#e74c3c;margin-top:0">🧠 NeuroAlert</h2>
                <p style="font-size:1.1rem">Seizure risk detected: 
                   <strong style="color:#e74c3c;font-size:1.4rem">{risk:.1f}%</strong></p>
                <p>Pre-ictal EEG pattern identified.<br>
                   <strong>Seizure may occur within 30 seconds.</strong></p>
                <p style="color:#888;font-size:0.85rem">
                   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr style="border-color:#1e1e2e">
                <p style="color:#666;font-size:0.8rem">
                   NeuroAlert Early Seizure Detection System</p>
              </div>
            </body></html>
            """

            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

            print("  [Email] ✓ Alert email sent")
            return True
        except Exception as e:
            print(f"  [Email] ✗ Error: {e}")
            return False

    def test(self):
        """Send a test alert to verify everything is working."""
        print("\nSending test alert...")
        self.last_alert_time = 0   # bypass cooldown for test
        self.send_all(risk=75.0)
        time.sleep(3)
        print("Test complete. Check your phone/email.\n")


# Singleton instance — import this in dashboard.py
alerter = NeuroAlerter()


if __name__ == "__main__":
    alerter.test()
