# PayPal Sandbox Setup Guide for RingBreaker

Follow this step-by-step guide to set up genuine PayPal Developer Sandbox credentials and webhook connectivity for RingBreaker.

---

## 1. Create a PayPal Developer Account & Sandbox App

1. Go to the [PayPal Developer Dashboard](https://developer.paypal.com/) and log in with your primary PayPal account (or sign up).
2. Navigate to **Dashboard > Apps & Credentials**.
3. Ensure the toggle at the top is set to **Sandbox** (never Live).
4. Click **Create App**:
   - **App Name:** `RingBreaker Sandbox`
   - **App Type:** `Merchant`
   - **Sandbox Business Account:** Select the default business account (or create one below).
5. Once created, you will see your sandbox credentials:
   - **Client ID**
   - **Secret** (click *Show* to reveal)
6. Copy these into your local `.env` file:
   ```ini
   PAYPAL_MODE=sandbox
   PAYPAL_CLIENT_ID=<your-sandbox-client-id>
   PAYPAL_CLIENT_SECRET=<your-sandbox-client-secret>
   ```

---

## 2. Verify Sandbox Accounts (Business & Personal Buyer)

Under **Dashboard > Testing Tools > Sandbox Accounts**:
1. You need two accounts:
   - **Business Account (Merchant):** Receives payments. This is the account tied to your Sandbox App.
   - **Personal Account (Buyer):** Emulates customers. Note down the generated email (e.g., `sb-buyer123@personal.example.com`) and password so you can approve test checkouts if using the buyer approval flow.
2. Under the Business Account profile, verify that **Orders v2** and **Negative Testing** (optional for triggering errors) are enabled.

---

## 3. Set Up Local Webhook Tunneling (ngrok or cloudflared)

PayPal Sandbox servers require a publicly accessible HTTPS URL to send webhook notifications to your local machine.

### Option A: Using `cloudflared` (Free, no account needed)
```bash
# In a separate terminal window:
cloudflared tunnel --url http://localhost:8000
```
Copy the generated public URL (e.g., `https://abc-123.trycloudflare.com`).

### Option B: Using `ngrok`
```bash
# In a separate terminal window:
ngrok http 8000
```
Copy the forwarding URL (e.g., `https://xyz.ngrok-free.app`).

---

## 4. Register the Webhook in PayPal Developer Dashboard

1. In **Dashboard > Apps & Credentials**, click on your `RingBreaker Sandbox` app.
2. Scroll down to **Webhooks** and click **Add Webhook**.
3. **Webhook URL:** Enter your public tunnel URL with the webhook path:
   ```
   https://<your-tunnel-domain>/api/webhooks/paypal
   ```
4. **Event Types:** Select:
   - `Payment authorization created` (`PAYMENT.AUTHORIZATION.CREATED`)
   - `Payment authorization voided` (`PAYMENT.AUTHORIZATION.VOIDED`)
   - `Payment capture completed` (`PAYMENT.CAPTURE.COMPLETED`)
   - `Checkout order approved` (`CHECKOUT.ORDER.APPROVED`)
5. Click **Save**.
6. Copy the generated **Webhook ID** (e.g., `9AB12345CD67890EF`) into your `.env`:
   ```ini
   PAYPAL_WEBHOOK_ID=<your-webhook-id>
   ```

---

## 5. Summary of `.env` Keys

Your `.env` file should contain:

```ini
# PayPal Developer Sandbox Credentials
PAYPAL_MODE=sandbox
PAYPAL_CLIENT_ID=your_actual_client_id_here
PAYPAL_CLIENT_SECRET=your_actual_client_secret_here
PAYPAL_WEBHOOK_ID=your_actual_webhook_id_here

# Operating Thresholds (Tuned on Validation Set)
THRESHOLD_LOW_CAPTURE=0.40
THRESHOLD_HIGH_VOID=0.60

# Database & Server
DATABASE_PATH=ringbreaker.db
PORT=8000
HOST=127.0.0.1
DEBUG=True

# LLM Investigator (Optional for Live Gemini / OpenAI)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> **Security Reminder:** Never commit `.env` or share client secrets. `.gitignore` ensures `.env` remains strictly local.
