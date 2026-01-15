import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Add parent directory to path to import config if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # REQUIRED for Community Promotion (Comments)
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
]
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"

def authenticate():
    print("🚀 Starting Manual YouTube Authentication...")
    
    if not os.path.exists(CLIENT_SECRET_FILE):
        print(f"❌ Error: {CLIENT_SECRET_FILE} not found!")
        print("Please download your OAuth 2.0 Client ID JSON from Google Cloud Console")
        print("and save it as 'client_secret.json' in the root directory.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    
    print("🌍 Opening browser for authentication...")
    print("👉 If browser doesn't open, copy the link below:")
    
    creds = flow.run_local_server(port=0)
    
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
        
    print(f"✅ Authentication successful! Token saved to {TOKEN_FILE}")
    print("You can now restart the bot or try uploading again.")

if __name__ == "__main__":
    # Ensure we are in the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    authenticate()
