import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'gmail_creds.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def get_unread_emails(service, max_results=10):
    """Fetches unread messages from the user's inbox."""
    results = service.users().messages().list(userId='me',
                                              labelIds=['INBOX'],
                                              q='is:unread').execute()
    messages = results.get('messages', [])

    if not messages:
        print("No unread messages found.")
        return []

    print(f"Found {len(messages)} unread messages. Showing top {max_results}:")
    unread_emails = []

    for message in messages[:max_results]:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        subject = ''
        sender = ''

        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']

        email_data = {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'snippet': msg.get('snippet', '')
        }
        unread_emails.append(email_data)
        print(f"\nFrom: {sender}\nSubject: {subject}\nSnippet: {msg.get('snippet', '')}")

    return unread_emails

if __name__ == '__main__':
    service = authenticate_gmail()
    get_unread_emails(service)