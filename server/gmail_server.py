from mcp.server.fastmcp import FastMCP
import os
import pickle
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

# Initialize FastMCP server
mcp = FastMCP("gmail_tools")

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

class MessageHeader(BaseModel):
    name: str
    value: str

class MessageBody(BaseModel):
    data: Optional[str] = None
    attachmentId: Optional[str] = None
    size: Optional[int] = None

class MessagePart(BaseModel):
    mimeType: str
    body: MessageBody
    partId: Optional[str] = None
    filename: Optional[str] = None
    headers: Optional[List[MessageHeader]] = None
    parts: Optional[List['MessagePart']] = None

class GmailMessage(BaseModel):
    id: str
    threadId: str
    snippet: Optional[str] = None
    historyId: Optional[str] = None
    internalDate: Optional[str] = None
    payload: Optional[MessagePart] = None
    sizeEstimate: Optional[int] = None
    raw: Optional[str] = None
    labelIds: Optional[List[str]] = None

def authenticate_gmail():
    """Authenticates with Gmail API and returns service object"""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("gmail_creds.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("gmail", "v1", credentials=creds)
    return service

service = authenticate_gmail()

def search_emails_impl(labels: List[str]=["INBOX"], query: str=None, max_results:int=None):
    results = (
        service.users()
        .messages()
        .list(userId="me", labelIds=labels, q=query, maxResults=max_results)
        .execute()
    )
    return [GmailMessage(**message) for message in results.get("messages", [])]

def get_email_impl(message_id: str, format: str = None):
    result = service.users().messages().get(userId="me", id=message_id, format=format).execute()
    return GmailMessage(**result)


@mcp.tool()
async def get_email_count(label: str = "INBOX", query: str = "is:unread") -> str:
    """Gets count of emails matching specified criteria in the user's Gmail account.

    This tool connects to Gmail API with the user's credentials and counts emails
    that match the given label and search query. If authentication is required,
    a browser window will open to complete the OAuth flow.

    Args:
        label: Gmail label identifier to search within. Default is 'INBOX'.
              Common values include:
              - 'INBOX': Primary inbox
              - 'SENT': Sent messages
              - 'DRAFT': Draft messages
              - 'SPAM': Spam/junk messages
              - 'TRASH': Deleted messages
              - Any custom label ID (use get_labels() to see all available labels)

        query: Gmail search query string. Default is 'is:unread'.
               Common search operators include:
               - 'is:unread': Unread messages
               - 'is:read': Read messages
               - 'from:example@gmail.com': Messages from specific sender
               - 'to:example@gmail.com': Messages to specific recipient
               - 'subject:hello': Messages with specific subject
               - 'has:attachment': Messages with attachments
               - 'newer_than:2d': Messages newer than 2 days
               - 'older_than:1w': Messages older than 1 week
               Multiple operators can be combined, e.g., 'is:unread has:attachment'

    Returns:
        A formatted string with the count of emails matching the criteria.
    """
    messages = search_emails_impl(labels=[label], query=query)
    count = len(messages)

    return f"""
Email Count Summary:
Found {count} email(s) matching your criteria
Label: {label}
Search query: {query}
"""


@mcp.tool()
async def get_labels() -> str:
    """Fetches all available Gmail labels for the authenticated user.

    This tool connects to the Gmail API using the user's credentials and retrieves
    all labels (both system labels and user-created labels) available in their account.
    If authentication is required, a browser window will open to complete the OAuth flow.

    Labels are used to categorize emails in Gmail and can be used with other tools
    to filter or search for specific messages.

    Args:
        None

    Returns:
        A formatted string listing all available Gmail labels.
    """
    results = service.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])

    if not labels:
        return "No labels found in your Gmail account."

    system_labels = []
    user_labels = []

    for label in labels:
        label_id = label["id"]
        label_name = label["name"]

        if label_id.isupper():  # System labels typically have uppercase IDs
            system_labels.append(f"{label_name} (ID: {label_id})")
        else:
            user_labels.append(f"{label_name} (ID: {label_id})")

    result = "Gmail Labels:\n\n"

    if system_labels:
        result += "System Labels:\n" + "\n".join(system_labels)

    if user_labels:
        if system_labels:
            result += "\n\n"
        result += "User Labels:\n" + "\n".join(user_labels)

    return result


@mcp.tool()
async def get_email_content(message_id: str) -> str:
    """Fetches the complete content of a specific email by its ID.

    This tool connects to the Gmail API using the authenticated user's credentials
    and retrieves the full content of a specific email, including headers, body text,
    HTML content if available, and information about attachments.
    If authentication is required, a browser window will open to complete the OAuth flow.

    Args:
        message_id: The unique Gmail message ID of the email to retrieve.
                   This ID can be obtained from the 'id' field returned by
                   the get_unread_emails tool.

    Returns:
        A formatted string containing the complete email information.
        You shouldn't expose any technical details about the email.
    """

    # Get the full message details
    msg = get_email_impl(message_id, format="full")

    # Extract basic message info
    email_data = {
        "id": msg.id,
        "thread_id": msg.threadId,
        "label_ids": msg.labelIds,
        "snippet": msg.snippet,
        "subject": "",
        "sender": "",
        "recipient": "",
        "date": "",
        "plain_text": "",
        "html": "",
        "attachments": [],
    }

    # Process headers
    headers = msg.payload.headers if msg.payload else []
    for header in headers:
        name = header.name.lower()
        email_data[name] = header.value

    # Process message parts recursively to extract content and attachments
    def process_parts(parts: List[MessagePart], email_data: Dict[str, Any]):
        if not parts:
            return

        for part in parts:
            mime_type = part.mimeType

            # Handle attachments
            if part.filename:
                attachment = {
                    "filename": part.filename,
                    "mime_type": mime_type,
                    "size": part.body.size,
                    "attachment_id": part.body.attachmentId,
                }
                email_data["attachments"].append(attachment)

            # Handle message content
            elif mime_type == "text/plain":
                data = part.body.data
                if data:
                    text = base64.urlsafe_b64decode(data).decode("utf-8")
                    email_data["plain_text"] = text

            elif mime_type == "text/html":
                data = part.body.data
                if data:
                    html = base64.urlsafe_b64decode(data).decode("utf-8")
                    email_data["html"] = html

            # Handle nested parts
            if part.parts:
                process_parts(part.parts, email_data)

    # Start processing from the top level
    payload = msg.payload

    # Handle single-part messages
    if payload: 
        mime_type = payload.mimeType
        data = payload.body.data
        if data:
            content = base64.urlsafe_b64decode(data).decode("utf-8")
            if mime_type == "text/plain":
                email_data["plain_text"] = content
            elif mime_type == "text/html":
                email_data["html"] = content

    # Handle multi-part messages
    if payload.parts:
        process_parts(payload.parts, email_data)

    # Format attachment information
    attachment_str = ""
    if email_data["attachments"]:
        attachment_list = []
        for i, att in enumerate(email_data["attachments"], 1):
            attachment_list.append(
                f"{i}. {att['filename']} ({att['mime_type']}, {att['size']} bytes)"
            )
        attachment_str = "Attachments:\n" + "\n".join(attachment_list)
    else:
        attachment_str = "Attachments: None"

    # Format the labels
    labels_str = (
        "Labels: " + ", ".join(email_data["label_ids"])
        if email_data["label_ids"]
        else "Labels: None"
    )

    # Create content preview
    content_preview = (
        email_data["plain_text"][:500] + "..."
        if len(email_data["plain_text"]) > 500
        else email_data["plain_text"]
    )

    email_content = f"""
Email Details:

Message ID: {email_data['id']}
Thread ID: {email_data['thread_id']}
{labels_str}

From: {email_data['sender'] or 'Unknown'}
To: {email_data['recipient'] or 'Unknown'}
Subject: {email_data['subject'] or 'No Subject'}
Date: {email_data['date'] or 'Unknown'}

{attachment_str}

Content Preview:
{content_preview or 'No text content available'}
"""

    return email_content


@mcp.tool()
async def mark_email_as_read(message_id: str) -> str:
    """Marks a specific email as read.

    This tool connects to the Gmail API using the authenticated user's credentials
    and modifies an email's labels to remove the UNREAD label, effectively marking
    it as read. If authentication is required, a browser window will open to complete
    the OAuth flow.
    You shouldn't expose any technical details about the email to users, eg the message id, thread id, etc.

    Args:
        message_id: The unique Gmail message ID of the email to mark as read.
                   This ID can be obtained from the 'id' field returned by
                   the get_unread_emails tool.

    Returns:
        A formatted string indicating whether the operation was successful.
    """

    # Create the label modification request
    label_mod = {"removeLabelIds": ["UNREAD"], "addLabelIds": []}

    # Execute the modification
    result = (
        service.users()
        .messages()
        .modify(userId="me", id=message_id, body=label_mod)
        .execute()
    )

    success = "labelIds" in result and "UNREAD" not in result["labelIds"]

    if success:
        return f"Email (ID: {message_id}) was successfully marked as read."
    else:
        return f"Failed to mark email (ID: {message_id}) as read. It may already be read or doesn't exist."


@mcp.tool()
async def search_emails(
    label: str = "INBOX", query: str = "", max_results: int = 10
) -> str:
    """Searches for emails matching specific criteria in the user's Gmail account.

    This tool combines the functionality of get_unread_emails and get_email_count,
    but with more flexibility. It allows searching within any label using any valid
    Gmail search query, and returns detailed information about the matching messages.

    Args:
        label: Gmail label identifier to search within. Default is 'INBOX'.
              Common values include 'INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH',
              or any custom label ID (use get_labels() to see all available labels).

        query: Gmail search query string. Default is '' (all messages in the label).
               Examples:
               - 'is:unread': Unread messages
               - 'from:example@gmail.com': Messages from specific sender
               - 'subject:hello': Messages with specific subject
               - 'has:attachment': Messages with attachments
               - 'newer_than:2d': Messages newer than 2 days
               - 'older_than:1w': Messages older than 1 week
               - Multiple operators can be combined (e.g., 'is:unread has:attachment')

        max_results: Maximum number of messages to return. Default is 10.
                    Higher values may result in slower response times.

    Returns:
        A formatted string containing information about the matching emails,
        or a message indicating that no matching emails were found.
        You shouldn't expose any technical details about the email, eg the message id, thread id, etc.
    """
    messages = search_emails_impl(labels=[label], query=query, max_results=max_results)

    if not messages:
        return f"No emails found matching your criteria.\nLabel: {label}\nQuery: {query or '(all)'}"

    email_parts = []
    total_count = len(messages)

    for message in messages:
        full_msg = get_email_impl(message.id)
        payload = full_msg.payload
        headers = payload.headers if payload else []
        subject = ""
        sender = ""
        date = ""

        for header in headers:
            if header.name == "Subject":
                subject = header.value
            elif header.name == "From":
                sender = header.value
            elif header.name == "Date":
                date = header.value

        # Get labels for this message
        label_ids = full_msg.labelIds
        label_status = []
        if "UNREAD" in label_ids:
            label_status.append("Unread")
        if "STARRED" in label_ids:
            label_status.append("Starred")
        if "IMPORTANT" in label_ids:
            label_status.append("Important")

        status_str = (
            f"Status: {', '.join(label_status)}" if label_status else "Status: Read"
        )

        email_info = f"""
Email ID: {full_msg.id}
From: {sender or 'Unknown sender'}
Date: {date or 'Unknown date'}
Subject: {subject or 'No subject'}
{status_str}
Preview: {full_msg.snippet or 'No preview available'}
"""
        email_parts.append(email_info)

    result = f"Found {total_count} email(s) matching your criteria (showing {len(messages)})\n"
    result += f"Label: {label}\nQuery: {query or '(all)'}\n\n"
    result += "\n---\n".join(email_parts)

    return result


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
