import email
import hashlib
import mailbox
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document

from src.utils.embedding_service import EmbeddingService

# Matches quoted reply lines: "> text"
QUOTED_LINE_RE = re.compile(r"^\s*>+.*$", re.MULTILINE)
# Matches common reply headers — cut everything below
REPLY_HEADER_RE = re.compile(
    r"^(On\s.+wrote:|From:.+\nSent:.+\nTo:.+|_{3,}|-{3,})",
    re.MULTILINE | re.IGNORECASE,
)


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def _strip_quoted_content(body: str) -> str:
    body = QUOTED_LINE_RE.sub("", body)
    match = REPLY_HEADER_RE.search(body)
    if match:
        body = body[: match.start()]
    return body.strip()


def _extract_thread_id(msg: email.message.Message) -> str:
    references = msg.get("References", "").strip()
    if references:
        return references.split()[0].strip("<>")
    in_reply_to = msg.get("In-Reply-To", "").strip()
    if in_reply_to:
        return in_reply_to.strip("<>")
    return msg.get("Message-ID", "").strip().strip("<>")


def _make_email_id(message_id: str) -> str:
    return hashlib.sha256(message_id.encode()).hexdigest()


def _get_plain_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
                charset = part.get_content_charset() or "utf-8"
                text = part.get_payload(decode=True).decode(charset, errors="replace")
                return _strip_html(text)
    else:
        charset = msg.get_content_charset() or "utf-8"
        text = msg.get_payload(decode=True).decode(charset, errors="replace")
        return _strip_html(text)
    return ""


def _parse_message(msg: email.message.Message, file_name: str) -> Document | None:
    sender = msg.get("From", "")
    subject = msg.get("Subject", "")
    message_id = msg.get("Message-ID", "").strip("<>") or f"{file_name}-{sender}-{subject}"

    try:
        email_date = parsedate_to_datetime(msg.get("Date", "")).isoformat()
    except Exception:
        email_date = datetime.now(timezone.utc).isoformat()

    thread_id = _extract_thread_id(msg)
    clean_body = _strip_quoted_content(_get_plain_body(msg))

    if not clean_body:
        return None

    text = f"{subject}\n\n{clean_body}" if subject else clean_body

    return Document(
        page_content=text,
        metadata={
            "file_name": file_name,
            "email_id": _make_email_id(message_id),
            "thread_id": thread_id,
            "sender": sender,
            "email_date": email_date,
            "subject": subject,
        },
    )


class EmailProcessor:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def process(self, file_path: str) -> tuple[list[Document], list[list[float]]]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".eml":
            return self._process_eml(file_path)
        elif suffix == ".mbox":
            return self._process_mbox(file_path)
        else:
            raise ValueError(f"Unsupported email format: {suffix}. Expected .eml or .mbox")

    def _process_eml(self, file_path: str) -> tuple[list[Document], list[list[float]]]:
        with open(file_path, "rb") as f:
            msg = email.message_from_bytes(f.read())

        doc = _parse_message(msg, Path(file_path).name)
        if doc is None:
            return [], []

        return [doc], [self.embedding_service.encode_one(doc.page_content)]

    def _process_mbox(self, file_path: str) -> tuple[list[Document], list[list[float]]]:
        mbox = mailbox.mbox(file_path)
        documents = [
            doc for msg in mbox
            if (doc := _parse_message(msg, Path(file_path).name)) is not None
        ]

        if not documents:
            return [], []

        embeddings = self.embedding_service.encode([doc.page_content for doc in documents])
        return documents, embeddings
