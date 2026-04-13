import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Altumatim Search", layout="wide")
st.title("Altumatim Legal Document Search")

with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader(
        "PDF, .eml, .mbox or .json",
        type=["pdf", "eml", "mbox", "json"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}…"):
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                )
            if resp.ok:
                data = resp.json()
                if "error" in data:
                    st.error(data["error"])
                elif data.get("status") == "already_indexed":
                    st.info(f"{f.name} already indexed.")
                elif "chunks_indexed" in data:
                    st.success(f"{f.name} — {data['chunks_indexed']} chunks indexed (v{data['version']})")
                else:
                    st.success(f"{f.name} — {data.get('emails_indexed', 0)} emails indexed")
            else:
                st.error(f"Upload failed: {resp.text}")


query = st.text_input("Ask a question or search for a clause", placeholder="e.g. termination clause notice period")
doc_search_filter = st.selectbox("Seach in following document types", ["All", "PDF", "Email", "JSON"], index=0)




if query:
    st.subheader("Results")
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.selectbox("Results", [3, 5, 10], index=1)
    with col2:
        doc_type_filter = st.selectbox("Document type", ["All", "PDF", "Email", "JSON"], index=0)
    with st.spinner("Searching…"):
        resp = requests.post(f"{API_URL}/query", params={"query": query, "size": top_k})

    if not resp.ok:
        st.error(f"Search failed: {resp.text}")
    else:
        results = resp.json().get("results", [])

        # client-side doc_type filter
        if doc_type_filter != "All":
            results = [r for r in results if r.get("doc_type", "").lower() == doc_type_filter.lower()]

        if not results:
            st.info("No results found.")
        else:
            st.markdown(f"**{len(results)} result(s)**")
            for i, hit in enumerate(results, 1):
                doc_type = hit.get("doc_type", "unknown")
                file_name = hit.get("file_name", "—")
                content = hit.get("content", "")

                if doc_type == "email":
                    subject = hit.get("subject", "—")
                    sender = hit.get("sender", "—")
                    date = hit.get("email_date", "—")[:10] if hit.get("email_date") else "—"
                    label = f"Email: {subject}  ·  {sender}  ·  {date}"
                else:
                    page = hit.get("page_number", "—")
                    label = f"PDF: {file_name}  ·  page {page}"

                with st.expander(f"#{i}  {label}"):
                    st.caption(f"Source: `{file_name}`")
                    st.write(content)
