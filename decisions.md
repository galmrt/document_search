04/09/2026
Understand and outline the requirements. 
Deliverable requirements:
    * File upload:
        * Excpected formats: native vs. scanned PDFs, email plain text with metadata, JSON or XML
    * Chat: 
        * Ask question in English -> get source if exist
    * Retrieval of relevant precedent/clause language
    
Technical notes:
    * ~ 2 million documents:
    * Mostly PDF contracts:
    * Scanned PDFs (mostly)
    * Some native PDF 
    * Memos (word docs)
    * Regulatory filings (RegTrack)
    * XML 
    * JSON (exports from compliance – can be 200MB+ in size)
    * Spreadsheets (minimal)
    * 
    * Email archives (800K):
    * Plain text 
    * Metadata:
    * Sender
    * Date
    * Subject 
    * Thread ID 
    * Speed: aim for < 5 sec
    * Accuracy is crucial. Better to not find the document that to provide hallucinated information 
    * Access control (not needed for implementation but an architecture to implement late) 
    * Semantic search 
    * Document versioning handling (updating docs)
    * Provide area/field specific search capability (search only on emails/docs)
    * Handle ZIP file bundles 



04/09/2026
