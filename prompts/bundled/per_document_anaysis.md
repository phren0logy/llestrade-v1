## Document Summary Instructions

Please analyze the document content, wrapped in <document-content> tags, and provide a comprehensive summary that includes:

- Key facts and information about the subject
- Significant events and dates mentioned
- Family and romantic relationships
- Early childhood history
- Educational history
- Employment history
- Military career history
- Legal issues or encounters with law enforcement
- Substance use and treatment history
- Medical and psychiatric history
- Any notable statements or quotes
- Notable patterns of behavior
- Adverse life events
- A timeline of events in a markdown table format with columns for Date, Event, and Significance

### Citation requirements:

- Add citation markers for factual claims using the exact format `[CIT:ev_<id>]`
- Use only IDs provided in the Citation Evidence Ledger
- Never invent citation IDs. If support is unavailable, state that the evidence is unavailable
- Place one or more markers at the end of each claim.
- Never invent IDs. If no supporting evidence exists in the ledger, explicitly say that evidence is unavailable.

## Timeline Instructions

- Using the subject's date of birth ({subject_dob}), calculate the subject's age at each event when relevant
- Create a timeline of events in a markdown table format with columns for Date, Age, Event, and Significance, and [CIT]:ev_<id>, where age refers to the subject's age
- When exact dates aren't provided, estimate years when possible and mark them with "(est.)"
- Organize the timeline chronologically with the most recent events at the bottom
- If there are multiple events with the same date and significance, list them in the order they occurred (if known)

Keep your analysis focused on factual information directly stated in the document.

Before finalizing results, review for accuracy, paying close attention to exact quotes and the correctness of citation markers.

## Document Content

<document-content>
{document_content}
</document-content>
