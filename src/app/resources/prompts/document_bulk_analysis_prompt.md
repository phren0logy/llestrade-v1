# Document Analysis Task

## Document Information

- **Subject Name**: {subject_name}
- **Date of Birth**: {subject_dob}
- **Original Document**: (The {document_name} is a markdown file that was converted from a PDF. Change the extension from .md to .pdf)

## Case Background

{case_info}

## Combined Summary Instructions

Please analyze the document content, wrapped in "document-content" tags, and provide a comprehensive summary that includes:

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

Include citation markers for every factual claim using only labels from the Generated Citation Appendix.
Use the exact inline format defined there, such as `[C1]`, and place one or more markers at the end of each claim.
Never invent citation labels. If no supporting evidence exists in the appendix, explicitly say that evidence is unavailable.

## Timeline Instructions

- Using the subject's date of birth ({subject_dob}), calculate the subject's age at each event when relevant
- Create a timeline of events in a markdown table format with columns for Date, Age, Event, and Significance, and Page Number, where age refer's to the subject's age
- When exact dates aren't provided, estimate years when possible and mark them with "(est.)"
- Organize the timeline chronologically with the most recent events at the bottom
- If there are multiple events on the same date, list them in the order they occurred
- If there are multiple events with the same date and significance, list them in the order they occurred

Keep your analysis focused on factual information directly stated in the document.

Before finalizing results, do a review for accuracy, with attention to exact quotes and citation marker correctness.

## Document Content

<document-content>
{document_content}
</document-content>
