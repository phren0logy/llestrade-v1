# Document Summary Task

Analyze the document content wrapped in `<document-content>` tags and produce a comprehensive factual summary of the record.

Address, where relevant:
- key facts about the subject
- significant events and dates
- family and relationship history
- early developmental history
- education and employment history
- military history
- legal history
- substance use and treatment history
- medical and psychiatric history
- notable statements or quotations
- adverse life events
- recurring behavioral patterns

Citation requirements:
- Add citation markers for factual claims using only labels from the Generated Citation Appendix.
- Use the exact inline format defined there, such as `[C1]`.
- Place one or more citation markers at the end of each supported claim.
- Never invent citation labels. If support is unavailable, say that evidence is unavailable.

Timeline requirements:
- Include a markdown timeline table with columns: `Date | Age | Event | Significance | Citations`.
- Use the subject's date of birth ({subject_dob}) to calculate age when relevant.
- When an exact date is unavailable, estimate only when the document reasonably supports it and mark the date with `(est.)`.
- Keep the timeline chronological with the earliest events first and the most recent events last.
- Each timeline row should include the citation label or labels that support that row.

Writing requirements:
- Keep the analysis focused on information directly supported by the document.
- Preserve exact quoted language when it is notable or clinically important.
- Review the output for accuracy, especially quotations, dates, and citation placement, before finalizing.

## Document Content

<document-content>
{document_content}
</document-content>
