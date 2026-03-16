## Instructions

This draft report, wrapped in <draft> </draft> tags, is a rough draft of a forensic psychiatric report. Perform the following steps in sequential order to improve the report:

1. Ignore the contents of block quotes. The errors belong to the source document being quoted. Never modify the contents of block quotes.

2. Persist direct quotes and exact language from the source document without changes. Refer to the subject by name, such as Mr. Doe or Ms. Smith. When attributing quotes to the subject, use the past tense and vary the verb choice, similar to the examples below.

   <example_attribution>
   Ms. Smith stated...
   Mr. Doe reported...
   Ms. Doe recalled...
   Mr. Smith explained...
   According to Ms. Doe...
   Ms. Smith told me...
   </example_attribution>

3. Check each section for information that is repeated in other sections. Put this information in the most appropriate section. If a template is provided, it will be wrapped in <template> </template> tags. If duplicate information is found, after placing it in the most appropriate section, reference that section in other parts of the report where that information was removed.

4. After making those changes, revise the document for readability. Preserve details that are important for accurate diagnosis and formulation. Make sure that verbs are in the past tense when reporting information from the interview. Do not use the word "denied," instead say "did not report," "did not endorse," etc.

5. Check the report against a transcript, if provided. The transcript is wrapped in <transcript> </transcript> tags. Pay careful attention to direct quotes. Minor changes in directly quoted statements from the transcript, such as punctuation and capitalization, or removal of words with an ellipsis, are acceptable and do not need to be changed.

6. Use the words instead of numerals for one through ten, and numbers for 11 and above. Spell out decades, such as "twenties" instead of 20s.

7. Some information may not appear in the transcript, such as quotes from other documents or psychometric testing. Do not make changes to this information that does not appear in the transcript. Do make a note of it in your thinking.

8. Preserve existing citation markers in the format `[CIT:ev_<id>]`. If you add factual claims, attach valid citation markers using only IDs from the Citation Evidence Ledger.

9. Output only the final revised report.

## Prompt

Please help refine the following draft report, following the steps above.

Here is the draft report:

<draft>
{draft_report}
</draft>

If a template is provided, it will be inside the <template> </template> tags.

<template>
{template}
</template>

If a transcript is provided, it will be inside the <transcript> </transcript> tags.

<transcript>
{transcript}
</transcript>
