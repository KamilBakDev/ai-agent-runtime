"""One-off generator for the two sample-corpus PDFs under data/sample_docs/.

Run manually if the PDFs ever need regenerating: ``python scripts/generate_sample_pdfs.py``.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

DOCS = {
    "sample_contract_force_majeure.pdf": (
        "Sample Commercial Lease Excerpt: Force Majeure Clause",
        [
            "Section 14. Force Majeure.",
            "Neither party shall be liable for any failure or delay in performance under this "
            "Lease to the extent such failure or delay is caused by acts of God, war, "
            "governmental order or regulation, or any other cause beyond the reasonable control "
            "of the affected party (each, a 'Force Majeure Event').",
            "",
            "The party claiming relief under this Section shall provide written notice to the "
            "other party within ten (10) business days of the onset of the Force Majeure Event, "
            "describing the event and its expected impact on performance.",
            "",
            "If a Force Majeure Event continues for more than one hundred eighty (180) "
            "consecutive days, either party may terminate this Lease upon written notice, "
            "without further liability except for obligations accrued prior to termination.",
        ],
    ),
    "sample_contract_indemnification.pdf": (
        "Sample Vendor Agreement Excerpt: Indemnification",
        [
            "Section 9. Indemnification.",
            "Each party (the 'Indemnifying Party') shall defend, indemnify, and hold harmless "
            "the other party from and against any third-party claims, damages, and reasonable "
            "attorneys' fees arising out of the Indemnifying Party's breach of this Agreement, "
            "gross negligence, or willful misconduct.",
            "",
            "The Indemnifying Party's duty to defend shall apply as claims are asserted, and "
            "shall not be contingent upon a final adjudication of liability.",
            "",
            "Except for claims arising from a breach of confidentiality or data protection "
            "obligations, the Indemnifying Party's aggregate liability under this Section shall "
            "not exceed the total fees paid under this Agreement in the twelve (12) months "
            "preceding the claim.",
        ],
    ),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, paragraphs) in DOCS.items():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        for para in paragraphs:
            pdf.multi_cell(0, 6, para)
            pdf.ln(2)
        out_path = OUT_DIR / filename
        pdf.output(str(out_path))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
