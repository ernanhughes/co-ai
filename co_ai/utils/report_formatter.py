import os
from datetime import datetime, timezone
from pathlib import Path

class ReportFormatter:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_report(self, run_id, context):
        timestamp = datetime.now(timezone.utc).isoformat()
        file_path = self.output_dir / f"{run_id}_report.md"

        content = f"""# 🧪 AI Co-Research Summary Report

**🗂️ Run ID:** `{run_id}`  
**🎯 Goal:** *{context.get("goal", "")}*  
**📅 Timestamp:** {timestamp}

---

### 🔬 Hypotheses Generated:
{self._format_list(context.get("generated", []))}

---

### 🧠 Persona Reviews:
{self._format_reviews(context.get("reviews", []))}

---

### 🧬 Evolution Outcome:
- {len(context.get("evolved", []))} hypotheses evolved.

---

### 📘 Meta-Review Summary:I
> {context.get("summary", "")}

---
"""

        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def _format_list(self, items):
        return "\n".join(f"1. **{item.strip()}**" for item in items)

    def _format_reviews(self, reviews):
        if not reviews:
            return "No reviews recorded."
        formatted = []
        for r in reviews:
            persona = r.get("persona", "Unknown")
            review = r.get("review", "")
            formatted.append(f"**{persona}:**\n> {review}")
        return "\n\n".join(formatted)
