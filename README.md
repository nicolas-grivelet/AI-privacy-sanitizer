# AI Privacy Sanitizer 🛡️

This project is a tool developed to secure data transmission to Large Language Models (LLMs).  
The goal is simple: sanitize text of any personal information before it leaves the local environment.

---

## 💡 Why this project?

When using AI tools, it becomes clear that sharing sensitive data (names, emails, coordinates) is a significant risk.

I built this local security gateway that utilizes two complementary methods:

### 🔹 Deterministic Regex  
For structured data (emails, IBANs, phone numbers).  
Fast, efficient, and highly reliable for fixed formats.

### 🔹 Contextual NER (AI)  
For person names, locations, and organizations.

Uses Transformer models:
- CamemBERT (French)
- BERT (English)

These models understand linguistic context instead of relying only on patterns.

---

## 🚀 Features

- **Hybrid Approach**: Combines regular expressions with Deep Learning  
- **Reversible**: Generates a mapping table to restore original values after LLM processing  
- **CPU Optimized**: No GPU required  
- **Multilingual**: Native support for French and English  

---

## 🛠️ Installation

```bash
git clone https://github.com/nicolas-grivelet/ai-privacy-sanitizer.git
cd ai-privacy-sanitizer
pip install -r requirements.txt
````

---

## 💻 Usage Example

```python
from privacy_guard import PrivacyGuard

pg = PrivacyGuard()
text = "Contact Nicolas Grivelet at nicolas@example.com"

# Anonymization
sanitized, mapping = pg.anonymize(text, language="en")
print(sanitized)
# Output: "Contact <PER_1> at <EMAIL_1>"

# Restoration (e.g., after LLM processing)
original = pg.restore(sanitized, mapping)
```

---

## 📝 License

Distributed under the MIT license.
See the `LICENSE` file for more information.
