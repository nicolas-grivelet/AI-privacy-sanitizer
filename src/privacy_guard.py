"""
PrivacyGuard: A tool to anonymize PII in text using a hybrid approach (Regex + NER).
Author: Nicolas Grivelet
"""

import re
import logging
from typing import Dict, List, Tuple
from transformers import pipeline

# Configure logging for library usage
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class PrivacyGuard:
    """
    A class to anonymize Personally Identifiable Information (PII) in text.

    It uses a hybrid approach combining Regex for structured data (emails, phones, IBANs)
    and Named Entity Recognition (NER) models for unstructured entities (persons, locations, organizations).
    """

    def __init__(self):
        """
        Initializes the PrivacyGuard instance.

        Loads the NER models for English and French and compiles the Regex patterns.
        """
        logger.info("Initializing PrivacyGuard...")
        self.models = self._load_models()
        self.regex_patterns = self._compile_regex_patterns()
        logger.info("PrivacyGuard initialized successfully.")

    def _load_models(self) -> Dict[str, pipeline]:
        """
        Loads the NER models for English and French.

        Models chosen for a balance between accuracy and CPU efficiency:
        - English: dslim/bert-base-NER (Robust performance, widely used, efficient enough for CPU).
        - French: Jean-Baptiste/camembert-ner (High accuracy for French, optimized architecture).

        These models are standard baselines that offer good precision without the extreme resource usage of LLMs.

        Returns:
            Dict[str, pipeline]: A dictionary containing the loaded pipelines.
        """
        logger.info("Loading NER models...")
        try:
            # Using aggregation_strategy="simple" to group tokens into whole words/entities
            models = {
                "en": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
                "fr": pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
            }
            logger.info("NER models loaded.")
            return models
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compiles Regex patterns for detecting structured PII.

        Returns:
            Dict[str, re.Pattern]: A dictionary of compiled regex patterns.
        """
        return {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            # Simple phone regex, can be improved for specific locales
            # Using (?<!\w) to allow matches starting with + preceded by space
            "PHONE": re.compile(r'(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b'),
            # Basic IBAN regex
            "IBAN": re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b')
        }

    def detect_pii_regex(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Detects PII using Regex patterns.

        Args:
            text (str): The input text.

        Returns:
            List[Tuple[int, int, str, str]]: A list of tuples (start, end, label, text).
        """
        matches = []
        for label, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), label, match.group()))
        return matches

    def detect_pii_ner(self, text: str, language: str = "en") -> List[Tuple[int, int, str, str]]:
        """
        Detects PII using NER models.

        Args:
            text (str): The input text.
            language (str): The language of the text ('en' or 'fr').

        Returns:
            List[Tuple[int, int, str, str]]: A list of tuples (start, end, label, text).
        """
        if language not in self.models:
            logger.warning(f"Language '{language}' not supported. Defaulting to English.")
            language = "en"
        
        ner_results = self.models[language](text)
        matches = []
        for entity in ner_results:
            # Hugging Face pipelines with aggregation_strategy="simple" return 'entity_group'
            label = entity.get('entity_group')
            # Use the actual text from the input string to ensure accuracy (e.g., preserving spaces)
            content = text[entity['start']:entity['end']]
            matches.append((entity['start'], entity['end'], label, content))
        return matches

    def anonymize(self, text: str, language: str = "en") -> Tuple[str, Dict[str, str]]:
        """
        Anonymizes PII in the text using both Regex and NER.

        Args:
            text (str): The input text.
            language (str): The language of the text.

        Returns:
            Tuple[str, Dict[str, str]]: The sanitized text and a mapping table to restore it.
        """
        logger.info(f"Anonymizing text (Language: {language})...")
        
        # Collect all matches
        regex_matches = self.detect_pii_regex(text)
        ner_matches = self.detect_pii_ner(text, language)
        
        # Combine and sort matches by start position
        all_matches = sorted(regex_matches + ner_matches, key=lambda x: x[0])
        
        # Filter overlapping matches (prefer longer matches or prioritizing regex if needed)
        # Simple strategy: if a match overlaps with a previous one, skip it
        filtered_matches = []
        last_end = -1
        for start, end, label, content in all_matches:
            if start >= last_end:
                filtered_matches.append((start, end, label, content))
                last_end = end
        
        # Create mapping table and replace text
        mapping_table = {}
        # Reconstruct text from parts to avoid index shifting issues
        sanitized_parts = []
        current_idx = 0
        
        counts = {}

        for start, end, label, content in filtered_matches:
            # Append text before the match
            sanitized_parts.append(text[current_idx:start])
            
            # Generate placeholder
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
            placeholder = f"<{label}_{counts[label]}>"
            
            # Store in mapping table
            mapping_table[placeholder] = content
            
            # Append placeholder
            sanitized_parts.append(placeholder)
            
            current_idx = end
        
        # Append remaining text
        sanitized_parts.append(text[current_idx:])
        
        sanitized_text = "".join(sanitized_parts)
        logger.info("Anonymization complete.")
        return sanitized_text, mapping_table

    def restore(self, sanitized_text: str, mapping_table: Dict[str, str]) -> str:
        """
        Restores the original text from the sanitized text using the mapping table.

        Args:
            sanitized_text (str): The anonymized text.
            mapping_table (Dict[str, str]): The mapping table to restore values.

        Returns:
            str: The restored original text.
        """
        logger.info("Restoring text...")
        restored_text = sanitized_text
        # Sort keys by length in descending order to prevent partial replacements (e.g., <PER_1> replacing part of <PER_10>)
        sorted_keys = sorted(mapping_table.keys(), key=len, reverse=True)
        for placeholder in sorted_keys:
            restored_text = restored_text.replace(placeholder, mapping_table[placeholder])
        return restored_text

if __name__ == "__main__":
    # Configure logging for script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize the tool
    pg = PrivacyGuard()
    
    # Example 1: English
    text_en = "Contact John Doe at john.doe@example.com or call +1-555-0199. He lives in New York."
    print("\n--- English Example ---")
    print(f"Original: {text_en}")
    sanitized_en, mapping_en = pg.anonymize(text_en, language="en")
    print(f"Sanitized: {sanitized_en}")
    print(f"Mapping: {mapping_en}")
    restored_en = pg.restore(sanitized_en, mapping_en)
    print(f"Restored: {restored_en}")
    
    # Example 2: French
    text_fr = "M. Jean Dupont habite à Paris. Son email est jean.dupont@orange.fr."
    print("\n--- French Example ---")
    print(f"Original: {text_fr}")
    sanitized_fr, mapping_fr = pg.anonymize(text_fr, language="fr")
    print(f"Sanitized: {sanitized_fr}")
    print(f"Mapping: {mapping_fr}")
    restored_fr = pg.restore(sanitized_fr, mapping_fr)
    print(f"Restored: {restored_fr}")

    # Example 3: Multiple entities (>10) to test sorting fix
    print("\n--- Stress Test (>10 entities) ---")
    names = ["John", "Paul", "George", "Ringo", "Mick", "Keith", "Charlie", "Ronnie", "Freddie", "Brian", "Roger", "Pete"]
    text_stress = ", ".join(names) + " live in London."
    print(f"Original: {text_stress}")
    sanitized_stress, mapping_stress = pg.anonymize(text_stress, language="en")
    print(f"Sanitized: {sanitized_stress}")
    print(f"Mapping: {mapping_stress}")
    restored_stress = pg.restore(sanitized_stress, mapping_stress)
    print(f"Restored: {restored_stress}")
