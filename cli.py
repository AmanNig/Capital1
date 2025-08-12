"""
Command-line interface for the Agricultural NLP Pipeline

Usage examples:
  python cli.py --text "गेहूं की फसल में पानी देना है"
  python cli.py --text "What is the price of rice in Punjab?" --details
  python cli.py               # starts interactive mode

Notes:
- Prints only ASCII-friendly output to avoid Windows console encoding issues.
"""

import argparse
import json
import sys
from typing import Dict

from nlp_pipeline import LanguageDetector, AdvancedIntentClassifier


def format_scores(scores: Dict[str, float]) -> str:
    """Return a compact JSON string for scores with 3 decimal precision."""
    rounded = {k: float(f"{v:.3f}") for k, v in scores.items()}
    return json.dumps(rounded, ensure_ascii=True)


def process_text(
    text: str,
    language_detector: LanguageDetector,
    intent_classifier: AdvancedIntentClassifier,
    show_details: bool,
) -> None:
    """Process a single text and print language and intent results using provided components."""

    # Language detection
    lang_scores = language_detector.detect_language(text)
    primary_lang = language_detector.get_primary_language(text)
    is_mixed = language_detector.is_code_mixed(text)

    # Intent classification
    intent_scores = intent_classifier.classify_intent(text)
    primary_intent = intent_classifier.get_primary_intent(text)
    intent_confidence = intent_classifier.get_intent_confidence(text)

    # Output (ASCII only)
    print("RESULT")
    print(f"- Language: {primary_lang}  (code_mixed: {is_mixed})")
    print(f"- Intent: {primary_intent}  (confidence: {intent_confidence:.3f})")

    if show_details:
        print(f"- Language scores: {format_scores(lang_scores)}")
        print(f"- Intent scores:   {format_scores(intent_scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for language and intent detection")
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to analyze. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Disable transformer models for faster, lightweight inference.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed score distributions for language and intent.",
    )

    args = parser.parse_args()
    use_transformer = not args.no_transformer

    # Initialize components once
    language_detector = LanguageDetector(use_transformer=use_transformer)
    intent_classifier = AdvancedIntentClassifier(use_semantic=True, use_zero_shot=True)

    if args.text:
        process_text(
            args.text,
            language_detector=language_detector,
            intent_classifier=intent_classifier,
            show_details=args.details,
        )
        return

    # Interactive mode
    print("Interactive mode. Press Enter on an empty line to exit.")
    print("Type your query:")
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break

        if not text.strip():
            break

        process_text(
            text,
            language_detector=language_detector,
            intent_classifier=intent_classifier,
            show_details=args.details,
        )


if __name__ == "__main__":
    # Ensure default encoding does not cause crashes on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    main()


