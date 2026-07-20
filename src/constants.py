from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


class Column:
    ID = "id"
    CATEGORY = "category"
    TEXT = "text_"
    RATING = "rating"
    LABEL = "label"

    REVIEW_LENGTH = "review_length"
    SENTENCE_COUNT = "sentence_count"
    CHARACTER_COUNT = "character_count"
    READABILITY_ARI = "readability_ari"
    EXTREMITY = "extremity"

    SUBJECTIVITY = "subjectivity"
    READABILITY_LLM = "readability_llm"
    INTERNAL_CONSISTENCY = "internal_consistency"

    ADJECTIVE_COUNT = "adjective_count"
    VERB_COUNT = "verb_count"
    SUPERLATIVE_COUNT = "superlative_count"

    ADJECTIVE_RATIO = "adjective_ratio"
    VERB_RATIO = "verb_ratio"
    SUPERLATIVE_RATIO = "superlative_ratio"

    FIRST_PERSON_PRONOUN_COUNT = "first_person_pronoun_count"
    SECOND_PERSON_PRONOUN_COUNT = "second_person_pronoun_count"
    THIRD_PERSON_PRONOUN_COUNT = "third_person_pronoun_count"

    FIRST_PERSON_PRONOUN_RATIO = "first_person_pronoun_ratio"
    SECOND_PERSON_PRONOUN_RATIO = "second_person_pronoun_ratio"
    THIRD_PERSON_PRONOUN_RATIO = "third_person_pronoun_ratio"


MODEL_FEATURE_COLUMNS = [
    Column.REVIEW_LENGTH,
    Column.SUBJECTIVITY,
    Column.READABILITY_ARI,
    Column.EXTREMITY,
    Column.INTERNAL_CONSISTENCY,
    Column.ADJECTIVE_RATIO,
    Column.VERB_RATIO,
    Column.SUPERLATIVE_RATIO,
    Column.FIRST_PERSON_PRONOUN_RATIO,
    Column.SECOND_PERSON_PRONOUN_RATIO,
    Column.THIRD_PERSON_PRONOUN_RATIO,
]
