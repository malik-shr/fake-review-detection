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

    WORD_COUNT = "word_count"
    CHAR_COUNT = "char_count"
    READABILITY_ARI = "readability_ari"
    EXTREMITY = "extremity"

    SUBJECTIVE_CLAIM_COUNT = "subjective_claim_count"
    OBJECTIVE_CLAIM_COUNT = "objective_claim_count"
    EXPERIENTIAL_DETAIL_CLAIM_COUNT = "experiential_detail_claim_count"
    POSITIVE_AFFECT_CLAIM_COUNT = "positive_affect_claim_count"
    NEGATIVE_AFFECT_CLAIM_COUNT = "negative_affect_claim_count"
    UNCERTAIN_CLAIM_COUNT = "uncertain_claim_count"
    TEXT_SENTIMENT = "text_sentiment"
    SUBJECTIVITY = "subjectivity"
    EXPERIENTIAL_DETAIL = "experiential_detail"
    POSITIVE_AFFECT = "positive_affect"
    NEGATIVE_AFFECT = "negative_affect"
    UNCERTAINTY = "uncertainty"
    INTERNAL_CONSISTENCY = "internal_consistency"

    ADJECTIVE_COUNT = "adjective_count"
    VERB_COUNT = "verb_count"
    SUPERLATIVE_COUNT = "superlative_count"

    ADJECTIVE_RATIO = "adjective_ratio"
    VERB_RATIO = "verb_ratio"
    SUPERLATIVE_RATIO = "superlative_ratio"

    FIRST_PERSON_PRONOUN_COUNT = "first_person_pronoun_count"
    THIRD_PERSON_PRONOUN_COUNT = "third_person_pronoun_count"
    FIRST_PERSON_PRONOUN_RATIO = "first_person_pronoun_ratio"
    THIRD_PERSON_PRONOUN_RATIO = "third_person_pronoun_ratio"


MODEL_FEATURE_COLUMNS = [
    Column.READABILITY_ARI,
    Column.EXTREMITY,

    #LLM
    Column.SUBJECTIVITY,
    Column.EXPERIENTIAL_DETAIL,
    Column.POSITIVE_AFFECT,
    Column.NEGATIVE_AFFECT,
    Column.UNCERTAINTY,
    Column.INTERNAL_CONSISTENCY,

    #Spacy
    Column.WORD_COUNT,
    Column.ADJECTIVE_RATIO,
    Column.VERB_RATIO,
    Column.SUPERLATIVE_RATIO,
    Column.FIRST_PERSON_PRONOUN_RATIO,
    Column.THIRD_PERSON_PRONOUN_RATIO,
]
