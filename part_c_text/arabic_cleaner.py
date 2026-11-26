import re

STOPWORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللذين", "اللتين", "اللواتي", "هو", "هي", "هم",
    "هن", "نحن", "أنا", "انا", "أنت", "انت", "أنتم", "كان", "كانت", "كانوا",
    "يكون", "تكون", "أن", "ان", "إن", "إذ", "إذا", "اذا", "ما", "ماذا", "متى",
    "أين", "كيف", "لماذا", "كم", "أي", "اي", "أو", "او", "ثم", "بل", "لكن",
    "حتى", "قد", "لقد", "كل", "بعض", "بين", "فوق", "تحت", "أمام", "خلف",
    "عند", "منذ", "خلال", "حول", "ضد", "نحو", "ضمن", "عبر", "وفق", "بعد", "قبل",
    "حيث", "كما", "لأن", "لان", "بينما", "فقط", "أيضا", "ايضا", "جدا", "هنا", "هناك",
}


def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""

    # remove html tags and urls
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove tashkeel (diacritics) and tatweel
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    # normalize arabic letters
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")

    # keep arabic letters and spaces only (removes digits and punctuation)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)

    # remove common stopwords while keeping negation words (لا، لم، لن، ليس، غير)
    tokens = [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]

    return " ".join(tokens)