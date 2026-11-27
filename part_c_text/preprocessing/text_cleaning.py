import html
import re

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
ALEF_RE = re.compile(r"[أإآ]")
DIGITS_RE = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
NON_LETTER_RE = re.compile(r"[^\u0621-\u064Aa-z\s]")

def normalize_arabic(text):
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = ALEF_RE.sub("ا", text)
    text = text.replace("ى", "ي")
    return text

RAW_STOPWORDS = """في من على الى إلى عن مع هذا هذه هذان هاتان ذلك تلك هؤلاء اولئك أولئك هنا هناك
                هو هي هم هما هن انا أنا نحن انت أنت انتم أنتم
                التي الذي الذين اللذان اللتان اللاتي اللواتي
                ان إن أن أنه انه أنها انها إنه إنها
                كان كانت كانوا يكون تكون يكن
                لا لم لن ما ماذا متى اين أين كيف لماذا هل
                كل بعض اي أي كلا
                او أو ثم و قد لقد حتى بين عند عندما بعد قبل خلال منذ
                ايضا أيضا فقط غير سوى لكن لكنه لكنها بل
                اذا إذا اذ إذ كما مثل لدى حيث حين
                له لها لهم لهما لهن به بها بهم فيه فيها فيهم منه منها منهم
                عليه عليها عليهم اليه اليها اليهم عنه عنها معه معها
                وقد وان وهو وهي ولا ولم وما وفي ومن وعلى والى وعن ومع
                بان بأن كي لو ليس ليست"""

STOPWORDS = {normalize_arabic(word) for word in RAW_STOPWORDS.split()}

def clean_arabic_text(text):
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = normalize_arabic(text)
    text = DIGITS_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)

    tokens = [
        token for token in text.split()
        if len(token) > 1 and token not in STOPWORDS
    ]

    return " ".join(tokens)
