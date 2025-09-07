import httpx
import re
from typing import Optional, List, Dict, Any, Tuple
from config import get_settings
import os
import json

settings = get_settings()


class KJVBibleService:
    """Service for fetching and searching Bible verses.

    Primary source: jsDelivr-hosted Bible API repo (public-domain versions).
    Fallback: small built-in KJV map for popular verses.
    """

    # jsDelivr CDN base hosting the bible-api repo
    CDN_BASE = "https://cdn.jsdelivr.net/gh/wldeh/bible-api/bibles"

    # Map short codes to repo version folders
    TRANSLATION_CODE_MAP: Dict[str, str] = {
        "kjv": "en-kjv",
        "asv": "en-asv",
        "web": "en-web",
        "ylt": "en-ylt",
    }

    FRIENDLY_NAMES: Dict[str, str] = {
        "kjv": "King James Version (KJV)",
        "asv": "American Standard Version (ASV)",
        "web": "World English Bible (WEB)",
        "ylt": "Young's Literal Translation (YLT)",
        "niv": "New International Version (NIV)",
    }

    # Whole-Bible JSON sources for licensed versions we have permission for
    VERSION_JSON_URLS: Dict[str, str] = {
        # NIV book→chapter→verse JSON
        "niv": "https://raw.githubusercontent.com/jadenzaleski/BibleTranslations/refs/heads/master/NIV/NIV_bible.json",
    }

    # Local repo path for offline JSONs
    LOCAL_TRANSLATIONS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'BibleTranslations')

    # Basic alias map for book slugs used by the repo
    BOOK_ALIASES: Dict[str, str] = {
        "psalm": "psalms",
        "song of songs": "song-of-solomon",
        "song of solomon": "song-of-solomon",
        "canticles": "song-of-solomon",
    }

    REF_RE = re.compile(r"^\s*([1-3]?\s*[A-Za-z ]+)\s+(\d+):(\d+)")

    def __init__(self):
        self.timeout = settings.REQUEST_TIMEOUT
        self._version_cache: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}

    async def get_verse(self, reference: str, translation: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a specific Bible verse by reference using the jsDelivr CDN.

        Accepts references like "John 3:16" or "Psalm 23:1" and short translation codes
        like "kjv" or "asv". Falls back to a small built-in KJV map.
        """
        ver_code = (translation or "kjv").lower()
        version_folder = self.TRANSLATION_CODE_MAP.get(ver_code, self.TRANSLATION_CODE_MAP["kjv"])

        try:
            book_slug, chapter, verse = self._parse_reference(reference)
        except ValueError:
            return self._get_fallback_verse(reference, translation)

        # If this translation has a full-Bible JSON source (e.g., NIV), try it first
        if ver_code in self.VERSION_JSON_URLS:
            data = await self._get_version_json(ver_code)
            if data:
                book_key = self._denormalize_book(book_slug)
                chap_key = str(chapter)
                verse_key = str(verse)
                try:
                    text = (data[book_key][chap_key][verse_key]).strip()
                    ref_string = f"{book_key} {chapter}:{verse}"
                    return {
                        "reference": ref_string,
                        "text": text,
                        "verses": [],
                        "translation": ver_code.upper(),
                    }
                except Exception:
                    # fall through to CDN-based attempts
                    pass

        url = f"{self.CDN_BASE}/{version_folder}/books/{book_slug}/chapters/{chapter}/verses/{verse}.json"
        # raw GitHub fallback (useful for large packages not served by jsDelivr)
        raw_url = (
            f"https://raw.githubusercontent.com/wldeh/bible-api/main/bibles/"
            f"{version_folder}/books/{book_slug}/chapters/{chapter}/verses/{verse}.json"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Try jsDelivr
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    text = (data.get("text") or "").strip()
                    ref_string = f"{self._denormalize_book(book_slug)} {chapter}:{verse}"
                    return {
                        "reference": ref_string,
                        "text": text,
                        "verses": [],
                        "translation": ver_code.upper(),
                    }
                # Try raw GitHub fallback for this version
                raw_resp = await client.get(raw_url)
                if raw_resp.status_code == 200:
                    data = raw_resp.json()
                    text = (data.get("text") or "").strip()
                    ref_string = f"{self._denormalize_book(book_slug)} {chapter}:{verse}"
                    return {
                        "reference": ref_string,
                        "text": text,
                        "verses": [],
                        "translation": ver_code.upper(),
                    }
                # attempt KJV fallback via CDN if another version failed
                if ver_code != "kjv":
                    kjv_url = url.replace(version_folder, self.TRANSLATION_CODE_MAP["kjv"])  # same path
                    kjv_resp = await client.get(kjv_url)
                    if kjv_resp.status_code == 200:
                        data = kjv_resp.json()
                        text = (data.get("text") or "").strip()
                        ref_string = f"{self._denormalize_book(book_slug)} {chapter}:{verse}"
                        return {
                            "reference": ref_string,
                            "text": text,
                            "verses": [],
                            "translation": "KJV",
                        }
                return self._get_fallback_verse(reference, translation)
            except Exception as e:
                print(f"Error fetching verse from CDN: {e}")
                return self._get_fallback_verse(reference, translation)

    async def _get_version_json(self, code: str) -> Optional[Dict[str, Dict[str, Dict[str, str]]]]:
        """Fetch and cache a whole-Bible JSON by translation code."""
        if code in self._version_cache:
            return self._version_cache[code]
        # Prefer local file if present (offline)
        local_file = None
        code_upper = code.upper()
        if os.path.isdir(self.LOCAL_TRANSLATIONS_ROOT):
            # Heuristic: most translations expose <CODE>_bible.json (e.g., NIV/NIV_bible.json)
            candidate = os.path.join(self.LOCAL_TRANSLATIONS_ROOT, code_upper, f"{code_upper}_bible.json")
            if os.path.isfile(candidate):
                local_file = candidate
        if local_file:
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._version_cache[code] = data
                return data
            except Exception as e:
                print(f"Error loading local {code} JSON: {e}")

        url = self.VERSION_JSON_URLS.get(code)
        if not url:
            return None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    # Expect structure: { Book: { chapter: { verse: text } } }
                    self._version_cache[code] = data
                    return data
            except Exception as e:
                print(f"Error loading {code} JSON: {e}")
        return None

    async def get_chapter(self, book: str, chapter: int, translation: Optional[str] = None) -> Dict[str, Any]:
        """Fetch an entire chapter as a list of verses.

        Returns shape: { book, chapter, translation, verses: [{ verse, text }] }
        """
        ver_code = (translation or "kjv").lower()
        version_folder = self.TRANSLATION_CODE_MAP.get(ver_code, self.TRANSLATION_CODE_MAP["kjv"])

        # NIV or other full-JSON versions
        if ver_code in self.VERSION_JSON_URLS:
            data = await self._get_version_json(ver_code)
            if data:
                book_key = self._denormalize_book(self._slugify_book(book))
                chap_key = str(chapter)
                try:
                    verses_map = data[book_key][chap_key]
                    verses_list = [
                        {"verse": int(vn), "text": (txt or "").strip()} for vn, txt in verses_map.items()
                    ]
                    verses_list.sort(key=lambda x: x["verse"])  # ensure numeric order
                    return {
                        "book": book_key,
                        "chapter": int(chapter),
                        "translation": ver_code.upper(),
                        "verses": verses_list,
                    }
                except Exception:
                    # fall through to CDN-based attempts
                    pass

        # CDN-based public domain versions (KJV/ASV/WEB/YLT)
        book_slug = self._slugify_book(book)
        url = f"{self.CDN_BASE}/{version_folder}/books/{book_slug}/chapters/{int(chapter)}.json"
        raw_url = (
            f"https://raw.githubusercontent.com/wldeh/bible-api/main/bibles/"
            f"{version_folder}/books/{book_slug}/chapters/{int(chapter)}.json"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data") or []
                    verses_list = [
                        {"verse": int(i.get("verse")), "text": (i.get("text") or "").strip()} for i in items if i.get("verse")
                    ]
                    return {
                        "book": self._denormalize_book(book_slug),
                        "chapter": int(chapter),
                        "translation": ver_code.upper(),
                        "verses": verses_list,
                    }
                raw_resp = await client.get(raw_url)
                if raw_resp.status_code == 200:
                    data = raw_resp.json()
                    items = data.get("data") or []
                    verses_list = [
                        {"verse": int(i.get("verse")), "text": (i.get("text") or "").strip()} for i in items if i.get("verse")
                    ]
                    return {
                        "book": self._denormalize_book(book_slug),
                        "chapter": int(chapter),
                        "translation": ver_code.upper(),
                        "verses": verses_list,
                    }
            except Exception as e:
                print(f"Error fetching chapter from CDN: {e}")

        # Fallback minimal when all else fails
        return {
            "book": self._denormalize_book(book_slug),
            "chapter": int(chapter),
            "translation": ver_code.upper(),
            "verses": [],
        }

    def _parse_reference(self, ref: str) -> Tuple[str, int, int]:
        """Parse human reference into (book_slug, chapter, verse)."""
        m = self.REF_RE.match(ref)
        if not m:
            raise ValueError("Invalid reference")
        raw_book, chap_str, verse_str = m.group(1), m.group(2), m.group(3)
        book_norm = self._slugify_book(raw_book)
        return book_norm, int(chap_str), int(verse_str)

    def _slugify_book(self, book: str) -> str:
        b = " ".join(book.strip().lower().split())
        # normalize numbered books e.g., "1 john" → "1 john"
        b = self.BOOK_ALIASES.get(b, b)
        return b.replace(" ", "-")

    def _denormalize_book(self, slug: str) -> str:
        # restore basic capitalization for display; keep hyphens as spaces
        words = slug.replace("-", " ").split()
        if not words:
            return slug
        # handle leading number tokens like "1" "2" "3"
        if words[0] in {"1", "2", "3"} and len(words) > 1:
            return f"{words[0]} {words[1].capitalize()}" + (" " + " ".join(w.capitalize() for w in words[2:]) if len(words) > 2 else "")
        return " ".join(w.capitalize() for w in words)

    def get_supported_versions(self) -> List[Dict[str, str]]:
        """Return supported translation codes and friendly names."""
        versions: List[Dict[str, str]] = []
        all_codes = set(self.TRANSLATION_CODE_MAP.keys()) | set(self.VERSION_JSON_URLS.keys())
        # Add any locally available folders dynamically
        if os.path.isdir(self.LOCAL_TRANSLATIONS_ROOT):
            for name in os.listdir(self.LOCAL_TRANSLATIONS_ROOT):
                folder = os.path.join(self.LOCAL_TRANSLATIONS_ROOT, name)
                if os.path.isdir(folder):
                    all_codes.add(name.lower())
        for code in sorted(all_codes):
            versions.append({
                "code": code,
                "name": self.FRIENDLY_NAMES.get(code, code.upper()),
                "source": (
                    ("local BibleTranslations" if os.path.isdir(os.path.join(self.LOCAL_TRANSLATIONS_ROOT, code.upper())) else
                     ("jadenzaleski/BibleTranslations (raw)" if code in self.VERSION_JSON_URLS else
                      "wldeh/bible-api via jsDelivr/raw GitHub"))
                )
            })
        return versions
    
    def _get_fallback_verse(self, reference: str, translation: Optional[str]) -> Dict[str, Any]:
        """Fallback verses for common references"""
        fallback_verses = {
            "John 3:16": "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.",
            "Philippians 4:13": "I can do all things through Christ which strengtheneth me.",
            "Jeremiah 29:11": "For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end.",
            "Romans 8:28": "And we know that all things work together for good to them that love God, to them who are the called according to his purpose.",
            "Psalm 23:1": "The LORD is my shepherd; I shall not want.",
            "Proverbs 3:5-6": "Trust in the LORD with all thine heart; and lean not unto thine own understanding. In all thy ways acknowledge him, and he shall direct thy paths.",
            "Isaiah 40:31": "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint.",
            "Matthew 6:33": "But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you.",
            "1 Corinthians 13:4": "Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up,",
            "Psalm 46:1": "God is our refuge and strength, a very present help in trouble."
        }
        
        text = fallback_verses.get(reference, "Seek and ye shall find; knock, and it shall be opened unto you.")
        return {
            "reference": reference,
            "text": text,
            "translation": translation.upper() if translation else "KJV"
        }
    
    async def search_verses_by_topic(self, topic: str) -> List[Dict[str, str]]:
        """Search for verses by topic or keyword"""
        topic_verses = self._get_topic_verses()
        
        topic_lower = topic.lower()
        for key, verses in topic_verses.items():
            if key in topic_lower or topic_lower in key:
                return verses
        
        return self._get_default_verses()
    
    def _get_topic_verses(self) -> Dict[str, List[Dict[str, str]]]:
        """Get topic-based verse recommendations"""
        return {
            "anxiety": [
                {"ref": "Philippians 4:6-7", "text": "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God."},
                {"ref": "1 Peter 5:7", "text": "Casting all your care upon him; for he careth for you."},
                {"ref": "Matthew 6:34", "text": "Take therefore no thought for the morrow: for the morrow shall take thought for the things of itself."}
            ],
            "faith": [
                {"ref": "Hebrews 11:1", "text": "Now faith is the substance of things hoped for, the evidence of things not seen."},
                {"ref": "2 Corinthians 5:7", "text": "For we walk by faith, not by sight."},
                {"ref": "Mark 11:24", "text": "Therefore I say unto you, What things soever ye desire, when ye pray, believe that ye receive them, and ye shall have them."}
            ],
            "love": [
                {"ref": "1 Corinthians 13:4-7", "text": "Charity suffereth long, and is kind; charity envieth not..."},
                {"ref": "1 John 4:19", "text": "We love him, because he first loved us."},
                {"ref": "John 13:34", "text": "A new commandment I give unto you, That ye love one another; as I have loved you."}
            ],
            "strength": [
                {"ref": "Isaiah 40:31", "text": "But they that wait upon the LORD shall renew their strength..."},
                {"ref": "Philippians 4:13", "text": "I can do all things through Christ which strengtheneth me."},
                {"ref": "2 Timothy 1:7", "text": "For God hath not given us the spirit of fear; but of power, and of love, and of a sound mind."}
            ],
            "peace": [
                {"ref": "John 14:27", "text": "Peace I leave with you, my peace I give unto you: not as the world giveth, give I unto you."},
                {"ref": "Isaiah 26:3", "text": "Thou wilt keep him in perfect peace, whose mind is stayed on thee: because he trusteth in thee."},
                {"ref": "Philippians 4:7", "text": "And the peace of God, which passeth all understanding, shall keep your hearts and minds through Christ Jesus."}
            ],
            "patience": [
                {"ref": "James 1:3-4", "text": "Knowing this, that the trying of your faith worketh patience."},
                {"ref": "Romans 12:12", "text": "Rejoicing in hope; patient in tribulation; continuing instant in prayer."},
                {"ref": "Galatians 6:9", "text": "And let us not be weary in well doing: for in due season we shall reap, if we faint not."}
            ]
        }
    
    def _get_default_verses(self) -> List[Dict[str, str]]:
        """Get default verses for any topic"""
        return [
            {"ref": "Proverbs 3:5-6", "text": "Trust in the LORD with all thine heart..."},
            {"ref": "Psalm 119:105", "text": "Thy word is a lamp unto my feet, and a light unto my path."},
            {"ref": "Romans 8:28", "text": "And we know that all things work together for good to them that love God."}
        ]