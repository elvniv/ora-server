from openai import OpenAI
from models import VerseRequest, VerseContext
from services.bible_service import KJVBibleService
from config import get_settings

settings = get_settings()
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.has_openai_key else None


async def get_verse_context_data(request: VerseRequest, bible_service: KJVBibleService) -> VerseContext:
    """Generate detailed context and explanation for a verse"""
    # Fetch the verse
    verse_data = await bible_service.get_verse(request.reference, request.translation)
    
    # Generate context using AI if available
    if openai_client:
        context_text = await _generate_ai_context(request.reference, verse_data['text'])
    else:
        context_text = "This verse offers spiritual wisdom and guidance for daily life."
    
    # Split into context and application
    parts = context_text.split("Personal application:" if "Personal application:" in context_text else ".")
    historical_context = parts[0].strip()
    personal_application = parts[1].strip() if len(parts) > 1 else "This verse reminds us of God's constant presence and love in our lives."
    
    # Get related verses
    book_name = request.reference.split()[0].lower()
    related = await bible_service.search_verses_by_topic(book_name)
    related_verses = [v["ref"] for v in related[:3]]
    
    return VerseContext(
        verse_reference=request.reference,
        verse_text=verse_data["text"],
        context=historical_context,
        personal_application=personal_application,
        related_verses=related_verses
    )


async def _generate_ai_context(reference: str, verse_text: str) -> str:
    """Generate AI context for a verse"""
    context_prompt = f"""Explain the context of {reference} ({verse_text}) in 2-3 sentences. 
    Include: 
    1. Historical context
    2. Personal application for someone seeking spiritual growth
    Keep the tone conversational and encouraging."""
    
    response = openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable but gentle Bible teacher. Use simple, accessible language."},
            {"role": "user", "content": context_prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content