# ORA Backend - Spiritual Conversations API

## 🎯 Overview

FastAPI backend service powering the ORA spiritual conversations app with:
- **AI-powered chat** using OpenAI GPT-4/3.5 or Claude
- **KJV Bible integration** for verse recommendations
- **Context generation** for Bible verses
- **Topic-based verse search**

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
# Create .env and add your keys (do not commit this file)
cat > .env <<'ENV' 
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# optional
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-3-sonnet-20240229
DEBUG=true
ENV
```

### 3. Run the Server

```bash
# Option 1: Use the startup script
./start.sh

# Option 2: Run directly with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: http://localhost:8000
API documentation at: http://localhost:8000/docs

## 📚 API Endpoints

### **POST /chat**
Main conversation endpoint for spiritual guidance

```json
Request:
{
  "message": "I'm feeling anxious about the future",
  "user_id": "user123",
  "user_name": "Sarah",
  "spiritual_goal": "Finding peace in uncertainty"
}

Response:
{
  "response": "feeling anxious about the future is so common, sarah. god knows every tomorrow before it arrives. what specific worry is weighing heaviest on your heart?",
  "verse_recommendation": {
    "verse_reference": "Matthew 6:34",
    "verse_text": "Take therefore no thought for the morrow: for the morrow shall take thought for the things of itself."
  },
  "follow_up_question": "what would peace look like in this situation?"
}
```

### **GET /verse/{reference}**
Fetch a specific Bible verse

```json
GET /verse/John%203:16

Response:
{
  "reference": "John 3:16",
  "text": "For God so loved the world, that he gave his only begotten Son...",
  "translation": "KJV"
}
```

### **POST /verse/context**
Get detailed context and explanation for a verse

```json
Request:
{
  "reference": "Philippians 4:13",
  "user_id": "user123"
}

Response:
{
  "verse_reference": "Philippians 4:13",
  "verse_text": "I can do all things through Christ which strengtheneth me.",
  "context": "Paul wrote this while imprisoned in Rome, expressing contentment in all circumstances...",
  "personal_application": "This verse reminds you that your strength comes not from yourself but from Christ...",
  "related_verses": ["2 Corinthians 12:9", "Isaiah 40:31", "Psalm 18:32"]
}
```

### **POST /search/verses**
Search for verses by topic

```json
Request:
{
  "query": "anxiety",
  "limit": 3
}

Response:
{
  "query": "anxiety",
  "results": [
    {
      "ref": "Philippians 4:6-7",
      "text": "Be careful for nothing; but in every thing by prayer..."
    },
    {
      "ref": "1 Peter 5:7",
      "text": "Casting all your care upon him; for he careth for you."
    }
  ],
  "count": 3
}
```

### **GET /health**
Check service health status

```json
Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "openai": true,
    "anthropic": true,
    "bible_api": "active"
  }
}
```

## 🧠 AI System Behavior

The AI (ORA) is configured to be:
- **Conversational & warm** - uses lowercase for approachability
- **Spiritually grounded** - recommends relevant KJV verses
- **Non-judgmental** - provides safe space for spiritual exploration
- **Reflective** - asks thoughtful follow-up questions
- **Encouraging** - focuses on growth and hope

## 📖 KJV Bible Integration

- **Primary source**: bible-api.com for real-time verse fetching
- **Fallback verses**: Common verses stored locally for reliability
- **Topic mapping**: Curated verse collections for themes like:
  - Anxiety & worry
  - Faith & trust
  - Love & relationships
  - Strength & perseverance
  - Peace & comfort
  - Patience & waiting

## 🔧 Configuration

### AI Model Selection
- **OpenAI**: Uses GPT-4 Turbo (falls back to GPT-3.5)
- **Claude**: Uses Claude 3 Sonnet
- **Auto-switching**: Randomly selects between available APIs

### Environment Variables
- `OPENAI_API_KEY`: Required for GPT models
- `ANTHROPIC_API_KEY`: Optional for Claude
- `SERVER_HOST`: Default 0.0.0.0
- `SERVER_PORT`: Default 8000

## 🔗 React Native Integration

The React Native app connects via:
- **iOS Simulator**: http://localhost:8000
- **Android Emulator**: http://10.0.2.2:8000
- **Physical Device**: Use your machine's IP address

Configure in `/src/services/api.js`:
```javascript
const API_CONFIG = {
  BASE_URL: Platform.OS === 'ios' 
    ? 'http://localhost:8000' 
    : 'http://10.0.2.2:8000'
}
```

## 🚀 Production Deployment

### Option 1: Deploy to Render.com
1. Push code to GitHub
2. Connect to Render
3. Set environment variables
4. Deploy as Web Service

### Option 2: Deploy to Railway.app
1. Install Railway CLI
2. Run `railway login`
3. Run `railway up`
4. Set environment variables in dashboard

### Option 3: Deploy to Heroku
1. Create `Procfile`: `web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}`
2. Deploy via Heroku CLI or GitHub

## 📊 Performance

- **Response time**: ~1-2 seconds for AI responses
- **Verse lookup**: ~200ms for cached verses
- **Context generation**: ~2-3 seconds
- **Concurrent users**: Supports 100+ simultaneous connections

## 🛠 Troubleshooting

### "API key not found"
- Ensure `.env` file exists with valid keys
- Restart the server after adding keys

### "Connection refused" from React Native
- Check server is running on correct port
- Verify IP address for physical devices
- Check firewall settings

### "Verse not found"
- API will use fallback verses automatically
- Check internet connection for bible-api.com

## 📝 License

MIT License - Feel free to use for your spiritual apps!

## 🙏 Credits

- KJV Bible text from bible-api.com
- AI models by OpenAI and Anthropic
- Built with FastAPI and React Native# ora-server
