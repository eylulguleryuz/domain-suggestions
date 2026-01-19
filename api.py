from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
from datetime import datetime
import uvicorn
from openai import OpenAI

# ------------------ #
#     CONSTANTS
# ------------------ #
NUM_OF_SUGGESTIONS = 5

# ------------------ #
#   OPENAI THINGS
# ------------------ #
client = OpenAI()

# a structure the LLM will use to reply with
class DomainName(BaseModel):
    name: str
    logic: str # why is this domain suggested?
    
class DomainSuggestions(BaseModel):
    domains: List[DomainName]

# ------------------ #
#       PROMPT
# ------------------ #

PROMPT = f"""
You are a world-class Brand Strategist and Naming Specialist (think Lexicon Branding or Pentagram).
Your task is to generate {NUM_OF_SUGGESTIONS} high-potential domain name ideas for a business.

### BRAND STRATEGY LOGIC:
1. **The Radio Test:** If heard once, the spelling must be intuitive. No intentional misspellings (e.g., no 'Kleen' for 'Clean').
2. **Outcome-Centric:** Prioritize the *feeling* or *benefit* (e.g., 'Swift' vs. 'FastDelivery').
3. **Phonosemantics:** Use hard consonants (k, t, p) for tech/efficiency; soft vowels (o, a, l) for wellness/luxury.
4. **Syllable Economy:** Maximum 3 syllables. 1-2 is the "Gold Standard."
5. **Alphabet Check:** If there are non-latin characters, convert them to latin. Suggestions should STRICTLY include ONLY latin characters. 

### NAMING ARCHETYPES (Provide a mix):
- **Evocative:** Uses a real word that captures a vibe (e.g., 'Patagonia', 'Slack').
- **Compound:** Two short words joined (e.g., 'DoorDash', 'YouTube').
- **Abstract/Blended:** Unique, brandable sounds or prefixes (e.g., 'Zillow', 'Vanta').
- **Oxymoronic:** If the business has conflicting goals (e.g., 'CheapLuxury'), create a name that bridges the gap (e.g., 'GrandLite').

### EXAMPLES OF HIGH QUALITY DOMAINS:

1. Business: AI-driven logistics platform that makes shipping invisible and effortless.
   Strategy: Focus on the "Outcome" (Benefit over Feature).
   Suggestions: ['EasyCargo', 'VanishShipping', 'PackageArrived']

2. Business: High-end organic skincare that uses ancient volcanic minerals.
   Strategy: Use "Phonosemantics" (Soft vowels for luxury, hard roots for minerals).
   Suggestions: ['MineralSkin', 'Vitre', 'AshLuxe', 'RelicCare']

3. Business: A budget airline that feels like a private club.
   Strategy: "Oxymoronic Branding" (Bridging high-end vibes with low-cost reality).
   Suggestions: ['WingPrive', 'Goldjet', 'ApexAir']

4. Business: Professional-grade coding tools for children/beginners.
   Strategy: "The Radio Test" (Short, punchy, easy to spell).
   Suggestions: ['Koda', 'Codio', 'FableCode']

5. Business: A neighborhood bakery in Brooklyn using traditional Polish recipes.
   Strategy: "Evocative/Local" (Hinting at heritage without being a literal map).
   Suggestions: ['CrustPL', 'Cracow', 'BabkaBakery']
"""

# ------------------ #
#      FastAPI
# ------------------ #
app = FastAPI(title="Domain Suggestions API", version="1.0")

# Request model
class QueryRequest(BaseModel):
    userinput: str

# Response model
class QueryResponse(BaseModel):
    suggestions: List[DomainName]
    input_tokens: int
    output_tokens: int
    api_speed_sec: float

# Query function
def query(userinput):
    response = client.responses.parse(
        model="gpt-4o-mini",
        instructions=PROMPT,
        input=userinput,
        text_format=DomainSuggestions,
    )
    api_speed_sec = (response.completed_at - response.created_at)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    suggestions_list = response.output_parsed.domains
    
    return suggestions_list, input_tokens, output_tokens, api_speed_sec

# API endpoint
@app.post("/generate", response_model=QueryResponse)
async def generate_domains(request: QueryRequest):
    try:
        suggestions, input_tokens, output_tokens, api_speed_sec = query(
            request.userinput
        )
        return {
            "suggestions": suggestions,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "api_speed_sec": api_speed_sec
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
