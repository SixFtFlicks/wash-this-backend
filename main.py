from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import openai
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv
import pymongo
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = FastAPI(title="Wash This?? API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = pymongo.MongoClient(MONGO_URL)
db = client.laundry_db
analyses_collection = db.analyses

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

class LaundryAnalysisRequest(BaseModel):
    images: List[str] = Field(..., min_items=1, max_items=5)
    analysis_type: str = Field(..., regex="^(clothing|wash_tag)$")
    user_notes: Optional[str] = None

    @validator('images')
    def validate_images(cls, v):
        for img in v:
            try:
                # Validate base64
                base64.b64decode(img)
            except Exception:
                raise ValueError("Invalid base64 format")
        return v

class WashingRecommendation(BaseModel):
    can_wash_together: bool
    temperature: str
    cycle: str
    detergent_type: str
    special_instructions: List[str]
    separate_loads: Optional[List[Dict[str, Any]]] = None
    reasoning: str

class LaundryAnalysisResponse(BaseModel):
    recommendation: WashingRecommendation
    items_analyzed: List[str]
    analysis_id: str
    timestamp: datetime

def get_enhanced_care_label_prompt():
    return """
You are an expert at reading and decoding clothing care labels. Analyze the care label image and decode each symbol you can see.

IMPORTANT: Do NOT give generic advice like "follow the symbols" or "check the label". Instead, decode each specific symbol you see in the image.

For each symbol visible, provide the exact meaning:
- Washing symbols: "Machine wash at [temperature]°C" or "Hand wash only" or "Do not wash"
- Bleaching symbols: "Do not bleach" or "Bleach allowed" or "Non-chlorine bleach only"
- Drying symbols: "Tumble dry low heat" or "Air dry" or "Do not tumble dry"
- Ironing symbols: "Iron at low/medium/high heat" or "Do not iron"
- Dry cleaning symbols: "Dry clean only" or "Do not dry clean"

Look carefully at each symbol and decode what it specifically means. If you can't see a symbol clearly, say "Symbol not clearly visible" for that category.

Provide specific washing instructions based on the symbols you can actually see in the image.
"""

def get_enhanced_clothing_prompt():
    return """
You are a laundry expert analyzing clothing items to determine if they can be washed together safely.

Analyze the clothing items in the image(s) and consider:
- Fabric types and colors
- Potential for color bleeding
- Different care requirements
- Fabric delicacy levels

Provide specific, actionable washing recommendations including exact temperature, cycle type, and detergent recommendations.

If items cannot be washed together, explain exactly why and suggest how to separate them.
"""

def analyze_with_openai(images: List[str], analysis_type: str, user_notes: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced AI analysis with better prompts for care labels"""
    
    # Choose the appropriate prompt based on analysis type
    if analysis_type == "wash_tag":
        system_prompt = get_enhanced_care_label_prompt()
    else:
        system_prompt = get_enhanced_clothing_prompt()
    
    # Prepare images for OpenAI
    image_urls = []
    for img_base64 in images:
        image_urls.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}",
                "detail": "high"
            }
        })
    
    # Prepare the message content
    content = [{"type": "text", "text": system_prompt}] + image_urls
    
    if user_notes:
        content.append({"type": "text", "text": f"Additional notes: {user_notes}"})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        
        # Enhanced parsing for care label responses
        if analysis_type == "wash_tag":
            return parse_care_label_response(ai_response)
        else:
            return parse_clothing_analysis_response(ai_response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

def parse_care_label_response(ai_response: str) -> Dict[str, Any]:
    """Enhanced parsing specifically for care label analysis"""
    
    # Extract temperature information
    temperature = "as indicated on label"
    if "30°C" in ai_response or "30 degrees" in ai_response.lower():
        temperature = "30°C maximum"
    elif "40°C" in ai_response or "40 degrees" in ai_response.lower():
        temperature = "40°C maximum"
    elif "60°C" in ai_response or "60 degrees" in ai_response.lower():
        temperature = "60°C maximum"
    elif "cold" in ai_response.lower() or "cold water" in ai_response.lower():
        temperature = "cold water only"
    elif "hand wash" in ai_response.lower():
        temperature = "hand wash only"
    elif "do not wash" in ai_response.lower():
        temperature = "do not wash"
    
    # Extract cycle information
    cycle = "normal"
    if "gentle" in ai_response.lower() or "delicate" in ai_response.lower():
        cycle = "gentle/delicate"
    elif "permanent press" in ai_response.lower():
        cycle = "permanent press"
    elif "hand wash" in ai_response.lower():
        cycle = "hand wash"
    
    # Extract detergent information
    detergent_type = "regular"
    if "no bleach" in ai_response.lower() or "do not bleach" in ai_response.lower():
        detergent_type = "color-safe (no bleach)"
    elif "gentle" in ai_response.lower() or "mild" in ai_response.lower():
        detergent_type = "gentle detergent"
    
    # Extract special instructions from the response
    special_instructions = []
    lines = ai_response.split('\n')
    for line in lines:
        line = line.strip()
        if line and ('•' in line or '-' in line or 'do not' in line.lower() or 'iron' in line.lower() or 'dry' in line.lower()):
            # Clean up the instruction
            instruction = line.replace('•', '').replace('-', '').strip()
            if instruction and len(instruction) > 10:  # Only add substantial instructions
                special_instructions.append(instruction)
    
    # If no specific instructions found, add some generic ones
    if not special_instructions:
        special_instructions = [
            "Follow care symbols on the label exactly",
            "Check water temperature symbols carefully",
            "Look for drying and ironing instructions",
            "Pay attention to bleaching restrictions"
        ]
    
    return {
        "can_wash_together": True,  # For care labels, this is always True since it's one item
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent_type,
        "special_instructions": special_instructions,
        "reasoning": ai_response,
        "items_analyzed": ["Care label"]
    }

def parse_clothing_analysis_response(ai_response: str) -> Dict[str, Any]:
    """Parse clothing analysis response"""
    
    # Determine if items can be washed together
    can_wash_together = True
    if any(phrase in ai_response.lower() for phrase in ["cannot", "can't", "should not", "separate", "different loads"]):
        can_wash_together = False
    
    # Extract temperature
    temperature = "cold"
    if "warm" in ai_response.lower():
        temperature = "warm"
    elif "hot" in ai_response.lower():
        temperature = "hot"
    
    # Extract cycle
    cycle = "normal"
    if "gentle" in ai_response.lower() or "delicate" in ai_response.lower():
        cycle = "gentle"
    
    # Extract detergent type
    detergent_type = "regular"
    if "color-safe" in ai_response.lower():
        detergent_type = "color-safe"
    
    # Extract special instructions
    special_instructions = []
    lines = ai_response.split('\n')
    for line in lines:
        line = line.strip()
        if line and ('•' in line or '-' in line or 'separate' in line.lower()):
            instruction = line.replace('•', '').replace('-', '').strip()
            if instruction and len(instruction) > 10:
                special_instructions.append(instruction)
    
    return {
        "can_wash_together": can_wash_together,
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent_type,
        "special_instructions": special_instructions,
        "reasoning": ai_response,
        "items_analyzed": ["Clothing items"]
    }

@app.get("/")
async def root():
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": True,
        "model": "gpt-4-vision-preview"
    }

@app.get("/api/")
async def api_root():
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": True,
        "model": "gpt-4-vision-preview"
    }

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)
async def analyze_laundry(request: LaundryAnalysisRequest):
    try:
        # Validate request
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Analyze with AI
        analysis_result = analyze_with_openai(
            request.images, 
            request.analysis_type, 
            request.user_notes
        )
        
        # Create response
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        recommendation = WashingRecommendation(
            can_wash_together=analysis_result["can_wash_together"],
            temperature=analysis_result["temperature"],
            cycle=analysis_result["cycle"],
            detergent_type=analysis_result["detergent_type"],
            special_instructions=analysis_result["special_instructions"],
            reasoning=analysis_result["reasoning"]
        )
        
        response = LaundryAnalysisResponse(
            recommendation=recommendation,
            items_analyzed=analysis_result["items_analyzed"],
            analysis_id=analysis_id,
            timestamp=timestamp
        )
        
        # Save to database
        try:
            analysis_doc = {
                "id": analysis_id,
                "timestamp": timestamp,
                "analysis_type": request.analysis_type,
                "recommendation": recommendation.dict(),
                "items_analyzed": analysis_result["items_analyzed"]
            }
            analyses_collection.insert_one(analysis_doc)
        except Exception as db_error:
            print(f"Database save error: {db_error}")
            # Continue without failing the request
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 10):
    try:
        analyses = list(analyses_collection.find().sort("timestamp", -1).limit(limit))
        
        # Convert MongoDB documents to proper format
        for analysis in analyses:
            analysis["_id"] = str(analysis["_id"])
        
        return {"analyses": analyses}
    except Exception as e:
        return {"analyses": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
