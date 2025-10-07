from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import os
from datetime import datetime

app = FastAPI(title="Wash This?? API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

class LaundryAnalysisRequest(BaseModel):
    images: List[str]
    analysis_type: str
    user_notes: Optional[str] = None

class WashingRecommendation(BaseModel):
    can_wash_together: bool
    temperature: str
    cycle: str
    detergent_type: str
    special_instructions: List[str]
    reasoning: str

class LaundryAnalysisResponse(BaseModel):
    recommendation: WashingRecommendation
    items_analyzed: List[str]

def analyze_with_openai_simple(images: List[str], analysis_type: str) -> dict:
    """Simplified OpenAI analysis using direct HTTP requests"""
    
    if not OPENAI_API_KEY:
        # Fallback response if no API key
        return get_fallback_response(analysis_type)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Prepare image data
        image_content = []
        for img_base64 in images[:1]:  # Only use first image to keep it simple
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # Choose prompt based on analysis type
        if analysis_type == "wash_tag":
            prompt = "Analyze this care label image and decode each care symbol you see. Provide specific washing instructions for each symbol (temperature, drying, ironing, etc.). Do not give generic advice - decode the actual symbols."
        else:
            prompt = "Analyze these clothing items and determine if they can be washed together safely. Consider fabric types, colors, and care requirements. Provide specific washing recommendations."
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + image_content
                }
            ],
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            return parse_ai_response(ai_response, analysis_type)
        else:
            return get_fallback_response(analysis_type)
            
    except Exception as e:
        print(f"AI analysis error: {e}")
        return get_fallback_response(analysis_type)

def get_fallback_response(analysis_type: str) -> dict:
    """Fallback response when AI is not available"""
    if analysis_type == "wash_tag":
        return {
            "can_wash_together": True,
            "temperature": "30°C maximum",
            "cycle": "gentle",
            "detergent_type": "mild detergent",
            "special_instructions": [
                "Check care label for specific temperature symbols",
                "Look for drying instructions on the label",
                "Follow ironing temperature symbols",
                "Check for bleaching restrictions"
            ],
            "reasoning": "Care label analysis: Please refer to the specific symbols on your care label for detailed instructions.",
            "items_analyzed": ["Care label"]
        }
    else:
        return {
            "can_wash_together": True,
            "temperature": "cold",
            "cycle": "normal",
            "detergent_type": "color-safe",
            "special_instructions": [
                "Separate dark and light colors",
                "Check fabric care labels",
                "Use appropriate water temperature",
                "Consider fabric delicacy"
            ],
            "reasoning": "General laundry recommendation: For best results, separate by color and fabric type.",
            "items_analyzed": ["Clothing items"]
        }

def parse_ai_response(ai_response: str, analysis_type: str) -> dict:
    """Parse AI response into structured data"""
    
    # Determine if items can be washed together (for clothing analysis)
    can_wash_together = True
    if analysis_type == "clothing" and any(word in ai_response.lower() for word in ["cannot", "separate", "different"]):
        can_wash_together = False
    
    # Extract temperature
    temperature = "cold"
    if "30°c" in ai_response.lower() or "30 degrees" in ai_response.lower():
        temperature = "30°C maximum"
    elif "40°c" in ai_response.lower() or "40 degrees" in ai_response.lower():
        temperature = "40°C maximum"
    elif "warm" in ai_response.lower():
        temperature = "warm"
    elif "hot" in ai_response.lower():
        temperature = "hot"
    elif "hand wash" in ai_response.lower():
        temperature = "hand wash only"
    
    # Extract cycle
    cycle = "normal"
    if "gentle" in ai_response.lower() or "delicate" in ai_response.lower():
        cycle = "gentle"
    elif "hand" in ai_response.lower():
        cycle = "hand wash"
    
    # Extract detergent type
    detergent_type = "regular"
    if "no bleach" in ai_response.lower() or "color-safe" in ai_response.lower():
        detergent_type = "color-safe"
    elif "gentle" in ai_response.lower() or "mild" in ai_response.lower():
        detergent_type = "mild detergent"
    
    # Extract special instructions
    special_instructions = []
    lines = ai_response.split('\n')
    for line in lines:
        line = line.strip()
        if line and any(marker in line for marker in ['•', '-', '1.', '2.', '3.']):
            instruction = line.replace('•', '').replace('-', '').strip()
            if len(instruction) > 10:
                special_instructions.append(instruction)
    
    # If no instructions found, add some basic ones
    if not special_instructions:
        if analysis_type == "wash_tag":
            special_instructions = [
                "Follow temperature symbols carefully",
                "Check drying and ironing instructions",
                "Look for bleaching restrictions"
            ]
        else:
            special_instructions = [
                "Separate by color and fabric type",
                "Check care labels before washing",
                "Use appropriate detergent"
            ]
    
    return {
        "can_wash_together": can_wash_together,
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent_type,
        "special_instructions": special_instructions,
        "reasoning": ai_response,
        "items_analyzed": ["Care label" if analysis_type == "wash_tag" else "Clothing items"]
    }

@app.get("/")
async def root():
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": bool(OPENAI_API_KEY),
        "model": "gpt-4o-mini"
    }

@app.get("/api/")
async def api_root():
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": bool(OPENAI_API_KEY),
        "model": "gpt-4o-mini"
    }

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)
async def analyze_laundry(request: LaundryAnalysisRequest):
    try:
        # Validate request
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        if request.analysis_type not in ["clothing", "wash_tag"]:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        # Analyze with AI or fallback
        analysis_result = analyze_with_openai_simple(
            request.images, 
            request.analysis_type
        )
        
        # Create recommendation
        recommendation = WashingRecommendation(
            can_wash_together=analysis_result["can_wash_together"],
            temperature=analysis_result["temperature"],
            cycle=analysis_result["cycle"],
            detergent_type=analysis_result["detergent_type"],
            special_instructions=analysis_result["special_instructions"],
            reasoning=analysis_result["reasoning"]
        )
        
        # Create response
        response = LaundryAnalysisResponse(
            recommendation=recommendation,
            items_analyzed=analysis_result["items_analyzed"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history():
    # Simple response without database for now
    return {
        "analyses": [],
        "message": "History feature will be available once database is connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
