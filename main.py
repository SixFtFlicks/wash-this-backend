```python

from fastapi import FastAPI, HTTPException

from fastapi.responses import JSONResponse

from starlette.middleware.cors import CORSMiddleware

import os

import logging

import uuid

import json

import re

import base64

from datetime import datetime

from pydantic import BaseModel, Field, validator

from typing import List, Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(title="Wash This?? API", version="1.0.0")

class LaundryAnalysisRequest(BaseModel):

images: List[str]

analysis_type: str = "clothing"

user_notes: Optional[str] = None


@validator('images')

def validate_images(cls, v):

if not v or len(v) == 0:

raise ValueError("At least one image is required")


if len(v) > 10:

raise ValueError("Maximum 10 images allowed per request")


for i, image in enumerate(v):

try:

if not image or len(image.strip()) == 0:

raise ValueError(f"Image {i+1} is empty")


# Remove data URL prefix if present

clean_image = image

if image.startswith('data:image'):

clean_image = image.split(',')[1]


# Validate base64 format

base64.b64decode(clean_image)


# Check size (max 10MB base64)

if len(clean_image) > 14000000: # ~10MB in base64

raise ValueError(f"Image {i+1} is too large (max 10MB)")


# Check minimum size (at least 100 bytes)

if len(clean_image) < 100:

raise ValueError(f"Image {i+1} is too small (minimum size required)")


except ValueError:

raise

except Exception:

raise ValueError(f"Image {i+1} is not valid base64 format")


return v


@validator('analysis_type')

def validate_analysis_type(cls, v):

valid_types = ["clothing", "wash_tag"]

if v not in valid_types:

raise ValueError(f"analysis_type must be one of: {', '.join(valid_types)}")

return v


@validator('user_notes')

def validate_user_notes(cls, v):

if v is not None:

if len(v) > 1000:

raise ValueError("User notes cannot exceed 1000 characters")

# Basic sanitization

if re.search(r'[<>"\']', v):

raise ValueError("User notes contain invalid characters")

return v

class WashingRecommendation(BaseModel):

can_wash_together: bool

temperature: str

cycle: str

detergent_type: str

special_instructions: List[str]

separate_loads: Optional[List[dict]] = None

reasoning: str

class LaundryAnalysisResponse(BaseModel):

id: str = Field(default_factory=lambda: str(uuid.uuid4()))

timestamp: datetime = Field(default_factory=datetime.utcnow)

recommendation: WashingRecommendation

items_analyzed: List[str]

# Get API key

OPENAI_API_KEY = os.getenv('EMERGENT_LLM_KEY') or os.getenv('OPENAI_API_KEY')

async def analyze_with_openai(images: List[str], analysis_type: str, user_notes: str = None) -> WashingRecommendation:

"""Analyze with OpenAI GPT-4 Vision or return detailed mock response"""

try:

if not OPENAI_API_KEY:

logger.info("No API key - using detailed mock analysis")

return get_detailed_mock_recommendation(analysis_type, user_notes)

# Import OpenAI after checking API key

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Prepare image (use first image)

image_base64 = images[0]

if image_base64.startswith('data:image'):

image_base64 = image_base64.split(',')[1]

# Validate base64 one more time before sending to API

try:

base64.b64decode(image_base64)

except Exception:

raise ValueError("Invalid image format for AI analysis")

# Create detailed prompt based on analysis type

if analysis_type == "wash_tag":

prompt = """Analyze this care label and provide detailed washing instructions in JSON format:

{

"can_wash_together": true,

"temperature": "cold/warm/hot",

"cycle": "normal/delicate/gentle",

"detergent_type": "regular/gentle",

"special_instructions": ["instruction1", "instruction2"],

"separate_loads": null,

"reasoning": "Detailed explanation of care symbols and recommended settings"

}"""

else:

prompt = """Analyze these clothing items for washing in JSON format:

{

"can_wash_together": false,

"temperature": "cold/warm/hot",

"cycle": "normal/delicate/heavy duty",

"detergent_type": "regular/gentle/color-safe",

"special_instructions": ["instruction1", "instruction2"],

"separate_loads": [{"items": ["item1"], "settings": "washing instructions"}],

"reasoning": "Detailed fabric and color analysis with washing recommendations"

}"""

# Make API call with proper error handling

try:

response = client.chat.completions.create(

model="gpt-4o-mini",

messages=[{

"role": "user",

"content": [

{"type": "text", "text": prompt},

{

"type": "image_url",

"image_url": {

"url": f"data:image/jpeg;base64,{image_base64}",

"detail": "low"

}

}

]

}],

max_tokens=1000,

temperature=0.1

)

except Exception as api_error:

logger.error(f"OpenAI API error: {str(api_error)}")

raise HTTPException(status_code=503, detail="AI analysis service temporarily unavailable")

# Parse and validate response

content = response.choices[0].message.content


if not content or len(content.strip()) == 0:

raise ValueError("Empty response from AI analysis")


try:

# Extract JSON with validation

start_idx = content.find('{')

end_idx = content.rfind('}') + 1


if start_idx == -1 or end_idx <= start_idx:

raise ValueError("No valid JSON found in AI response")


json_str = content[start_idx:end_idx]

analysis_data = json.loads(json_str)


# Validate required fields

required_fields = ["can_wash_together", "temperature", "cycle", "detergent_type", "reasoning"]

missing_fields = [field for field in required_fields if field not in analysis_data]


if missing_fields:

raise ValueError(f"AI response missing required fields: {', '.join(missing_fields)}")


# Validate field types

if not isinstance(analysis_data["can_wash_together"], bool):

raise ValueError("Invalid can_wash_together value - must be boolean")


if len(analysis_data.get("reasoning", "")) < 20:

raise ValueError("AI reasoning too short - insufficient analysis")


return WashingRecommendation(

can_wash_together=analysis_data["can_wash_together"],

temperature=analysis_data["temperature"],

cycle=analysis_data["cycle"],

detergent_type=analysis_data["detergent_type"],

special_instructions=analysis_data.get("special_instructions", []),

separate_loads=analysis_data.get("separate_loads"),

reasoning=analysis_data["reasoning"]

)


except json.JSONDecodeError as json_error:

logger.error(f"JSON parsing error: {json_error}")

raise HTTPException(status_code=502, detail="AI response format error")

except ValueError as val_error:

logger.error(f"Validation error: {val_error}")

raise HTTPException(status_code=502, detail=f"AI response validation failed: {str(val_error)}")


except HTTPException:

raise

except Exception as e:

logger.error(f"Unexpected error in AI analysis: {str(e)}")

return get_detailed_mock_recommendation(analysis_type, user_notes)

def get_detailed_mock_recommendation(analysis_type: str, user_notes: str = None) -> WashingRecommendation:

"""Provide detailed mock recommendations"""

if analysis_type == "wash_tag":

return WashingRecommendation(

can_wash_together=True,

temperature="as indicated on label",

cycle="normal",

detergent_type="regular",

special_instructions=[

"Follow care symbols on the label exactly",

"Check water temperature symbols carefully",

"Look for drying and ironing instructions",

"Pay attention to bleaching restrictions"

],

reasoning="Care label analysis shows specific washing symbols that indicate manufacturer's recommended care instructions. Following these symbols ensures garment quality and prevents damage."

)

else:

return WashingRecommendation(

can_wash_together=False,

temperature="cold",

cycle="normal",

detergent_type="color-safe",

special_instructions=[

"Separate lights and darks to prevent color bleeding",

"Turn jeans and dark items inside out to preserve color",

"Use cold water to prevent shrinking and fading",

"Check all pockets for items before washing"

],

separate_loads=[

{

"items": ["dark items", "jeans"],

"settings": "Wash together with cold water and color-safe detergent on normal cycle"

},

{

"items": ["light colored items"],

"settings": "Wash separately with regular detergent in cold or warm water"

}

],

reasoning="The items include different fabric types and colors requiring separation to prevent damage. Dark colors can bleed onto lighter fabrics, especially in warm water."

)

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)

async def analyze_laundry(request: LaundryAnalysisRequest):

"""Analyze laundry images with strict validation"""

try:

# Additional runtime validation

if not hasattr(request, 'images') or not request.images:

raise HTTPException(status_code=422, detail="Images field is required and cannot be empty")


# Validate request size

request_size = sum(len(img) for img in request.images)

if request_size > 50000000: # 50MB total limit

raise HTTPException(status_code=413, detail="Request too large - reduce image sizes")


# Get recommendation with comprehensive error handling

try:

recommendation = await analyze_with_openai(request.images, request.analysis_type, request.user_notes)

except HTTPException:

raise

except Exception as analysis_error:

logger.error(f"Analysis error: {str(analysis_error)}")

raise HTTPException(status_code=500, detail="Internal analysis error - please try again")


# Create response with validation

try:

analysis_response = LaundryAnalysisResponse(

recommendation=recommendation,

items_analyzed=["analyzed_item"]

)

except Exception as response_error:

logger.error(f"Response creation error: {str(response_error)}")

raise HTTPException(status_code=500, detail="Failed to create analysis response")


return analysis_response

except HTTPException:

raise

except Exception as e:

logger.error(f"Unexpected error in analyze_laundry: {str(e)}")

raise HTTPException(status_code=500, detail="Internal server error - please contact support")

@app.get("/api/analysis-history")

async def get_analysis_history(limit: int = 10):

"""Get analysis history with parameter validation"""

try:

# Validate limit parameter

if limit < 1:

raise HTTPException(status_code=422, detail="Limit must be at least 1")

if limit > 100:

raise HTTPException(status_code=422, detail="Limit cannot exceed 100")


return {

"analyses": [],

"message": "History feature available with database",

"limit": limit,

"total": 0

}

except HTTPException:

raise

except Exception as e:

logger.error(f"Error in analysis history: {str(e)}")

raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

@app.get("/api/")

async def root():

"""Health check with comprehensive status"""

try:

return {

"message": "Wash This?? API is running",

"version": "1.0.0",

"ai_enabled": bool(OPENAI_API_KEY),

"mode": "ai" if OPENAI_API_KEY else "detailed_mock",

"status": "healthy",

"timestamp": datetime.utcnow().isoformat()

}

except Exception as e:

logger.error(f"Health check error: {str(e)}")

raise HTTPException(status_code=500, detail="Service health check failed")

@app.get("/")

async def home():

return {

"message": "Wash This?? Backend",

"docs": "/docs",

"api": "/api/",

"version": "1.0.0"

}

# Enhanced error handlers

@app.exception_handler(422)

async def validation_exception_handler(request, exc):

return JSONResponse(

status_code=422,

content={

"detail": "Validation error - please check your request format",

"errors": str(exc),

"timestamp": datetime.utcnow().isoformat()

}

)

@app.exception_handler(413)

async def payload_too_large_handler(request, exc):

return JSONResponse(

status_code=413,

content={

"detail": "Request too large - please reduce image sizes",

"max_size": "10MB per image, 50MB total",

"timestamp": datetime.utcnow().isoformat()

}

)

# CORS configuration

app.add_middleware(

CORSMiddleware,

allow_origins=["*"],

allow_credentials=True,

allow_methods=["GET", "POST", "OPTIONS"],

allow_headers=["*"],

max_age=86400,

)

@app.get("/health")

async def health_check():

"""Detailed health check endpoint"""

try:

return {

"status": "healthy",

"timestamp": datetime.utcnow().isoformat(),

"ai_configured": bool(OPENAI_API_KEY),

"version": "1.0.0",

"environment": "production"

}

except Exception as e:

logger.error(f"Health check failed: {str(e)}")

raise HTTPException(status_code=503, detail="Service temporarily unavailable")

if __name__ == "__main__":

import uvicorn

uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
