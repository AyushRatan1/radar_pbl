"""
SAR Image Super-Resolution API
FastAPI backend for automotive radar image enhancement
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import base64
import time
from typing import Optional

from models import SARSuperResolution

# Initialize FastAPI app
app = FastAPI(
    title="SAR Super-Resolution API",
    description="Deep learning-based super-resolution for automotive SAR/radar images",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the super-resolution model
sr_model = SARSuperResolution(scale_factor=4)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


@app.get("/")
async def root():
    """API health check and info"""
    return {
        "status": "online",
        "service": "SAR Super-Resolution API",
        "version": "1.0.0",
        "model": "ESPCN (4x upscaling)",
        "description": "Resolution enhancement for automotive SAR images"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Enhance a SAR image using super-resolution
    
    Args:
        file: Uploaded image file (supports PNG, JPEG, TIFF)
        
    Returns:
        JSON with original and enhanced images as base64
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/tiff", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    try:
        # Read and process the image
        start_time = time.time()
        
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Get original dimensions
        original_width, original_height = input_image.size
        
        # Perform super-resolution
        upscaled_original, enhanced_image = sr_model.enhance_with_comparison(input_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get enhanced dimensions
        enhanced_width, enhanced_height = enhanced_image.size
        
        # Convert images to base64 for JSON response
        original_base64 = image_to_base64(upscaled_original)
        enhanced_base64 = image_to_base64(enhanced_image)
        
        return JSONResponse({
            "success": True,
            "processing_time_ms": round(processing_time * 1000, 2),
            "original": {
                "width": original_width,
                "height": original_height,
                "image": original_base64
            },
            "enhanced": {
                "width": enhanced_width,
                "height": enhanced_height,
                "image": enhanced_base64
            },
            "scale_factor": 4,
            "model": "ESPCN"
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/enhance/download")
async def enhance_and_download(file: UploadFile = File(...)):
    """
    Enhance a SAR image and return the enhanced image directly
    
    Args:
        file: Uploaded image file
        
    Returns:
        Enhanced image as PNG
    """
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Perform super-resolution
        _, enhanced_image = sr_model.enhance_with_comparison(input_image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=enhanced_sar.png"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "ESPCN",
        "architecture": "Efficient Sub-Pixel Convolutional Neural Network",
        "scale_factor": 4,
        "input_channels": 1,
        "description": "Lightweight CNN optimized for real-time super-resolution",
        "use_case": "Automotive SAR/Radar image enhancement",
        "benefits": [
            "4x resolution enhancement",
            "Real-time processing capability",
            "Optimized for grayscale radar imagery",
            "Low memory footprint"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
