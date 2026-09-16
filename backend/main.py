from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from datetime import datetime
from routes import prices, predictions, calculator, news
from config import config

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except:
    HAS_GEMINI = False

app = FastAPI(
    title="Gold Master - AI Trading Hub",
    description="Advanced AI-powered Forex & Commodities analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prices.router)
app.include_router(predictions.router)
app.include_router(calculator.router)
app.include_router(news.router)

# Configure Gemini API
if config.GEMINI_API_KEY and HAS_GEMINI:
    genai.configure(api_key=config.GEMINI_API_KEY)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": "Gold Master - AI Trading Hub",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "documentation": "/docs",
            "prices": "/api/prices",
            "predictions": "/api/predictions",
            "calculator": "/api/calculator",
            "news": "/api/news",
            "ai": "/api/ai"
        },
        "available_features": [
            "Real-time price data for XAUUSD and USDJPY",
            "AI-powered price predictions",
            "Risk management calculator",
            "Market news and sentiment analysis",
            "Technical analysis with multiple indicators",
            "Economic calendar and bank news"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/ai/analyze")
async def ai_analysis(symbol: str = "XAUUSD", prompt: str = ""):
    """AI analysis using Gemini API"""
    try:
        if not config.GEMINI_API_KEY or not HAS_GEMINI:
            return {
                "response": "Gemini API not configured. Please set GEMINI_API_KEY environment variable.",
                "model": "gemini-1.5-flash",
                "disclaimer": "AI analysis requires API key"
            }
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = f"""You are an expert in macroeconomics, forex trading, and commodities analysis.
        Focus on {symbol} analysis. Provide:
        1. Current technical setup
        2. Key levels (support/resistance)
        3. Risk factors
        4. Potential scenarios
        5. Risk management advice
        
        Keep response concise and professional for traders."""
        
        full_prompt = f"{system_prompt}\\n\\nAnalysis Request: {prompt if prompt else f'Provide current analysis for {symbol}'}"
        
        response = model.generate_content(full_prompt)
        
        return {
            "symbol": symbol,
            "response": response.text if response else "Unable to generate analysis",
            "model": "gemini-1.5-flash",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

@app.post("/api/ai/predict-sentiment")
async def predict_sentiment(text: str):
    """Predict sentiment of market news"""
    try:
        if not config.GEMINI_API_KEY or not HAS_GEMINI:
            return {"error": "Gemini API not configured"}
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Analyze this market news for forex/gold trading impact.
        Rate sentiment as: BULLISH, BEARISH, or NEUTRAL
        Score: -1.0 (Very Bearish) to +1.0 (Very Bullish)
        Explain impact on XAUUSD and USDJPY
        
        News: {text}
        
        Response format: SENTIMENT | SCORE | EXPLANATION"""
        
        response = model.generate_content(prompt)
        
        return {
            "analysis": response.text if response else "Unable to analyze",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
