#!/usr/bin/env python3
"""
Agricultural Advisor Bot - Handles both policy and weather queries
Uses existing NLP pipeline and weather service
"""

import os
import json
import re
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from improved_policy_chatbot import ImprovedPolicyChatbot
from nlp_pipeline.pipeline import QueryProcessingPipeline
from weather_service import WeatherService, LocationInfo
from init_mandi_soil import AgriculturalDataManager
from dotenv import load_dotenv
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryClassifier:
    """NLP-based query classifier using existing pipeline"""
    
    def __init__(self):
        # Initialize the existing NLP pipeline
        self.nlp_pipeline = QueryProcessingPipeline(
            use_transformer=True,
            use_spacy=True,
            use_semantic=True,
            use_zero_shot=True
        )
        
        # Define intent mappings for our categories
        self.intent_mappings = {
            'weather': ['weather_query', 'weather_inquiry', 'climate_question', 'temperature_question', 'rainfall_question'],
            'policy': ['policy_query', 'policy_inquiry', 'scheme_question', 'subsidy_question', 'government_help'],
            'price': ['price_query', 'market_question', 'price_inquiry', 'mandi_question'],
            'technical': ['technical_support', 'equipment_question', 'technology_question', 'repair_question'],
            'general': ['general_inquiry', 'basic_question', 'information_question'],
            'agriculture': ['crop_advice', 'crop_question', 'farming_advice', 'soil_question', 'pest_question', 'irrigation_question']
        }
    
    def classify_query(self, query: str) -> str:
        """Classify the user query using NLP pipeline"""
        try:
            # Process query through NLP pipeline
            result = self.nlp_pipeline.process_query(query)
            
            # Get primary intent
            primary_intent = result.primary_intent.lower()
            
            # Map intent to our categories
            for category, intents in self.intent_mappings.items():
                if any(intent in primary_intent for intent in intents):
                    return category
            
            # Fallback classification based on keywords
            return self._fallback_classification(query)
            
        except Exception as e:
            logger.error(f"Error in NLP classification: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> str:
        """Fallback classification using keywords"""
        query_lower = query.lower()
        
        weather_keywords = [
            'weather', 'temperature', 'rain', 'rainfall', 'drought', 'flood',
            'humidity', 'wind', 'climate', 'forecast', 'seasonal', 'monsoon',
            'hot', 'cold', 'dry', 'wet', 'storm', 'cyclone', 'heat wave',
            'frost', 'hail', 'snow', 'sunny', 'cloudy', 'overcast', 'mausam', 'baarish'
        ]
        
        policy_keywords = [
            'policy', 'scheme', 'subsidy', 'loan', 'insurance', 'support',
            'government', 'pm kisan', 'pmksy', 'soil health', 'mandi',
            'procurement', 'msp', 'fertilizer', 'seed', 'equipment',
            'guidelines', 'procedure', 'application', 'eligibility',
            'benefit', 'assistance', 'fund', 'grant', 'certificate', 'yojana', 'sarkar'
        ]
        
        price_keywords = [
            'price', 'rate', 'cost', 'worth', 'value', 'mandi', 'market', 'bhav', 'dam',
            'mulya', 'keemat', 'rupees', 'rs', 'quintal', 'ton', 'kg', 'per', 'auction'
        ]
        
        technical_keywords = [
            'tractor', 'equipment', 'machine', 'system', 'technology', 'repair', 'maintenance',
            'digital', 'automated', 'troubleshoot', 'fix', 'technical'
        ]
        
        general_keywords = [
            'what is', 'tell me about', 'information about', 'guide for', 'basic', 
            'how to start', 'beginner', 'overview', 'introduction', 'general'
        ]
        
        agriculture_keywords = [
            'crop', 'farming', 'agriculture', 'soil', 'fertilizer', 'pesticide',
            'irrigation', 'harvest', 'planting', 'seeding', 'pest', 'disease',
            'yield', 'production', 'storage', 'transport', 'organic', 'traditional', 'modern'
        ]
        
        # Count keyword matches
        weather_score = sum(1 for keyword in weather_keywords if keyword in query_lower)
        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        price_score = sum(1 for keyword in price_keywords if keyword in query_lower)
        technical_score = sum(1 for keyword in technical_keywords if keyword in query_lower)
        general_score = sum(1 for keyword in general_keywords if keyword in query_lower)
        agriculture_score = sum(1 for keyword in agriculture_keywords if keyword in query_lower)
        
        # Classification logic with priority
        if weather_score > 0:
            return "weather"
        elif policy_score > 0:
            return "policy"
        elif price_score > 0:
            return "price"
        elif technical_score > 0:
            return "technical"
        elif general_score > 0:
            return "general"
        elif agriculture_score > 0:
            return "agriculture"
        else:
            return "general"

class GroqAgriculturalAdvisor:
    """Groq-powered agricultural advisor"""
    
    def __init__(self, api_key: str = None):
        # Use provided api_key, or get from environment variable
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_weather_advice(self, query: str, current_weather, forecast=None) -> str:
        """Generate agricultural advice based on weather data"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Build weather information string
            weather_info = f"""Current Weather Conditions:
- Temperature: {current_weather.temperature_avg:.1f}°C (Max: {current_weather.temperature_max:.1f}°C, Min: {current_weather.temperature_min:.1f}°C)
- Humidity: {current_weather.humidity:.1f}%
- Weather: {current_weather.description}
- Wind Speed: {current_weather.wind_speed:.1f} km/h, Direction: {current_weather.wind_direction}
- Precipitation: {current_weather.precipitation:.1f} mm
- Pressure: {current_weather.pressure:.1f} hPa
- UV Index: {current_weather.uv_index:.1f}"""

            if forecast:
                forecast_info = "\n\n7-Day Weather Forecast:"
                for i, day in enumerate(forecast[:3], 1):  # Show next 3 days
                    forecast_info += f"\nDay {i}: {day.date} - {day.temperature_avg:.1f}°C, {day.description}, {day.precipitation_amount:.1f}mm rain"
            else:
                forecast_info = "\n\nForecast: Not available"

            prompt = f"""You are an expert agricultural advisor. Based on the current weather conditions and forecast, provide specific agricultural advice to the farmer.

{weather_info}{forecast_info}

Farmer's Question: {query}

Instructions:
1. Analyze the current weather conditions and their impact on agriculture
2. Consider the weather forecast for planning agricultural activities
3. Provide specific, actionable advice for the farmer
4. Consider crop-specific recommendations if mentioned
5. Include timing suggestions for agricultural activities
6. Mention any precautions or warnings based on weather
7. Be encouraging and supportive
8. Use simple, understandable language
9. Structure your response with clear sections
10. Focus on practical farming decisions

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledgeable and caring agricultural advisor with expertise in weather-based farming recommendations. Provide practical, science-based advice that helps farmers make informed decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 800
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating weather advice: {e}")
            return f"❌ Error generating advice: {e}"
    
    def generate_weather_advice_comprehensive(self, query: str, location_info, current_weather, forecast_data, agricultural_insights) -> str:
        """Generate comprehensive agricultural advice based on weather data"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Build comprehensive weather information
            location_str = f"Location: {location_info['name']}, {location_info['state']}, {location_info['country']}"
            
            current_weather_str = ""
            if current_weather:
                current_weather_str = f"""Current Weather:
- Temperature: {current_weather['temperature']['avg']:.1f}°C (Max: {current_weather['temperature']['max']:.1f}°C, Min: {current_weather['temperature']['min']:.1f}°C)
- Humidity: {current_weather['humidity']:.1f}%
- Weather: {current_weather['condition']}
- Wind Speed: {current_weather['wind']['speed']:.1f} km/h, Direction: {current_weather['wind']['direction']}
- Precipitation: {current_weather['precipitation']:.1f} mm"""

            forecast_str = ""
            if forecast_data:
                forecast_str = "\n\n7-Day Weather Forecast:"
                for i, day in enumerate(forecast_data[:3], 1):  # Show next 3 days
                    forecast_str += f"\nDay {i}: {day['date']} - {day['temperature']['avg']:.1f}°C, {day['condition']}, {day['precipitation']['amount']:.1f}mm rain"

            insights_str = ""
            if agricultural_insights:
                try:
                    insights_str = "Agricultural Insights:\n"
                    
                    if 'soil_moisture' in agricultural_insights:
                        soil_moisture = agricultural_insights['soil_moisture']
                        if 'status' in soil_moisture:
                            insights_str += f"- Soil Moisture: {soil_moisture['status']}"
                            if 'risk' in soil_moisture:
                                insights_str += f" ({soil_moisture['risk']})"
                            insights_str += "\n"
                        
                        if 'recent_precipitation' in soil_moisture:
                            insights_str += f"- Recent Precipitation: {soil_moisture['recent_precipitation']:.1f} mm\n"
                        
                        if 'forecast_precipitation' in soil_moisture:
                            insights_str += f"- Forecast Precipitation: {soil_moisture['forecast_precipitation']:.1f} mm\n"
                    
                    if 'crop_health' in agricultural_insights and 'temperature_stress' in agricultural_insights['crop_health']:
                        insights_str += f"- Crop Health: {agricultural_insights['crop_health']['temperature_stress']} temperature stress\n"
                    
                    if 'irrigation_needs' in agricultural_insights and 'status' in agricultural_insights['irrigation_needs']:
                        insights_str += f"- Irrigation Needs: {agricultural_insights['irrigation_needs']['status']}\n"
                        
                except Exception as e:
                    logger.warning(f"Error formatting insights for Groq: {e}")
                    insights_str = "Agricultural Insights: Available but format may vary\n"

            prompt = f"""You are an expert agricultural advisor. Based on the comprehensive weather data and agricultural insights, provide specific agricultural advice to the farmer.

{location_str}

{current_weather_str}{forecast_str}

{insights_str}

Farmer's Question: {query}

Instructions:
1. Analyze the current weather conditions and their impact on agriculture
2. Consider the weather forecast for planning agricultural activities
3. Use the agricultural insights to provide targeted advice
4. Provide specific, actionable advice for the farmer
5. Consider crop-specific recommendations if mentioned
6. Include timing suggestions for agricultural activities
7. Mention any precautions or warnings based on weather
8. Address soil moisture and irrigation needs
9. Be encouraging and supportive
10. Use simple, understandable language
11. Structure your response with clear sections
12. Focus on practical farming decisions

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledgeable and caring agricultural advisor with expertise in weather-based farming recommendations. Provide practical, science-based advice that helps farmers make informed decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 800
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating comprehensive weather advice: {e}")
            return f"❌ Error generating advice: {e}"
        
    def generate_sql(self, user_query: str) -> str:
        """Translate natural language agricultural price queries into SQL"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            prompt = f"""
    You are an assistant that translates natural language agricultural price questions into SQL queries.
    The database is SQLite with a table `mandi_prices`.

    Columns:
    - State (TEXT)
    - District (TEXT)
    - Market (TEXT)
    - Commodity (TEXT)
    - Variety (TEXT)
    - Arrival_Date (TEXT in YYYY-MM-DD format)
    - Min_Price (INTEGER)
    - Max_Price (INTEGER)
    - Modal_Price (INTEGER)

    Return only SQL code without explanation.

    User query: "{user_query}"
    """

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an expert SQL generator. Always return valid SQLite SQL code only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 300
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return f"❌ Error generating SQL: {e}"

    
    def generate_general_advice(self, query: str) -> str:
        """Generate general agricultural advice"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            prompt = f"""You are an expert agricultural advisor. Provide helpful advice to the farmer's question.

Farmer's Question: {query}

Instructions:
1. Provide practical, science-based agricultural advice
2. Consider Indian farming context and conditions
3. Include specific recommendations when possible
4. Be encouraging and supportive
5. Use simple, understandable language
6. Structure your response clearly
7. If the question is about crops, mention suitable varieties and practices
8. If about soil, mention testing and improvement methods
9. If about pests/diseases, mention prevention and treatment

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledgeable and caring agricultural advisor with expertise in Indian farming practices. Provide practical, science-based advice that helps farmers improve their agricultural practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 600
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating general advice: {e}")
            return f"❌ Error generating advice: {e}"

class AgriculturalAdvisorBot:
    """Main agricultural advisor bot"""
    
    def __init__(self):
        self.classifier = QueryClassifier()
        self.weather_service = WeatherService()  # Using existing weather service
        self.groq_advisor = GroqAgriculturalAdvisor()
        
        # Initialize policy chatbot with proper database loading
        self.policy_chatbot = ImprovedPolicyChatbot()
        
        # Check if policy database is loaded
        if not self.policy_chatbot.is_loaded:
            logger.warning("Policy database not loaded. Attempting to load from default location...")
            # Try to load the database
            if os.path.exists("improved_vector_db"):
                self.policy_chatbot = ImprovedPolicyChatbot(db_dir="improved_vector_db")
                if self.policy_chatbot.is_loaded:
                    logger.info("Policy database loaded successfully!")
                else:
                    logger.error("Failed to load policy database from improved_vector_db")
            else:
                logger.error("Policy database directory 'improved_vector_db' not found")
        
        # Initialize agricultural data manager
        self.data_manager = AgriculturalDataManager()
        
        self.user_city = None
        self.user_crop = None

    def get_latest_prices_all_markets(self, city: str, crop: str, db_path="agri_data.db"):
        """Get the most recent price entry for each market in a given city and crop"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
        SELECT Market, MAX(Arrival_Date), Modal_Price
        FROM mandi_prices
        WHERE District LIKE ? AND Commodity LIKE ?
        GROUP BY Market
        ORDER BY MAX(Arrival_Date) DESC;
        """

        cursor.execute(query, (f"%{city}%", f"%{crop}%"))
        results = cursor.fetchall()
        conn.close()

        # Returns list of (market, date, price)
        return results

    
    def process_query(self, query: str) -> str:
        """Process user query and provide appropriate response"""
        # Classify the query
        query_type = self.classifier.classify_query(query)
        
        if query_type == "weather":
            return self._handle_weather_query(query)
        elif query_type == "policy":
            return self._handle_policy_query(query)
        elif query_type == "price":
            return self._handle_price_query(query)
        elif query_type == "technical":
            return self._handle_technical_query(query)
        elif query_type == "general":
            return self._handle_general_query(query)
        elif query_type == "agriculture":
            return self._handle_agriculture_query(query)
        else:
            return self._handle_general_query(query)
    
    def _handle_weather_query(self, query: str) -> str:
        """Handle weather-related queries"""
        intent_info = f"🎯 **Detected Intent: Weather Query**\n\n"
        
        if not self.user_city:
            return intent_info + "📍 Please set your city first using: 'city [cityname]' (e.g., 'city Mumbai')"
        
        try:
            # Get comprehensive weather report using the existing weather service
            print(f"🔍 Fetching weather data for: {self.user_city}")
            report = self.weather_service.get_comprehensive_weather_report(self.user_city)
            
            if 'error' in report:
                error_msg = report['error']
                if 'Could not find location' in error_msg:
                    return f"❌ Sorry, I couldn't find weather data for '{self.user_city}'. Please check the city name and try again.\n\n💡 Try using a major city name like 'Mumbai', 'Delhi', 'Bangalore', etc."
                else:
                    return f"❌ Sorry, I couldn't fetch weather data for {self.user_city}. Error: {error_msg}"
            
            # Extract weather data from the report with error handling
            try:
                location_info = report.get('location', {})
                historical_data = report.get('historical_data', [])
                forecast_data = report.get('forecast_data', [])
                agricultural_insights = report.get('agricultural_insights', {})
                
                # Get current weather (latest from historical data)
                current_weather = historical_data[-1] if historical_data else None
                
                logger.info(f"Successfully extracted weather data for {location_info.get('name', 'Unknown location')}")
                
            except Exception as e:
                logger.error(f"Error extracting weather data from report: {e}")
                return f"❌ Error processing weather data: {e}"
            
            # Generate advice using Groq with comprehensive data
            advice = self.groq_advisor.generate_weather_advice_comprehensive(
                query, location_info, current_weather, forecast_data, agricultural_insights
            )
            
            # Format response
            response = f"🌤️ **Weather-Based Agricultural Advice for {location_info['name']}**\n\n"
            
            # Current weather
            if current_weather:
                response += f"📊 **Current Weather:**\n"
                try:
                    temp_data = current_weather.get('temperature', {})
                    response += f"• Temperature: {temp_data.get('avg', 0):.1f}°C"
                    if 'max' in temp_data and 'min' in temp_data:
                        response += f" (Max: {temp_data['max']:.1f}°C, Min: {temp_data['min']:.1f}°C)"
                    response += "\n"
                    
                    response += f"• Humidity: {current_weather.get('humidity', 0):.1f}%\n"
                    response += f"• Weather: {current_weather.get('condition', 'Unknown').title()}\n"
                    
                    wind_data = current_weather.get('wind', {})
                    response += f"• Wind Speed: {wind_data.get('speed', 0):.1f} km/h\n"
                    
                    response += f"• Precipitation: {current_weather.get('precipitation', 0):.1f} mm\n\n"
                except Exception as e:
                    logger.warning(f"Error formatting current weather: {e}")
                    response += "• Current weather data available but format may vary\n\n"
            
            # Forecast summary
            if forecast_data:
                response += f"📅 **7-Day Forecast Summary:**\n"
                try:
                    avg_temp = sum([f.get('temperature', {}).get('avg', 0) for f in forecast_data]) / len(forecast_data)
                    total_precip = sum([f.get('precipitation', {}).get('amount', 0) for f in forecast_data])
                    response += f"• Average Temperature: {avg_temp:.1f}°C\n"
                    response += f"• Total Precipitation: {total_precip:.1f} mm\n"
                    
                    if forecast_data and 'wind' in forecast_data[0]:
                        wind_data = forecast_data[0]['wind']
                        response += f"• Wind Conditions: {wind_data.get('direction', 'Unknown')} at {wind_data.get('speed', 0):.1f} km/h\n"
                    response += "\n"
                    
                except Exception as e:
                    logger.warning(f"Error formatting forecast summary: {e}")
                    response += "• Forecast data available but format may vary\n\n"
            
            # Agricultural insights
            if agricultural_insights:
                response += f"🌾 **Agricultural Insights:**\n"
                try:
                    if 'soil_moisture' in agricultural_insights and 'status' in agricultural_insights['soil_moisture']:
                        response += f"• Soil Moisture: {agricultural_insights['soil_moisture']['status']}"
                        if 'risk' in agricultural_insights['soil_moisture']:
                            response += f" ({agricultural_insights['soil_moisture']['risk']})"
                        response += "\n"
                    
                    if 'crop_health' in agricultural_insights and 'temperature_stress' in agricultural_insights['crop_health']:
                        response += f"• Crop Health: {agricultural_insights['crop_health']['temperature_stress']} temperature stress\n"
                    
                    if 'irrigation_needs' in agricultural_insights and 'status' in agricultural_insights['irrigation_needs']:
                        response += f"• Irrigation Needs: {agricultural_insights['irrigation_needs']['status']}\n"
                except Exception as e:
                    logger.warning(f"Error formatting agricultural insights: {e}")
                    response += "• Agricultural insights available but format may vary\n"
                response += "\n"
            
            response += f"🤖 **AI Agricultural Advice:**\n{advice}"
            
            # Add source attribution
            sources = f"\n📚 **Sources:**\n"
            sources += f"• Weather Data: Open-Meteo API (Real-time)\n"
            sources += f"• Location Data: Geocoding API\n"
            sources += f"• AI Analysis: Groq API (Llama3-8b-8192 model)\n"
            sources += f"• Agricultural Insights: Weather-based calculations\n"
            
            return intent_info + response + sources
            
        except Exception as e:
            logger.error(f"Error handling weather query: {e}")
            return intent_info + f"❌ Error processing weather query: {e}"
            
            return response + sources
            
        except Exception as e:
            logger.error(f"Error handling weather query: {e}")
            return f"❌ Error processing weather query: {e}"
    
    def _handle_price_query(self, query: str) -> str:
        """Handle mandi price queries using LLM-based SQL generation."""
        try:
            # Step 1: Ask Groq to generate SQL
            sql_query = self.groq_advisor.generate_sql(query).strip()

            logger.info(f"Groq SQL: {sql_query}")

            # Step 2: Try executing the SQL
            conn = sqlite3.connect(self.db.db_path)
            try:
                results = conn.execute(sql_query).fetchall()
            except Exception as e:
                logger.warning(f"Groq SQL failed: {e}")
                results = []

            conn.close()

            # Step 3: If Groq SQL worked
            if results:
                return self.groq_advisor.generate_general_advice(
                    f"{query}\nData retrieved from the database:\n{results}\n"
                    f"Please summarize this for farmers in simple words using the data retrieved."
                )

            # Step 4: Fallback → general safe query
            else:
                city = self.user_city(query)
                crop = self.user_crop(query)
                if not city or not crop:
                    return "Please specify both city and crop for accurate price information."

                fallback_results = self.get_latest_prices_all_markets(city, crop)
                if not fallback_results:
                    return f"No data found for {crop} in {city}."

                formatted = "\n".join([
                    f"{market}: ₹{price}/quintal ({date})"
                    for market, date, price in fallback_results
                ])
                return self.groq_advisor.generate_general_advice(
                    f"{query}. "
                    f"Here’s price data for {crop} in {city}:\n{formatted}\n"
                    f"Summarize in farmer-friendly advice."
                )

        except Exception as e:
            logger.error(f"Error handling price query: {e}")
            return "I could not fetch the price data right now. Please try again later."


    def _handle_policy_query(self, query: str) -> str:
        """Handle policy-related queries"""
        intent_info = f"🎯 **Detected Intent: Policy Query**\n\n"
        
        if not self.policy_chatbot.is_loaded:
            # Fallback to general advice when policy database is not available
            logger.warning("Policy database not loaded, falling back to general advice")
            return intent_info + self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India."
            )
        
        try:
            # Use Groq for better policy responses
            policy_response = self.policy_chatbot.ask_question_with_groq(query)
            
            # Add source attribution
            sources = f"\n📚 **Sources:**\n"
            sources += f"• Policy Documents: `pdfs/` directory (12 PDF files)\n"
            sources += f"• Vector Database: `improved_vector_db/` (973 sections)\n"
            sources += f"• AI Processing: Groq API (Llama3-8b-8192 model)\n"
            sources += f"• Documents: PM Kisan, PMKSY, Soil Health Card, Crop Insurance, etc.\n"
            
            return intent_info + policy_response + sources
        except Exception as e:
            logger.error(f"Error handling policy query: {e}")
            # Fallback to general advice
            return intent_info + self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India."
            )
    
    def _handle_agriculture_query(self, query: str) -> str:
        """Handle general agricultural queries"""
        intent_info = f"🎯 **Detected Intent: Agriculture Query**\n\n"
        
        # Check if query is about soil health
        query_lower = query.lower()
        soil_keywords = ["soil", "ph", "nitrogen", "phosphorus", "potassium", "organic carbon"]
        
        if any(keyword in query_lower for keyword in soil_keywords):
            return intent_info + self._handle_soil_health_query(query)
        
        # Check if query is about crop recommendations
        crop_keywords = ["crop", "plant", "grow", "suitable", "recommend"]
        if any(keyword in query_lower for keyword in crop_keywords):
            return intent_info + self._handle_crop_recommendation_query(query)
        
        # Default to general agricultural advice
        ai_advice = self.groq_advisor.generate_general_advice(query)
        
        # Add source attribution
        sources = f"\n📚 **Sources:**\n"
        sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice + sources
    
    def _handle_soil_health_query(self, query: str) -> str:
        """Handle soil health related queries"""
        location = self.user_city or "Kanpur"  # Default location
        
        # Map common city names to district names in our database
        location_mapping = {
            "kanpur": "Kanpur Nagar",
            "kannauj": "Kannauj", 
            "agra": "Agra",
            "unnao": "Unnao",
            "lucknow": "Lucknow"
        }
        
        # Try to find the correct district name
        search_location = location_mapping.get(location.lower(), location)
        
        try:
            soil_result = self.data_manager.get_soil_health(search_location)
            
            if "error" in soil_result:
                return f"❌ {soil_result['error']}\n\n🤖 **General Soil Advice:**\n{self.groq_advisor.generate_general_advice(query)}"
            
            # Combine soil data with AI advice
            ai_advice = self.groq_advisor.generate_general_advice(
                f"Soil health question: {query}. Based on soil data: pH {soil_result['ph']}, Organic Carbon {soil_result['organic_carbon']}%, N {soil_result['nitrogen']} kg/ha, P {soil_result['phosphorus']} kg/ha, K {soil_result['potassium']} kg/ha. Please provide soil management advice."
            )
            
            # Add source attribution
            sources = f"\n📚 **Sources:**\n"
            sources += f"• Soil Data: `soil_health.csv` (5 districts)\n"
            sources += f"• Database: `agri_data.db` (SQLite)\n"
            sources += f"• AI Advice: Groq API (Llama3-8b-8192 model)\n"
            
            return f"🌱 **Soil Health Data for {location}:**\n{soil_result['formatted']}\n\n🤖 **Soil Management Advice:**\n{ai_advice}{sources}"
            
        except Exception as e:
            logger.error(f"Error handling soil health query: {e}")
            return f"❌ Error retrieving soil data: {str(e)}\n\n🤖 **General Soil Advice:**\n{self.groq_advisor.generate_general_advice(query)}"
    
    def _handle_crop_recommendation_query(self, query: str) -> str:
        """Handle crop recommendation queries"""
        location = self.user_city or "Kanpur"  # Default location
        
        # Map common city names to district names in our database
        location_mapping = {
            "kanpur": "Kanpur Nagar",
            "kannauj": "Kannauj", 
            "agra": "Agra",
            "unnao": "Unnao",
            "lucknow": "Lucknow"
        }
        
        # Try to find the correct district name
        search_location = location_mapping.get(location.lower(), location)
        
        try:
            # Get soil data for recommendations
            soil_result = self.data_manager.get_soil_health(search_location)
            available_crops = self.data_manager.get_available_crops(location)
            
            soil_info = ""
            if "error" not in soil_result:
                soil_info = f"Based on soil data: pH {soil_result['ph']}, Organic Carbon {soil_result['organic_carbon']}%, N {soil_result['nitrogen']} kg/ha, P {soil_result['phosphorus']} kg/ha, K {soil_result['potassium']} kg/ha. "
            
            crop_info = ""
            if available_crops:
                crop_info = f"Commonly grown crops in {location}: {', '.join(available_crops[:5])}. "
            
            # Generate AI recommendation
            ai_advice = self.groq_advisor.generate_general_advice(
                f"Crop recommendation question: {query}. {soil_info}{crop_info}Please provide crop recommendations and farming advice."
            )
            
            response = f"🌾 **Crop Information for {location}:**\n"
            if available_crops:
                response += f"• Available crops: {', '.join(available_crops[:5])}\n"
            if "error" not in soil_result:
                response += f"• Soil conditions: pH {soil_result['ph']}, Organic Carbon {soil_result['organic_carbon']}%\n"
            response += f"\n🤖 **Crop Recommendations:**\n{ai_advice}"
            
            # Add source attribution
            sources = f"\n📚 **Sources:**\n"
            sources += f"• Crop Data: `mandi_prices.csv` (35,522 records)\n"
            sources += f"• Soil Data: `soil_health.csv` (5 districts)\n"
            sources += f"• Database: `agri_data.db` (SQLite)\n"
            sources += f"• AI Recommendations: Groq API (Llama3-8b-8192 model)\n"
            
            return response + sources
            
        except Exception as e:
            logger.error(f"Error handling crop recommendation query: {e}")
            return f"❌ Error retrieving crop data: {str(e)}\n\n🤖 **General Crop Advice:**\n{self.groq_advisor.generate_general_advice(query)}"
    
    def _handle_price_query(self, query: str) -> str:
        """Handle price-related queries"""
        intent_info = f"🎯 **Detected Intent: Price Query**\n\n"
        
        # Extract crop and location from query
        crop, location = self._extract_crop_and_location(query)
        
        if not location:
            location = self.user_city or "Kanpur"  # Default to Kanpur if no city set
        
        if not crop:
            # If no specific crop mentioned, provide general price information
            return intent_info + self._get_general_price_info(location)
        
        # Get specific crop price information
        price_info = self._get_crop_price_info(crop, location)
        
        # Combine with AI-generated advice
        ai_advice = self.groq_advisor.generate_general_advice(
            f"Market price question: {query}. Based on the price data: {price_info}. Please provide additional market insights and pricing advice for farmers."
        )
        
        # Add source attribution
        sources = f"\n📚 **Sources:**\n"
        sources += f"• Price Data: `mandi_prices.csv` (35,522 records)\n"
        sources += f"• Database: `agri_data.db` (SQLite)\n"
        sources += f"• AI Insights: Groq API (Llama3-8b-8192 model)\n"
        
        return intent_info + f"📊 **Price Information:**\n{price_info}\n\n🤖 **AI Market Insights:**\n{ai_advice}{sources}"
    
    def _extract_crop_and_location(self, query: str) -> tuple:
        """Extract crop and location from query"""
        query_lower = query.lower()
        
        # Common crops - order matters for better matching
        crops = ["wheat", "rice", "maize", "cotton", "potato", "tomato", "sugarcane", "pulses", "oilseeds"]
        crop = None
        
        # More specific matching to avoid false positives
        for c in crops:
            # Check for exact word boundaries to avoid partial matches
            if f" {c} " in f" {query_lower} " or query_lower.startswith(c) or query_lower.endswith(c):
                crop = c.title()
                break
        
        # Common locations (cities in our database)
        locations = ["agra", "kannuj", "kanpur", "lucknow", "unnao", "mumbai", "delhi", "bangalore"]
        location = None
        for loc in locations:
            if loc in query_lower:
                location = loc.title()
                break
        
        return crop, location
    
    def _get_crop_price_info(self, crop: str, location: str) -> str:
        """Get specific crop price information"""
        try:
            # Get latest price
            price_result = self.data_manager.get_latest_price(location, crop)
            
            if "error" in price_result:
                return f"❌ {price_result['error']}"
            
            # Get price trends
            trend_result = self.data_manager.get_price_trends(location, crop, days=30)
            
            price_info = f"🌾 **{crop} Prices in {location}:**\n"
            price_info += f"• {price_result['formatted']}\n"
            
            if "error" not in trend_result:
                price_info += f"• {trend_result['formatted']}\n"
            
            return price_info
            
        except Exception as e:
            logger.error(f"Error getting price info: {e}")
            return f"❌ Error retrieving price information: {str(e)}"
    
    def _get_general_price_info(self, location: str) -> str:
        """Get general price information for a location"""
        try:
            # Get available crops
            available_crops = self.data_manager.get_available_crops(location)
            
            if not available_crops:
                return f"❌ No price data available for {location}"
            
            price_info = f"📊 **Available Crops in {location}:**\n"
            price_info += f"• {', '.join(available_crops[:5])}\n\n"
            price_info += "💡 **Tip:** Ask for specific crop prices like 'What is the price of wheat in Kanpur?'"
            
            return price_info
            
        except Exception as e:
            logger.error(f"Error getting general price info: {e}")
            return f"❌ Error retrieving price information: {str(e)}"
    
    def _handle_technical_query(self, query: str) -> str:
        """Handle technical support queries"""
        intent_info = f"🎯 **Detected Intent: Technical Support**\n\n"
        ai_advice = self.groq_advisor.generate_general_advice(
            f"Technical question: {query}. Please provide information about agricultural equipment, technology, maintenance, and technical solutions for farming."
        )
        
        # Add source attribution
        sources = f"\n📚 **Sources:**\n"
        sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice + sources
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        intent_info = f"🎯 **Detected Intent: General Inquiry**\n\n"
        ai_advice = self.groq_advisor.generate_general_advice(query)
        
        # Add source attribution
        sources = f"\n📚 **Sources:**\n"
        sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice + sources
    
    def set_user_city(self, city: str) -> str:
        """Set user's city for weather queries"""
        self.user_city = city
        return f"✅ City set to: {city}\n🌤️ Now I can provide weather-based agricultural advice for your area!"
    
    def get_user_city(self) -> str:
        """Get current user city"""
        return self.user_city or "Not set"
    
    def run_interactive(self):
        """Run interactive agricultural advisor bot"""
        print("=" * 80)
        print("🌾 AGRICULTURAL ADVISOR BOT 🌾")
        print("=" * 80)
        print("🤖 I'm your AI agricultural advisor! I can help you with:")
        print("   • Weather-based farming advice")
        print("   • Government policy information")
        print("   • Market prices and trends")
        print("   • Technical support and equipment")
        print("   • General agricultural guidance")
        print("   • Crop and soil management tips")
        print("\n💡 Commands: 'city', 'stats', 'help', 'quit'")
        print("\n🌤️ For weather advice: First set your city with 'city [cityname]', then ask weather questions!")
        print("-" * 80)
        
        while True:
            try:
                user_input = input(f"\n🌾 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using Agricultural Advisor Bot!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                
                elif user_input.lower().startswith('city'):
                    self._handle_city_command(user_input)
                
                # Handle direct city input (when user just types city name)
                elif not self.user_city and len(user_input.split()) <= 2 and not any(word in user_input.lower() for word in ['weather', 'rain', 'temperature', 'forecast', 'hot', 'cold', 'sunny', 'cloudy', 'wind', 'humidity']):
                    # If no city is set and user inputs what looks like a city name (not weather-related words)
                    response = self.set_user_city(user_input)
                    print(f"🤖 Advisor: {response}")
                    continue
                
                elif user_input:
                    print("🔄 Processing your query...")
                    start_time = time.time()
                    
                    response = self.process_query(user_input)
                    
                    end_time = time.time()
                    print(f"\n🤖 Advisor: {response}")
                    print(f"⚡ Response time: {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\n👋 Thank you for using Agricultural Advisor Bot!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 **Agricultural Advisor Bot Help**")
        print("=" * 50)
        print("🌤️ **Weather Queries:**")
        print("  Step 1: Set your city - 'city Mumbai'")
        print("  Step 2: Ask weather questions:")
        print("    - 'How does this weather affect my wheat crop?'")
        print("    - 'Should I irrigate today?'")
        print("    - 'Is this good weather for planting?'")
        print("    - 'How to protect crops from heat wave?'")
        print("    - 'Will it rain tomorrow?'")
        
        print("\n📋 **Policy Queries:**")
        print("  - 'What is PM Kisan scheme?'")
        print("  - 'How to apply for crop insurance?'")
        print("  - 'What are the benefits of soil health card?'")
        print("  - 'Who is eligible for PMKSY?'")
        
        print("\n🌾 **General Agriculture:**")
        print("  - 'How to improve soil fertility?'")
        print("  - 'Best time to plant tomatoes?'")
        print("  - 'How to control pest infestation?'")
        print("  - 'Organic farming methods?'")
        
        print("\n🔧 **Commands:**")
        print("  - 'city [cityname]': Set your city for weather advice")
        print("  - 'stats': Show system statistics")
        print("  - 'help': Show this help")
        print("  - 'quit': Exit the bot")
    
    def _show_stats(self):
        """Show system statistics"""
        print("\n📊 **System Statistics**")
        print("=" * 30)
        print(f"🏙️ User City: {self.get_user_city()}")
        
        if self.policy_chatbot.is_loaded:
            stats = self.policy_chatbot.get_statistics()
            print(f"📚 Policy Database: ✅ Loaded")
            print(f"   {stats}")
        else:
            print("📚 Policy Database: ❌ Not loaded")
            print("   💡 Policy queries will use general AI advice")
        
        print(f"🌤️ Weather Service: {'Available' if self.weather_service.api_key or self.weather_service.primary_api == 'openmeteo' else 'API key needed'}")
        print(f"🤖 AI Advisor: {'Available' if self.groq_advisor.api_key else 'API key needed'}")
        print(f"📊 Agricultural Data: ✅ Available (Prices & Soil)")
    
    def _handle_city_command(self, command: str):
        """Handle city setting command"""
        parts = command.split(' ', 1)
        if len(parts) > 1:
            city = parts[1].strip()
            response = self.set_user_city(city)
            print(f"🤖 Advisor: {response}")
        else:
            print(f"🤖 Advisor: Current city: {self.get_user_city()}")
            print("💡 To set city: 'city [cityname]' (e.g., 'city Mumbai')")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agricultural Advisor Bot")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--city", type=str, help="Set default city")
    
    args = parser.parse_args()
    
    bot = AgriculturalAdvisorBot()
    
    if args.city:
        print(bot.set_user_city(args.city))
    
    if args.query:
        print(f"🌾 Query: {args.query}")
        response = bot.process_query(args.query)
        print(f"🤖 Advisor: {response}")
    
    elif args.interactive:
        bot.run_interactive()
    
    else:
        bot.run_interactive()

if __name__ == "__main__":
    main()
