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
            'frost', 'hail', 'snow', 'sunny', 'cloudy', 'overcast', 'mausam', 'baarish',
            'मौसम', 'तापमान', 'बारिश', 'सूखा', 'बाढ़', 'आर्द्रता', 'हवा', 'जलवायु',
            'पूर्वानुमान', 'मानसून', 'गर्म', 'ठंडा', 'सूखा', 'गीला', 'तूफान'
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
            'mulya', 'keemat', 'rupees', 'rs', 'quintal', 'ton', 'kg', 'per', 'auction',
            'मूल्य', 'दर', 'कीमत', 'भाव', 'मंडी', 'बाजार', 'रुपये', 'क्विंटल', 'किलो'
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
            'yield', 'production', 'storage', 'transport', 'organic', 'traditional', 'modern',
            'फसल', 'खेती', 'कृषि', 'मिट्टी', 'खाद', 'कीटनाशक', 'सिंचाई', 'फसल कटाई',
            'बुवाई', 'बीज', 'कीट', 'रोग', 'उपज', 'उत्पादन', 'भंडारण', 'परिवहन'
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
    
    def _detect_language(self, text: str) -> str:
        """Detect if the text contains Hindi/Devanagari script"""
        # Check for Devanagari script (Hindi, Marathi, etc.)
        devanagari_pattern = r'[\u0900-\u097F]'
        if re.search(devanagari_pattern, text):
            return "Hindi"
        return "English"
    
    def _get_language_instructions(self, language: str) -> str:
        """Get language-specific instructions for AI prompts"""
        if language.lower() == "hindi":
            return """
1. Respond in Hindi (Devanagari script)
2. Use simple, understandable Hindi language
3. Keep responses concise and direct (max 4-5 sentences)
4. Do NOT use formal greetings like "प्रिय किसान" or "धन्यवाद"
5. Do NOT add signatures or formal closings
6. Focus on actionable advice only
7. Use agricultural terms in Hindi when possible
"""
        else:
            return """
1. Keep responses concise and direct (max 4-5 sentences)
2. Do NOT use formal greetings like "Dear Farmer" or "Sincerely"
3. Do NOT add signatures or formal closings
4. Focus on actionable advice only
"""
    
    def _get_language_system_message(self, language: str, context: str = "general") -> str:
        """Get language-specific system message for AI"""
        if language.lower() == "hindi":
            if context == "weather":
                return "आप एक सीधे और व्यावहारिक कृषि सलाहकार हैं। मौसम आधारित संक्षिप्त, कार्रवाई योग्य सलाह दें। औपचारिक भाषा या अभिवादन का उपयोग न करें। भारतीय किसानों के लिए व्यावहारिक समाधान पर ध्यान दें।"
            elif context == "price":
                return "आप एक सीधे और व्यावहारिक कृषि बाजार सलाहकार हैं। संक्षिप्त, कार्रवाई योग्य मूल्य सलाह दें। औपचारिक भाषा या अभिवादन का उपयोग न करें। भारतीय किसानों के लिए व्यावहारिक समाधान पर ध्यान दें।"
            else:
                return "आप एक सीधे और व्यावहारिक कृषि सलाहकार हैं। संक्षिप्त, कार्रवाई योग्य सलाह दें। औपचारिक भाषा या अभिवादन का उपयोग न करें। भारतीय किसानों के लिए व्यावहारिक समाधान पर ध्यान दें।"
        else:
            if context == "weather":
                return "You are a direct and practical agricultural advisor. Provide concise, actionable weather-based advice without formal language or greetings. Focus on practical solutions for Indian farmers."
            elif context == "price":
                return "You are a direct and practical agricultural market advisor. Provide concise, actionable price advice without formal language or greetings. Focus on practical solutions for Indian farmers."
            else:
                return "You are a direct and practical agricultural advisor. Provide concise, actionable advice without formal language or greetings. Focus on practical solutions for Indian farmers."
    
    def generate_weather_advice(self, query: str, current_weather, forecast=None, language: str = "English") -> str:
        """Generate agricultural advice based on weather data"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Detect language from query if not specified
            if language == "English":
                detected_lang = self._detect_language(query)
                if detected_lang == "Hindi":
                    language = "Hindi"
            
            language_instructions = self._get_language_instructions(language)
            system_message = self._get_language_system_message(language, "weather")
            
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
10. Focus on practical farming decisions{language_instructions}

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a direct and practical agricultural advisor. Provide concise, actionable weather-based advice without formal language or greetings. Focus on practical solutions for Indian farmers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 400
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating weather advice: {e}")
            return f"❌ Error generating advice: {e}"
    
    def generate_weather_advice_comprehensive(self, query: str, location_info, current_weather, forecast_data, agricultural_insights, language: str = "English") -> str:
        """Generate comprehensive agricultural advice based on weather data"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Detect language from query if not specified
            if language == "English":
                detected_lang = self._detect_language(query)
                if detected_lang == "Hindi":
                    language = "Hindi"
            
            language_instructions = self._get_language_instructions(language)
            system_message = self._get_language_system_message(language, "weather")
            
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
12. Focus on practical farming decisions{language_instructions}

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a direct and practical agricultural advisor. Provide concise, actionable weather-based advice without formal language or greetings. Focus on practical solutions for Indian farmers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 400
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating comprehensive weather advice: {e}")
            return f"❌ Error generating advice: {e}"
        
    def generate_sql(self, user_query: str):
        """Translate natural language agricultural price queries into SQL"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            prompt = f"""
You are an assistant that translates natural language agricultural price questions into SQL queries.
The database is SQLite with a table mandi_prices.

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

Rules:
1. For comparative queries (e.g., "compare wheat prices in Kanpur vs Lucknow"):
   - Do NOT use self-joins.
   - Use WHERE District IN ('X','Y') and GROUP BY District.
   - Return aggregated stats: AVG(Modal_Price), MAX(Modal_Price), MIN(Modal_Price).
   - LIMIT 20 rows max.

2. For *trend/time-series queries* (e.g., "price trend of rice in Lucknow over last month"):
   - Use ORDER BY Arrival_Date.
   - Return AVG(Modal_Price) grouped by Arrival_Date (daily or weekly).
   - Limit to the last 30 records.

3. For *best mandi / where to sell* queries (e.g., "which is the best mandi to sell wheat in Kanpur"):
   - Use MAX(Modal_Price).
   - ORDER BY Modal_Price DESC.
   - LIMIT 1–5 markets.

4. For *latest price queries* (e.g., "what is the price of wheat in Kanpur today"):
   - ORDER BY Arrival_Date DESC.
   - LIMIT 1.

5. Always restrict output to at most 20 rows.
6. Prefer aggregate queries (AVG, MAX, MIN) over raw rows.
7. Return only SQL code, no explanation.


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

    
    def generate_general_advice(self, query: str, language: str = "English") -> str:
        """Generate general agricultural advice"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Detect language from query if not specified
            if language == "English":
                detected_lang = self._detect_language(query)
                if detected_lang == "Hindi":
                    language = "Hindi"
            
            language_instructions = self._get_language_instructions(language)
            system_message = self._get_language_system_message(language, "general")
            
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
9. If about pests/diseases, mention prevention and treatment{language_instructions}

Agricultural Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 300
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating general advice: {e}")
            return f"❌ Error generating advice: {e}"
    
    def generate_price_advice(self, query: str, price_data: str, language: str = "English") -> str:
        """Generate concise price-related advice"""
        if not self.api_key:
            return "❌ Groq API key not found."
        
        try:
            # Detect language from query if not specified
            if language == "English":
                detected_lang = self._detect_language(query)
                if detected_lang == "Hindi":
                    language = "Hindi"
            
            language_instructions = self._get_language_instructions(language)
            system_message = self._get_language_system_message(language, "price")
            
            prompt = f"""You are an agricultural market advisor. Provide brief, actionable price advice.

Query: {query}
Price Data: {price_data}

Instructions:
1. Give concise market insights (2-3 sentences max)
2. Focus on actionable advice for farmers
3. Use simple, direct language
4. Provide specific recommendations if possible{language_instructions}

Market Advice:"""

            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating price advice: {e}")
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
        
        # User preferences
        self.user_city = None
        self.user_crop = None
        self.user_language = "English"  # Default language
        self.is_initialized = False  # Track if user has completed initial setup

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
        # Check if user is initialized (except for help and setup commands)
        if not self.is_user_initialized() and not any(cmd in query.lower() for cmd in ['help', 'city', 'crop', 'language', 'stats']):
            return "🎯 **Setup Required**\n\n📍 Please complete the initial setup first. Use 'help' to see available commands."
        
        # Classify the query
        try:
            query_type = self.classifier.classify_query(query)
            print(f"🔍 **DEBUG: Query '{query}' classified as: {query_type}**")
        except Exception as e:
            print(f"❌ **DEBUG: Classification failed: {e}**")
            # Fallback classification
            query_lower = query.lower()
            if any(word in query_lower for word in ['price', 'rate', 'cost', 'mandi', 'market', 'bhav']):
                query_type = "price"
            elif any(word in query_lower for word in ['weather', 'temperature', 'rain', 'forecast']):
                query_type = "weather"
            elif any(word in query_lower for word in ['policy', 'scheme', 'government']):
                query_type = "policy"
            else:
                query_type = "general"
            print(f"🔍 **DEBUG: Fallback classification: {query_type}**")
        
        if query_type == "weather":
            return self._handle_weather_query(query)
        elif query_type == "policy":
            return self._handle_policy_query(query)
        elif query_type == "price":
            print(f"💰 **PRICE QUERY DETECTED: {query}**")
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
        
        if not self.is_user_initialized():
            return intent_info + "📍 Please complete the initial setup first. Use 'help' to see available commands."
        
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
                query, location_info, current_weather, forecast_data, agricultural_insights, self.user_language
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
            if forecast_data and len(forecast_data) > 0:
                response += f"📅 **7-Day Forecast Summary:**\n"
                try:
                    # Safe division to prevent division by zero
                    temp_values = [f.get('temperature', {}).get('avg', 0) for f in forecast_data]
                    precip_values = [f.get('precipitation', {}).get('amount', 0) for f in forecast_data]
                    
                    avg_temp = sum(temp_values) / len(temp_values) if temp_values else 0
                    total_precip = sum(precip_values) if precip_values else 0
                    
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
            # sources = f"\n📚 **Sources:**\n"
            # sources += f"• Weather Data: Open-Meteo API (Real-time)\n"
            # sources += f"• Location Data: Geocoding API\n"
            # sources += f"• AI Analysis: Groq API (Llama3-8b-8192 model)\n"
            # sources += f"• Agricultural Insights: Weather-based calculations\n"
            
            return intent_info + response
            
        except Exception as e:
            logger.error(f"Error handling weather query: {e}")
            if "division by zero" in str(e).lower():
                return intent_info + "❌ Weather data is currently unavailable. Please try again later or check your internet connection."
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                return intent_info + "❌ Weather service is temporarily unavailable. Please try again in a few minutes."
            else:
                return intent_info + f"❌ Error processing weather query: {e}"
    
    def _handle_policy_query(self, query: str) -> str:
        """Handle policy-related queries"""
        intent_info = f"🎯 **Detected Intent: Policy Query**\n\n"
        
        if not self.policy_chatbot.is_loaded:
            # Fallback to general advice when policy database is not available
            logger.warning("Policy database not loaded, falling back to general advice")
            return intent_info + self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India.", self.user_language
            )
        
        try:
            # Use Groq for better policy responses
            policy_response = self.policy_chatbot.ask_question_with_groq(query)
            
            # Add source attribution
            # sources = f"\n📚 **Sources:**\n"
            # sources += f"• Policy Documents: `pdfs/` directory (12 PDF files)\n"
            # sources += f"• Vector Database: `improved_vector_db/` (973 sections)\n"
            # sources += f"• AI Processing: Groq API (Llama3-8b-8192 model)\n"
            # sources += f"• Documents: PM Kisan, PMKSY, Soil Health Card, Crop Insurance, etc.\n"
            
            return intent_info + policy_response
        except Exception as e:
            logger.error(f"Error handling policy query: {e}")
            # Fallback to general advice
            return intent_info + self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India.", self.user_language
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
        ai_advice = self.groq_advisor.generate_general_advice(query, self.user_language)
        
        # Add source attribution
        # sources = f"\n📚 **Sources:**\n"
        # sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        # sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice 
    
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
                return f"❌ {soil_result['error']}\n\n🤖 **General Soil Advice:**\n{self.groq_advisor.generate_general_advice(query, self.user_language)}"
            
            # Combine soil data with AI advice
            ai_advice = self.groq_advisor.generate_general_advice(
                f"Soil health question: {query}. Based on soil data: pH {soil_result['ph']}, Organic Carbon {soil_result['organic_carbon']}%, N {soil_result['nitrogen']} kg/ha, P {soil_result['phosphorus']} kg/ha, K {soil_result['potassium']} kg/ha. Please provide soil management advice.", self.user_language
            )
            
            # Add source attribution
            # sources = f"\n📚 **Sources:**\n"
            # sources += f"• Soil Data: `soil_health.csv` (5 districts)\n"
            # sources += f"• Database: `agri_data.db` (SQLite)\n"
            # sources += f"• AI Advice: Groq API (Llama3-8b-8192 model)\n"
            
            return f"🌱 **Soil Health Data for {location}:**\n{soil_result['formatted']}\n\n🤖 **Soil Management Advice:**\n{ai_advice}"
            
        except Exception as e:
            logger.error(f"Error handling soil health query: {e}")
            return f"❌ Error retrieving soil data: {str(e)}\n\n🤖 **General Soil Advice:**\n{self.groq_advisor.generate_general_advice(query, self.user_language)}"
    
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
                f"Crop recommendation question: {query}. {soil_info}{crop_info}Please provide crop recommendations and farming advice.", self.user_language
            )
            
            response = f"🌾 **Crop Information for {location}:**\n"
            if available_crops:
                response += f"• Available crops: {', '.join(available_crops[:5])}\n"
            if "error" not in soil_result:
                response += f"• Soil conditions: pH {soil_result['ph']}, Organic Carbon {soil_result['organic_carbon']}%\n"
            response += f"\n🤖 **Crop Recommendations:**\n{ai_advice}"
            
            # Add source attribution
            # sources = f"\n📚 **Sources:**\n"
            # sources += f"• Crop Data: `mandi_prices.csv` (35,522 records)\n"
            # sources += f"• Soil Data: `soil_health.csv` (5 districts)\n"
            # sources += f"• Database: `agri_data.db` (SQLite)\n"
            # sources += f"• AI Recommendations: Groq API (Llama3-8b-8192 model)\n"
            
            return response 
            
        except Exception as e:
            logger.error(f"Error handling crop recommendation query: {e}")
            return f"❌ Error retrieving crop data: {str(e)}\n\n🤖 **General Crop Advice:**\n{self.groq_advisor.generate_general_advice(query, self.user_language)}"
    
    def _handle_price_query(self, query: str):
        """Handle mandi price queries using LLM-based SQL generation."""
        print("in _handle_price_query")
        try:
            # Step 1: Ask Groq to generate SQL
            sql_query = self.groq_advisor.generate_sql(query).strip()

            logger.info(f"Groq SQL: {sql_query}")
            print("query:",sql_query)

            # Step 2: Try executing the SQL
            conn = sqlite3.connect("agri_data.db")
            try:
                results = conn.execute(sql_query).fetchall()
            except Exception as e:
                logger.warning(f"Groq SQL failed: {e}")
                results = []

            conn.close()

            # Step 3: If Groq SQL worked
            if results:

                if len(results) > 50:
                    results = results[:50]
                    results.append(("...", "...", "..."))  # marker for truncation

                    results = "\n".join([str(r) for r in results])

                return self.groq_advisor.generate_general_advice(
                    f"{query}\nData retrieved from the database:\n{results}\n"
                    f"Please summarize this for farmers in simple words using the data retrieved.", self.user_language
                )

            # Step 4: Fallback → general safe query
            else:
                city = self.user_city
                crop = self.user_crop
                if not city or not crop:
                    return "Please specify both city and crop for accurate price information."

                fallback_results = self.groq_advisor.get_latest_prices_all_markets(city, crop)
                if not fallback_results:
                    return f"No data found for {crop} in {city}."

                formatted = "\n".join([
                    f"{market}: ₹{price}/quintal ({date})"
                    for market, date, price in fallback_results
                ])
                return self.groq_advisor.generate_general_advice(
                    f"{query}. "
                    f"Here's price data for {crop} in {city}:\n{formatted}\n"
                    f"Summarize in farmer-friendly advice.", self.user_language
                )

        except Exception as e:
            logger.error(f"Error handling price query: {e}")
            return "I could not fetch the price data right now. Please try again later."
    
    def _extract_crop_and_location(self, query: str) -> tuple:
        """Extract crop and location from query"""
        query_lower = query.lower()
        
        # Common crops in English and Hindi - order matters for better matching
        crops_english = ["wheat", "rice", "maize", "cotton", "potato", "tomato", "sugarcane", "pulses", "oilseeds"]
        crops_hindi = ["गेहूं", "चावल", "मक्का", "कपास", "आलू", "टमाटर", "गन्ना", "दालें", "तिलहन"]
        
        # Combine English and Hindi crops
        all_crops = crops_english + crops_hindi
        crop = None
        
        # More specific matching to avoid false positives
        for i, c in enumerate(all_crops):
            # Check for exact word boundaries to avoid partial matches
            if f" {c} " in f" {query_lower} " or query_lower.startswith(c) or query_lower.endswith(c):
                # Map Hindi crops back to English for database lookup
                if i >= len(crops_english):
                    crop = crops_english[i - len(crops_english)].title()
                else:
                    crop = c.title()
                break
        
        # Common locations (cities in our database)
        locations = ["agra", "kannauj", "kanpur", "lucknow", "unnao", "mumbai", "delhi", "bangalore"]
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
            f"Technical question: {query}. Please provide information about agricultural equipment, technology, maintenance, and technical solutions for farming.", self.user_language
        )
        
        # Add source attribution
        # sources = f"\n📚 **Sources:**\n"
        # sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        # sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice 
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        intent_info = f"🎯 **Detected Intent: General Inquiry**\n\n"
        ai_advice = self.groq_advisor.generate_general_advice(query, self.user_language)
        
        # Add source attribution
        # sources = f"\n📚 **Sources:**\n"
        # sources += f"• AI Knowledge: Groq API (Llama3-8b-8192 model)\n"
        # sources += f"• Agricultural Expertise: Pre-trained model knowledge\n"
        
        return intent_info + ai_advice
    
    def set_user_city(self, city: str) -> str:
        """Set user's city for weather queries"""
        self.user_city = city
        self.is_initialized = True  # Mark as initialized when city is set
        return f"✅ City set to: {city}\n🌤️ Now I can provide weather-based agricultural advice for your area!"
    
    def get_user_city(self) -> str:
        """Get current user city"""
        return self.user_city or "Not set"
    
    def set_user_crop(self, crop: str) -> str:
        """Set user's primary crop"""
        self.user_crop = crop
        return f"✅ Primary crop set to: {crop}\n🌾 I'll provide crop-specific advice for {crop}!"
    
    def get_user_crop(self) -> str:
        """Get current user crop"""
        return self.user_crop or "Not set"
    
    def set_user_language(self, language: str) -> str:
        """Set user's preferred language"""
        self.user_language = language
        return f"✅ Language set to: {language}\n🌍 I'll communicate in {language}!"
    
    def get_user_language(self) -> str:
        """Get current user language"""
        return self.user_language or "English"
    
    def is_user_initialized(self) -> bool:
        """Check if user has completed initial setup"""
        return self.is_initialized and self.user_city is not None
    
    def run_interactive(self):
        """Run interactive agricultural advisor bot"""
        print("=" * 80)
        print("🌾 AGRICULTURAL ADVISOR BOT 🌾")
        print("=" * 80)
        
        # Check if user needs to complete initial setup
        if not self.is_user_initialized():
            self._run_initial_setup()
        
        print("🤖 I'm your AI agricultural advisor! I can help you with:")
        print("   • Weather-based farming advice")
        print("   • Government policy information")
        print("   • Market prices and trends")
        print("   • Technical support and equipment")
        print("   • General agricultural guidance")
        print("   • Crop and soil management tips")
        print("\n💡 Commands: 'city', 'crop', 'language', 'stats', 'help', 'quit'")
        print(f"\n📍 Your location: {self.get_user_city()}")
        print(f"🌾 Your crop: {self.get_user_crop()}")
        print(f"🌍 Your language: {self.get_user_language()}")
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
                
                elif user_input.lower().startswith('crop'):
                    self._handle_crop_command(user_input)
                
                elif user_input.lower().startswith('language'):
                    self._handle_language_command(user_input)
                
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
        print("  - 'How does this weather affect my wheat crop?'")
        print("  - 'Should I irrigate today?'")
        print("  - 'Is this good weather for planting?'")
        print("  - 'How to protect crops from heat wave?'")
        print("  - 'Will it rain tomorrow?'")
        
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
        print("  - 'crop [cropname]': Set your primary crop")
        print("  - 'language [language]': Set your preferred language")
        print("  - 'stats': Show system statistics")
        print("  - 'help': Show this help")
        print("  - 'quit': Exit the bot")
        
        print(f"\n📍 **Your Current Settings:**")
        print(f"  - Location: {self.get_user_city()}")
        print(f"  - Primary Crop: {self.get_user_crop()}")
        print(f"  - Language: {self.get_user_language()}")
    
    def _show_stats(self):
        """Show system statistics"""
        print("\n📊 **System Statistics**")
        print("=" * 30)
        print(f"🏙️ User City: {self.get_user_city()}")
        print(f"🌾 User Crop: {self.get_user_crop()}")
        print(f"🌍 User Language: {self.get_user_language()}")
        
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
    
    def _run_initial_setup(self):
        """Run initial setup to get user preferences"""
        print("\n🎯 **WELCOME TO AGRICULTURAL ADVISOR BOT** 🎯")
        print("=" * 60)
        print("Before we start, I need to know your location to provide")
        print("personalized agricultural advice!")
        print("=" * 60)
        
        # Get location (mandatory)
        while not self.user_city:
            try:
                city_input = input("\n📍 Please enter your city/location: ").strip()
                if city_input:
                    response = self.set_user_city(city_input)
                    print(f"🤖 Advisor: {response}")
                else:
                    print("❌ City is required. Please enter a valid city name.")
            except KeyboardInterrupt:
                print("\n👋 Setup cancelled. Goodbye!")
                exit(0)
        
        # Get crop (optional)
        try:
            crop_input = input("\n🌾 Enter your primary crop (optional, press Enter to skip): ").strip()
            if crop_input:
                response = self.set_user_crop(crop_input)
                print(f"🤖 Advisor: {response}")
            else:
                print("🤖 Advisor: No crop set. You can set it later with 'crop [cropname]'")
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled. Goodbye!")
            exit(0)
        
        # Get language (optional)
        try:
            print("\n🌍 Available languages:")
            print("   • English (default)")
            print("   • Hindi")
            print("   • Marathi")
            print("   • Gujarati")
            print("   • Bengali")
            print("   • Tamil")
            print("   • Telugu")
            print("   • Kannada")
            print("   • Malayalam")
            print("   • Punjabi")
            
            lang_input = input("\n🌍 Choose your preferred language (optional, press Enter for English): ").strip()
            if lang_input:
                response = self.set_user_language(lang_input)
                print(f"🤖 Advisor: {response}")
            else:
                print("🤖 Advisor: Language set to English (default)")
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled. Goodbye!")
            exit(0)
        
        print("\n✅ Setup complete! You're ready to start!")
        print("💡 You can change these settings anytime using:")
        print("   • 'city [cityname]' - Change location")
        print("   • 'crop [cropname]' - Change primary crop")
        print("   • 'language [language]' - Change language")
        print("-" * 60)
    
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
    
    def _handle_crop_command(self, command: str):
        """Handle crop setting command"""
        parts = command.split(' ', 1)
        if len(parts) > 1:
            crop = parts[1].strip()
            response = self.set_user_crop(crop)
            print(f"🤖 Advisor: {response}")
        else:
            print(f"🤖 Advisor: Current crop: {self.get_user_crop()}")
            print("💡 To set crop: 'crop [cropname]' (e.g., 'crop wheat')")
    
    def _handle_language_command(self, command: str):
        """Handle language setting command"""
        parts = command.split(' ', 1)
        if len(parts) > 1:
            language = parts[1].strip()
            response = self.set_user_language(language)
            print(f"🤖 Advisor: {response}")
        else:
            print(f"🤖 Advisor: Current language: {self.get_user_language()}")
            print("💡 To set language: 'language [language]' (e.g., 'language Hindi')")
            print("🌍 Available languages: English, Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi")



def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agricultural Advisor Bot")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--city", type=str, help="Set default city")
    parser.add_argument("--crop", type=str, help="Set default crop")
    parser.add_argument("--language", type=str, help="Set default language")
    
    args = parser.parse_args()
    
    bot = AgriculturalAdvisorBot()
    
    # Set default values if provided
    if args.city:
        print(bot.set_user_city(args.city))
    
    if args.crop:
        print(bot.set_user_crop(args.crop))
    
    if args.language:
        print(bot.set_user_language(args.language))
    
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
