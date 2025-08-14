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
from dotenv import load_dotenv

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
            'weather': ['weather_inquiry', 'climate_question', 'temperature_question', 'rainfall_question'],
            'policy': ['policy_inquiry', 'scheme_question', 'subsidy_question', 'government_help'],
            'agriculture': ['crop_question', 'farming_advice', 'soil_question', 'pest_question', 'irrigation_question']
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
            'frost', 'hail', 'snow', 'sunny', 'cloudy', 'overcast'
        ]
        
        policy_keywords = [
            'policy', 'scheme', 'subsidy', 'loan', 'insurance', 'support',
            'government', 'pm kisan', 'pmksy', 'soil health', 'mandi',
            'procurement', 'msp', 'fertilizer', 'seed', 'equipment',
            'guidelines', 'procedure', 'application', 'eligibility',
            'benefit', 'assistance', 'fund', 'grant', 'certificate'
        ]
        
        agriculture_keywords = [
            'crop', 'farming', 'agriculture', 'soil', 'fertilizer', 'pesticide',
            'irrigation', 'harvest', 'planting', 'seeding', 'pest', 'disease',
            'yield', 'production', 'market', 'price', 'storage', 'transport',
            'organic', 'traditional', 'modern', 'technology', 'equipment'
        ]
        
        # Count keyword matches
        weather_score = sum(1 for keyword in weather_keywords if keyword in query_lower)
        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        agriculture_score = sum(1 for keyword in agriculture_keywords if keyword in query_lower)
        
        # Classification logic
        if weather_score > 0:
            return "weather"
        elif policy_score > 0:
            return "policy"
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
        
        self.user_city = None
    
    def process_query(self, query: str) -> str:
        """Process user query and provide appropriate response"""
        # Classify the query
        query_type = self.classifier.classify_query(query)
        
        if query_type == "weather":
            return self._handle_weather_query(query)
        elif query_type == "policy":
            return self._handle_policy_query(query)
        elif query_type == "agriculture":
            return self._handle_agriculture_query(query)
        else:
            return self._handle_general_query(query)
    
    def _handle_weather_query(self, query: str) -> str:
        """Handle weather-related queries"""
        if not self.user_city:
            return "🌤️ I can help you with weather-based agricultural advice! Please tell me your city name first using the 'city' command (e.g., 'city Mumbai').\n\n💡 Example: 'city Mumbai' then ask 'will it rain tomorrow?'"
        
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
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling weather query: {e}")
            return f"❌ Error processing weather query: {e}"
    
    def _handle_policy_query(self, query: str) -> str:
        """Handle policy-related queries"""
        if not self.policy_chatbot.is_loaded:
            # Fallback to general advice when policy database is not available
            logger.warning("Policy database not loaded, falling back to general advice")
            return self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India."
            )
        
        try:
            # Use Groq for better policy responses
            return self.policy_chatbot.ask_question_with_groq(query)
        except Exception as e:
            logger.error(f"Error handling policy query: {e}")
            # Fallback to general advice
            return self.groq_advisor.generate_general_advice(
                f"Government policy question: {query}. Please provide general information about government agricultural policies and schemes in India."
            )
    
    def _handle_agriculture_query(self, query: str) -> str:
        """Handle general agricultural queries"""
        return self.groq_advisor.generate_general_advice(query)
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        return self.groq_advisor.generate_general_advice(query)
    
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
