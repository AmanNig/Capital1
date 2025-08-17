"""
Weather Service for Agricultural NLP Pipeline
Fetches historical weather data (past 20 days) and provides 7-day forecast
"""

import requests
import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather data structure"""
    date: str
    temperature_max: float
    temperature_min: float
    temperature_avg: float
    humidity: float
    precipitation: float
    wind_speed: float
    wind_direction: str
    pressure: float
    visibility: float
    uv_index: float
    condition: str
    description: str

@dataclass
class WeatherForecast:
    """Weather forecast structure"""
    date: str
    temperature_max: float
    temperature_min: float
    temperature_avg: float
    humidity: float
    precipitation_probability: float
    precipitation_amount: float
    wind_speed: float
    wind_direction: str
    pressure: float
    uv_index: float
    condition: str
    description: str

@dataclass
class LocationInfo:
    """Location information"""
    latitude: float
    longitude: float
    city: str
    state: str
    country: str
    timezone: str

class WeatherService:
    """Comprehensive weather service with multiple API support"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather service
        
        Args:
            api_key: API key for weather services (optional)
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.geocoder = Nominatim(user_agent="agricultural_nlp_pipeline")
        
        # API endpoints (using free/open APIs)
        self.apis = {
            'openweathermap': {
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'requires_key': True
            },
            'weatherapi': {
                'base_url': 'http://api.weatherapi.com/v1',
                'requires_key': True
            },
            'openmeteo': {
                'base_url': 'https://api.open-meteo.com/v1',
                'requires_key': False
            }
        }
        
        # Fallback to Open-Meteo (free, no API key required)
        self.primary_api = 'openmeteo'
        
    def get_location_coordinates(self, location: str) -> Optional[LocationInfo]:
        """
        Get coordinates for a location
        
        Args:
            location: Location string (city, state, country)
            
        Returns:
            LocationInfo object with coordinates and details
        """
        try:
            # Try to geocode the location
            location_data = self.geocoder.geocode(location)
            
            if location_data:
                # Get timezone information
                timezone = self._get_timezone(location_data.latitude, location_data.longitude)
                
                return LocationInfo(
                    latitude=location_data.latitude,
                    longitude=location_data.longitude,
                    city=location_data.raw.get('address', {}).get('city', ''),
                    state=location_data.raw.get('address', {}).get('state', ''),
                    country=location_data.raw.get('address', {}).get('country', ''),
                    timezone=timezone
                )
            else:
                logger.warning(f"Could not find coordinates for location: {location}")
                return None
                
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            logger.error(f"Geocoding error for {location}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in geocoding: {e}")
            return None
    
    def _get_timezone(self, lat: float, lon: float) -> str:
        """Get timezone for coordinates"""
        try:
            response = requests.get(
                f"https://api.open-meteo.com/v1/forecast",
                params={
                    'latitude': lat,
                    'longitude': lon,
                    'timezone': 'auto'
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('timezone', 'UTC')
        except Exception as e:
            logger.warning(f"Could not get timezone: {e}")
        return 'UTC'
    
    def get_historical_weather(self, location: str, days: int = 20) -> List[WeatherData]:
        """
        Get historical weather data for the past N days
        
        Args:
            location: Location string
            days: Number of days to fetch (default: 20)
            
        Returns:
            List of WeatherData objects
        """
        location_info = self.get_location_coordinates(location)
        if not location_info:
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use Open-Meteo for historical data (free, no API key)
        url = f"{self.apis['openmeteo']['base_url']}/forecast"
        
        params = {
            'latitude': location_info.latitude,
            'longitude': location_info.longitude,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,winddirection_10m,pressure_msl,visibility,uv_index',
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant',
            'timezone': location_info.timezone,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        # Add retry logic for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_historical_data(data, location_info)
                else:
                    logger.error(f"Failed to fetch historical weather: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    return []
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return []
            except Exception as e:
                logger.error(f"Error fetching historical weather: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return []
        
        return []
    
    def _parse_historical_data(self, data: Dict, location_info: LocationInfo) -> List[WeatherData]:
        """Parse historical weather data from API response"""
        weather_list = []
        
        try:
            daily_data = data.get('daily', {})
            hourly_data = data.get('hourly', {})
            
            dates = daily_data.get('time', [])
            temp_max = daily_data.get('temperature_2m_max', [])
            temp_min = daily_data.get('temperature_2m_min', [])
            temp_mean = daily_data.get('temperature_2m_mean', [])
            precipitation = daily_data.get('precipitation_sum', [])
            wind_speed = daily_data.get('windspeed_10m_max', [])
            wind_direction = daily_data.get('winddirection_10m_dominant', [])
            
            # Get hourly data for additional details
            hourly_times = hourly_data.get('time', [])
            humidity = hourly_data.get('relative_humidity_2m', [])
            pressure = hourly_data.get('pressure_msl', [])
            visibility = hourly_data.get('visibility', [])
            uv_index = hourly_data.get('uv_index', [])
            
            for i, date in enumerate(dates):
                # Get average values for the day
                day_start = f"{date}T00:00"
                day_end = f"{date}T23:00"
                
                # Find hourly data for this day
                day_humidity = []
                day_pressure = []
                day_visibility = []
                day_uv = []
                
                for j, time_str in enumerate(hourly_times):
                    if day_start <= time_str <= day_end:
                        if j < len(humidity):
                            day_humidity.append(humidity[j])
                        if j < len(pressure):
                            day_pressure.append(pressure[j])
                        if j < len(visibility):
                            day_visibility.append(visibility[j])
                        if j < len(uv_index):
                            day_uv.append(uv_index[j])
                
                # Calculate averages
                avg_humidity = sum(day_humidity) / len(day_humidity) if day_humidity else 0
                avg_pressure = sum(day_pressure) / len(day_pressure) if day_pressure else 1013.25
                avg_visibility = sum(day_visibility) / len(day_visibility) if day_visibility else 10
                avg_uv = sum(day_uv) / len(day_uv) if day_uv else 0
                
                # Determine weather condition based on data
                condition = self._determine_weather_condition(
                    temp_mean[i] if i < len(temp_mean) else 0,
                    precipitation[i] if i < len(precipitation) else 0,
                    avg_humidity
                )
                
                weather_data = WeatherData(
                    date=date,
                    temperature_max=temp_max[i] if i < len(temp_max) else 0,
                    temperature_min=temp_min[i] if i < len(temp_min) else 0,
                    temperature_avg=temp_mean[i] if i < len(temp_mean) else 0,
                    humidity=avg_humidity,
                    precipitation=precipitation[i] if i < len(precipitation) else 0,
                    wind_speed=wind_speed[i] if i < len(wind_speed) else 0,
                    wind_direction=self._get_wind_direction(wind_direction[i] if i < len(wind_direction) else 0),
                    pressure=avg_pressure,
                    visibility=avg_visibility,
                    uv_index=avg_uv,
                    condition=condition,
                    description=self._get_weather_description(condition)
                )
                
                weather_list.append(weather_data)
                
        except Exception as e:
            logger.error(f"Error parsing historical data: {e}")
        
        return weather_list
    
    def get_weather_forecast(self, location: str, days: int = 7) -> List[WeatherForecast]:
        """
        Get weather forecast for the next N days
        
        Args:
            location: Location string
            days: Number of days to forecast (default: 7)
            
        Returns:
            List of WeatherForecast objects
        """
        location_info = self.get_location_coordinates(location)
        if not location_info:
            return []
        
        # Use Open-Meteo for forecast data
        url = f"{self.apis['openmeteo']['base_url']}/forecast"
        
        params = {
            'latitude': location_info.latitude,
            'longitude': location_info.longitude,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,windspeed_10m,winddirection_10m,pressure_msl,uv_index',
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_probability_max,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant',
            'timezone': location_info.timezone,
            'forecast_days': days
        }
        
        # Add retry logic for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_forecast_data(data, location_info)
                else:
                    logger.error(f"Failed to fetch weather forecast: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    return []
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on forecast attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return []
            except Exception as e:
                logger.error(f"Error fetching weather forecast: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return []
        
        return []
    
    def _parse_forecast_data(self, data: Dict, location_info: LocationInfo) -> List[WeatherForecast]:
        """Parse forecast weather data from API response"""
        forecast_list = []
        
        try:
            daily_data = data.get('daily', {})
            
            dates = daily_data.get('time', [])
            temp_max = daily_data.get('temperature_2m_max', [])
            temp_min = daily_data.get('temperature_2m_min', [])
            temp_mean = daily_data.get('temperature_2m_mean', [])
            precip_prob = daily_data.get('precipitation_probability_max', [])
            precipitation = daily_data.get('precipitation_sum', [])
            wind_speed = daily_data.get('windspeed_10m_max', [])
            wind_direction = daily_data.get('winddirection_10m_dominant', [])
            
            # Get hourly data for additional details
            hourly_data = data.get('hourly', {})
            hourly_times = hourly_data.get('time', [])
            humidity = hourly_data.get('relative_humidity_2m', [])
            pressure = hourly_data.get('pressure_msl', [])
            uv_index = hourly_data.get('uv_index', [])
            
            for i, date in enumerate(dates):
                # Get average values for the day
                day_start = f"{date}T00:00"
                day_end = f"{date}T23:00"
                
                # Find hourly data for this day
                day_humidity = []
                day_pressure = []
                day_uv = []
                
                for j, time_str in enumerate(hourly_times):
                    if day_start <= time_str <= day_end:
                        if j < len(humidity):
                            day_humidity.append(humidity[j])
                        if j < len(pressure):
                            day_pressure.append(pressure[j])
                        if j < len(uv_index):
                            day_uv.append(uv_index[j])
                
                # Calculate averages
                avg_humidity = sum(day_humidity) / len(day_humidity) if day_humidity else 0
                avg_pressure = sum(day_pressure) / len(day_pressure) if day_pressure else 1013.25
                avg_uv = sum(day_uv) / len(day_uv) if day_uv else 0
                
                # Determine weather condition
                condition = self._determine_weather_condition(
                    temp_mean[i] if i < len(temp_mean) else 0,
                    precipitation[i] if i < len(precipitation) else 0,
                    avg_humidity
                )
                
                forecast_data = WeatherForecast(
                    date=date,
                    temperature_max=temp_max[i] if i < len(temp_max) else 0,
                    temperature_min=temp_min[i] if i < len(temp_min) else 0,
                    temperature_avg=temp_mean[i] if i < len(temp_mean) else 0,
                    humidity=avg_humidity,
                    precipitation_probability=precip_prob[i] if i < len(precip_prob) else 0,
                    precipitation_amount=precipitation[i] if i < len(precipitation) else 0,
                    wind_speed=wind_speed[i] if i < len(wind_speed) else 0,
                    wind_direction=self._get_wind_direction(wind_direction[i] if i < len(wind_direction) else 0),
                    pressure=avg_pressure,
                    uv_index=avg_uv,
                    condition=condition,
                    description=self._get_weather_description(condition)
                )
                
                forecast_list.append(forecast_data)
                
        except Exception as e:
            logger.error(f"Error parsing forecast data: {e}")
        
        return forecast_list
    
    def _determine_weather_condition(self, temp: float, precipitation: float, humidity: float) -> str:
        """Determine weather condition based on temperature, precipitation, and humidity"""
        if precipitation > 10:
            return "Heavy Rain"
        elif precipitation > 5:
            return "Moderate Rain"
        elif precipitation > 0:
            return "Light Rain"
        elif humidity > 80:
            return "Humid"
        elif temp > 30:
            return "Hot"
        elif temp > 20:
            return "Warm"
        elif temp > 10:
            return "Mild"
        elif temp > 0:
            return "Cool"
        else:
            return "Cold"
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind direction degrees to cardinal directions"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def _get_weather_description(self, condition: str) -> str:
        """Get detailed weather description"""
        descriptions = {
            "Heavy Rain": "Heavy rainfall with potential flooding",
            "Moderate Rain": "Moderate rainfall, good for crops",
            "Light Rain": "Light rainfall, beneficial for agriculture",
            "Humid": "High humidity, may affect crop growth",
            "Hot": "High temperatures, may stress crops",
            "Warm": "Pleasant temperatures, ideal for growth",
            "Mild": "Moderate temperatures, good growing conditions",
            "Cool": "Cool temperatures, may slow growth",
            "Cold": "Cold temperatures, may damage crops"
        }
        return descriptions.get(condition, "Variable weather conditions")
    
    def get_agricultural_insights(self, historical_data: List[WeatherData], 
                                forecast_data: List[WeatherForecast]) -> Dict[str, Any]:
        """
        Generate agricultural insights from weather data
        
        Args:
            historical_data: Past weather data
            forecast_data: Future weather forecast
            
        Returns:
            Dictionary with agricultural insights
        """
        insights = {
            'soil_moisture': self._analyze_soil_moisture(historical_data, forecast_data),
            'crop_health': self._analyze_crop_health(historical_data, forecast_data),
            'irrigation_needs': self._analyze_irrigation_needs(historical_data, forecast_data),
            'pest_risk': self._analyze_pest_risk(historical_data, forecast_data),
            'harvest_timing': self._analyze_harvest_timing(forecast_data),
            'recommendations': self._generate_recommendations(historical_data, forecast_data)
        }
        
        return insights
    
    def _analyze_soil_moisture(self, historical: List[WeatherData], 
                             forecast: List[WeatherForecast]) -> Dict[str, Any]:
        """Analyze soil moisture conditions"""
        recent_historical = historical[-7:] if len(historical) >= 7 else historical
        recent_precip = sum([h.precipitation for h in recent_historical]) if recent_historical else 0  # Last 7 days
        forecast_precip = sum([f.precipitation_amount for f in forecast]) if forecast else 0
        
        total_precip = recent_precip + forecast_precip
        
        if total_precip > 50:
            moisture_status = "High"
            risk = "Waterlogging possible"
        elif total_precip > 25:
            moisture_status = "Adequate"
            risk = "Good for most crops"
        elif total_precip > 10:
            moisture_status = "Moderate"
            risk = "May need irrigation"
        else:
            moisture_status = "Low"
            risk = "Irrigation recommended"
        
        return {
            'status': moisture_status,
            'risk': risk,
            'recent_precipitation': recent_precip,
            'forecast_precipitation': forecast_precip,
            'total_precipitation': total_precip
        }
    
    def _analyze_crop_health(self, historical: List[WeatherData], 
                           forecast: List[WeatherForecast]) -> Dict[str, Any]:
        """Analyze crop health based on weather conditions"""
        avg_temp = sum([h.temperature_avg for h in historical]) / len(historical) if historical else 0
        avg_humidity = sum([h.humidity for h in historical]) / len(historical) if historical else 0
        
        # Temperature stress analysis
        if avg_temp > 35:
            temp_stress = "High"
            temp_impact = "Heat stress may reduce yields"
        elif avg_temp > 30:
            temp_stress = "Moderate"
            temp_impact = "Monitor for heat stress"
        elif avg_temp < 5:
            temp_stress = "High"
            temp_impact = "Cold stress may damage crops"
        else:
            temp_stress = "Low"
            temp_impact = "Optimal temperature conditions"
        
        # Humidity impact
        if avg_humidity > 85:
            humidity_impact = "High humidity may promote disease"
        elif avg_humidity < 40:
            humidity_impact = "Low humidity may stress crops"
        else:
            humidity_impact = "Optimal humidity conditions"
        
        return {
            'temperature_stress': temp_stress,
            'temperature_impact': temp_impact,
            'humidity_impact': humidity_impact,
            'average_temperature': avg_temp,
            'average_humidity': avg_humidity
        }
    
    def _analyze_irrigation_needs(self, historical: List[WeatherData], 
                                forecast: List[WeatherForecast]) -> Dict[str, Any]:
        """Analyze irrigation requirements"""
        recent_historical = historical[-3:] if len(historical) >= 3 else historical
        recent_precip = sum([h.precipitation for h in recent_historical]) if recent_historical else 0  # Last 3 days
        forecast_precip = sum([f.precipitation_amount for f in forecast[:3]]) if forecast else 0  # Next 3 days
        
        total_water = recent_precip + forecast_precip
        
        if total_water < 10:
            irrigation_needed = "Yes"
            frequency = "Daily"
            amount = "Moderate"
        elif total_water < 20:
            irrigation_needed = "Maybe"
            frequency = "Every 2-3 days"
            amount = "Light"
        else:
            irrigation_needed = "No"
            frequency = "Not required"
            amount = "None"
        
        return {
            'irrigation_needed': irrigation_needed,
            'frequency': frequency,
            'amount': amount,
            'recent_water': recent_precip,
            'forecast_water': forecast_precip,
            'total_water': total_water
        }
    
    def _analyze_pest_risk(self, historical: List[WeatherData], 
                          forecast: List[WeatherForecast]) -> Dict[str, Any]:
        """Analyze pest risk based on weather conditions"""
        avg_temp = sum([h.temperature_avg for h in historical]) / len(historical) if historical else 0
        avg_humidity = sum([h.humidity for h in historical]) / len(historical) if historical else 0
        
        # Pest risk assessment
        if avg_temp > 25 and avg_humidity > 70:
            pest_risk = "High"
            pests = "Fungal diseases, insects"
        elif avg_temp > 20 and avg_humidity > 60:
            pest_risk = "Moderate"
            pests = "Some fungal diseases possible"
        else:
            pest_risk = "Low"
            pests = "Minimal pest pressure"
        
        return {
            'risk_level': pest_risk,
            'potential_pests': pests,
            'average_temperature': avg_temp,
            'average_humidity': avg_humidity
        }
    
    def _analyze_harvest_timing(self, forecast: List[WeatherForecast]) -> Dict[str, Any]:
        """Analyze optimal harvest timing"""
        dry_days = sum(1 for f in forecast if f.precipitation_probability < 30)
        
        if dry_days >= 5:
            harvest_timing = "Good"
            recommendation = "Plan harvest for dry days"
        elif dry_days >= 3:
            harvest_timing = "Moderate"
            recommendation = "Monitor weather closely"
        else:
            harvest_timing = "Poor"
            recommendation = "Consider delaying harvest"
        
        return {
            'timing': harvest_timing,
            'recommendation': recommendation,
            'dry_days_forecast': dry_days,
            'total_forecast_days': len(forecast)
        }
    
    def _generate_recommendations(self, historical: List[WeatherData], 
                                forecast: List[WeatherForecast]) -> List[str]:
        """Generate agricultural recommendations"""
        recommendations = []
        
        # Analyze recent conditions
        recent_historical = historical[-7:] if len(historical) >= 7 else historical
        recent_temp = sum([h.temperature_avg for h in recent_historical]) / len(recent_historical) if recent_historical else 0
        recent_precip = sum([h.precipitation for h in recent_historical]) if recent_historical else 0
        
        # Temperature-based recommendations
        if recent_temp > 30:
            recommendations.append("Consider shade structures for sensitive crops")
            recommendations.append("Increase irrigation frequency due to high temperatures")
        elif recent_temp < 10:
            recommendations.append("Protect crops from cold stress with covers")
            recommendations.append("Delay planting of warm-season crops")
        
        # Precipitation-based recommendations
        if recent_precip < 10:
            recommendations.append("Implement irrigation system if not already in place")
            recommendations.append("Monitor soil moisture levels closely")
        elif recent_precip > 50:
            recommendations.append("Ensure proper drainage to prevent waterlogging")
            recommendations.append("Monitor for fungal diseases")
        
        # Forecast-based recommendations
        forecast_precip = sum([f.precipitation_amount for f in forecast])
        if forecast_precip > 30:
            recommendations.append("Prepare for wet conditions in the coming week")
            recommendations.append("Delay field operations if possible")
        
        # General recommendations
        recommendations.append("Monitor weather forecasts regularly")
        recommendations.append("Adjust farming practices based on weather conditions")
        
        return recommendations
    
    def get_comprehensive_weather_report(self, location: str) -> Dict[str, Any]:
        """
        Get comprehensive weather report with historical data, forecast, and insights
        
        Args:
            location: Location string
            
        Returns:
            Complete weather report dictionary
        """
        logger.info(f"Fetching comprehensive weather report for: {location}")
        
        # Get location information
        location_info = self.get_location_coordinates(location)
        if not location_info:
            return {'error': f'Could not find location: {location}'}
        
        # Get historical data (past 20 days)
        historical_data = self.get_historical_weather(location, days=20)
        
        # Get forecast data (next 7 days)
        forecast_data = self.get_weather_forecast(location, days=7)
        
        # Generate agricultural insights
        insights = self.get_agricultural_insights(historical_data, forecast_data)
        
        # Compile comprehensive report
        report = {
            'location': {
                'name': location,
                'coordinates': {
                    'latitude': location_info.latitude,
                    'longitude': location_info.longitude
                },
                'city': location_info.city,
                'state': location_info.state,
                'country': location_info.country,
                'timezone': location_info.timezone
            },
            'current_time': datetime.now().isoformat(),
            'historical_data': [
                {
                    'date': w.date,
                    'temperature': {
                        'max': w.temperature_max,
                        'min': w.temperature_min,
                        'avg': w.temperature_avg
                    },
                    'humidity': w.humidity,
                    'precipitation': w.precipitation,
                    'wind': {
                        'speed': w.wind_speed,
                        'direction': w.wind_direction
                    },
                    'pressure': w.pressure,
                    'visibility': w.visibility,
                    'uv_index': w.uv_index,
                    'condition': w.condition,
                    'description': w.description
                }
                for w in historical_data
            ],
            'forecast_data': [
                {
                    'date': f.date,
                    'temperature': {
                        'max': f.temperature_max,
                        'min': f.temperature_min,
                        'avg': f.temperature_avg
                    },
                    'humidity': f.humidity,
                    'precipitation': {
                        'probability': f.precipitation_probability,
                        'amount': f.precipitation_amount
                    },
                    'wind': {
                        'speed': f.wind_speed,
                        'direction': f.wind_direction
                    },
                    'pressure': f.pressure,
                    'uv_index': f.uv_index,
                    'condition': f.condition,
                    'description': f.description
                }
                for f in forecast_data
            ],
            'agricultural_insights': insights,
            'summary': {
                'historical_days': len(historical_data),
                'forecast_days': len(forecast_data),
                'data_source': 'Open-Meteo API',
                'last_updated': datetime.now().isoformat()
            }
        }
        
        return report

def main():
    """Demo function to test the weather service"""
    weather_service = WeatherService()
    
    # Test location
    test_location = "Mumbai, Maharashtra, India"
    
    print(f"Weather Service Demo for: {test_location}")
    print("=" * 50)
    
    # Get comprehensive report
    report = weather_service.get_comprehensive_weather_report(test_location)
    
    if 'error' in report:
        print(f"Error: {report['error']}")
        return
    
    # Display location info
    location = report['location']
    print(f"Location: {location['name']}")
    print(f"Coordinates: {location['coordinates']['latitude']:.4f}, {location['coordinates']['longitude']:.4f}")
    print(f"City: {location['city']}, State: {location['state']}, Country: {location['country']}")
    print(f"Timezone: {location['timezone']}")
    print()
    
    # Display historical summary
    historical = report['historical_data']
    if historical and len(historical) > 0:
        print("Historical Weather Summary (Past 20 Days):")
        print("-" * 40)
        avg_temp = sum([h['temperature']['avg'] for h in historical]) / len(historical) if historical else 0
        total_precip = sum([h['precipitation'] for h in historical]) if historical else 0
        print(f"Average Temperature: {avg_temp:.1f}°C")
        print(f"Total Precipitation: {total_precip:.1f} mm")
        print()
    
    # Display forecast summary
    forecast = report['forecast_data']
    if forecast:
        print("Weather Forecast (Next 7 Days):")
        print("-" * 35)
        for day in forecast[:3]:  # Show first 3 days
            print(f"{day['date']}: {day['condition']} - "
                  f"Temp: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C, "
                  f"Precip: {day['precipitation']['probability']:.0f}%")
        print()
    
    # Display agricultural insights
    insights = report['agricultural_insights']
    print("Agricultural Insights:")
    print("-" * 20)
    print(f"Soil Moisture: {insights['soil_moisture']['status']} - {insights['soil_moisture']['risk']}")
    print(f"Crop Health: {insights['crop_health']['temperature_stress']} temperature stress")
    print(f"Irrigation: {insights['irrigation_needs']['irrigation_needed']} - {insights['irrigation_needs']['frequency']}")
    print(f"Pest Risk: {insights['pest_risk']['risk_level']} - {insights['pest_risk']['potential_pests']}")
    print(f"Harvest Timing: {insights['harvest_timing']['timing']} - {insights['harvest_timing']['recommendation']}")
    print()
    
    # Display recommendations
    print("Recommendations:")
    print("-" * 15)
    for i, rec in enumerate(insights['recommendations'][:5], 1):  # Show first 5 recommendations
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
