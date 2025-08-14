#!/usr/bin/env python3
"""
Weather Service Demo
Simple demonstration of the weather service functionality
"""

from weather_service import WeatherService
import json
from datetime import datetime

def demo_weather_service():
    """Demonstrate weather service functionality"""
    print("=" * 60)
    print("           WEATHER SERVICE DEMO")
    print("=" * 60)
    
    # Initialize weather service
    weather_service = WeatherService()
    
    # Test locations
    test_locations = [
        "Mumbai, Maharashtra, India",
        "Delhi, India", 
        "Bangalore, Karnataka, India",
        "Pune, Maharashtra, India"
    ]
    
    for i, location in enumerate(test_locations, 1):
        print(f"\n{'='*20} TEST {i}: {location} {'='*20}")
        
        try:
            # Get comprehensive weather report
            print(f"Fetching weather data for: {location}")
            report = weather_service.get_comprehensive_weather_report(location)
            
            if 'error' in report:
                print(f"❌ Error: {report['error']}")
                continue
            
            # Display key information
            location_info = report['location']
            print(f"✅ Location found: {location_info['city']}, {location_info['state']}, {location_info['country']}")
            print(f"   Coordinates: {location_info['coordinates']['latitude']:.4f}, {location_info['coordinates']['longitude']:.4f}")
            print(f"   Timezone: {location_info['timezone']}")
            
            # Historical data summary
            historical = report['historical_data']
            if historical:
                avg_temp = sum([h['temperature']['avg'] for h in historical]) / len(historical)
                total_precip = sum([h['precipitation'] for h in historical])
                print(f"📊 Historical (20 days): Avg temp: {avg_temp:.1f}°C, Total precip: {total_precip:.1f}mm")
            
            # Forecast summary
            forecast = report['forecast_data']
            if forecast:
                avg_temp_forecast = sum([f['temperature']['avg'] for f in forecast]) / len(forecast)
                total_precip_forecast = sum([f['precipitation']['amount'] for f in forecast])
                print(f"🔮 Forecast (7 days): Avg temp: {avg_temp_forecast:.1f}°C, Total precip: {total_precip_forecast:.1f}mm")
            
            # Agricultural insights
            insights = report['agricultural_insights']
            print(f"🌾 Agricultural Insights:")
            print(f"   Soil Moisture: {insights['soil_moisture']['status']}")
            print(f"   Crop Health: {insights['crop_health']['temperature_stress']} stress")
            print(f"   Irrigation: {insights['irrigation_needs']['irrigation_needed']}")
            print(f"   Pest Risk: {insights['pest_risk']['risk_level']}")
            print(f"   Harvest Timing: {insights['harvest_timing']['timing']}")
            
            # Show top 3 recommendations
            recommendations = insights['recommendations'][:3]
            print(f"💡 Top Recommendations:")
            for j, rec in enumerate(recommendations, 1):
                print(f"   {j}. {rec}")
            
        except Exception as e:
            print(f"❌ Error processing {location}: {e}")
        
        print(f"\n{'='*60}")
    
    print("\n✅ Weather Service Demo Completed!")
    print("Use 'python weather_cli.py --interactive' for interactive mode")
    print("Use 'python weather_cli.py --location \"Your Location\"' for specific location")

def demo_specific_features():
    """Demonstrate specific weather service features"""
    print("\n" + "=" * 60)
    print("           FEATURE DEMO")
    print("=" * 60)
    
    weather_service = WeatherService()
    location = "Mumbai, Maharashtra, India"
    
    print(f"\n📍 Testing individual features for: {location}")
    
    # Test location coordinates
    print("\n1. Testing Location Coordinates:")
    location_info = weather_service.get_location_coordinates(location)
    if location_info:
        print(f"   ✅ Found: {location_info.city}, {location_info.state}, {location_info.country}")
        print(f"   Coordinates: {location_info.latitude:.4f}, {location_info.longitude:.4f}")
    else:
        print("   ❌ Location not found")
    
    # Test historical weather
    print("\n2. Testing Historical Weather (5 days):")
    historical = weather_service.get_historical_weather(location, days=5)
    if historical:
        print(f"   ✅ Retrieved {len(historical)} days of historical data")
        for day in historical[-3:]:  # Show last 3 days
            print(f"   {day.date}: {day.condition} - {day.temperature_avg:.1f}°C, {day.precipitation:.1f}mm")
    else:
        print("   ❌ No historical data available")
    
    # Test weather forecast
    print("\n3. Testing Weather Forecast (3 days):")
    forecast = weather_service.get_weather_forecast(location, days=3)
    if forecast:
        print(f"   ✅ Retrieved {len(forecast)} days of forecast data")
        for day in forecast:
            print(f"   {day.date}: {day.condition} - {day.temperature_avg:.1f}°C, {day.precipitation_probability:.0f}% chance")
    else:
        print("   ❌ No forecast data available")
    
    # Test agricultural insights
    print("\n4. Testing Agricultural Insights:")
    if historical and forecast:
        insights = weather_service.get_agricultural_insights(historical, forecast)
        print(f"   ✅ Generated agricultural insights")
        print(f"   Soil Moisture: {insights['soil_moisture']['status']}")
        print(f"   Irrigation Needed: {insights['irrigation_needs']['irrigation_needed']}")
        print(f"   Pest Risk: {insights['pest_risk']['risk_level']}")
    else:
        print("   ❌ Cannot generate insights without data")

def save_sample_report():
    """Save a sample weather report to JSON file"""
    print("\n" + "=" * 60)
    print("           SAVING SAMPLE REPORT")
    print("=" * 60)
    
    weather_service = WeatherService()
    location = "Mumbai, Maharashtra, India"
    
    print(f"\n📄 Saving sample report for: {location}")
    
    try:
        report = weather_service.get_comprehensive_weather_report(location)
        
        if 'error' not in report:
            filename = f"weather_report_{location.replace(', ', '_').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Sample report saved to: {filename}")
            print(f"   File contains: {len(report['historical_data'])} days historical data")
            print(f"   File contains: {len(report['forecast_data'])} days forecast data")
            print(f"   File contains: {len(report['agricultural_insights']['recommendations'])} recommendations")
        else:
            print(f"❌ Error: {report['error']}")
            
    except Exception as e:
        print(f"❌ Error saving report: {e}")

if __name__ == "__main__":
    # Run main demo
    demo_weather_service()
    
    # Run feature demo
    demo_specific_features()
    
    # Save sample report
    save_sample_report()
    
    print("\n🎉 All demos completed successfully!")
    print("\nNext steps:")
    print("1. Run: python weather_cli.py --interactive")
    print("2. Run: python weather_cli.py --location \"Your Location\"")
    print("3. Check the generated JSON file for detailed data")
