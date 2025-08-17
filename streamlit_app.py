import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from agricultural_advisor_bot import AgriculturalAdvisorBot
    from weather_service import WeatherService
    from improved_policy_chatbot import ImprovedPolicyChatbot
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all required files are in the same directory")

# Page configuration
st.set_page_config(
    page_title="🌾 Agricultural Advisor Bot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .language-selector {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_city' not in st.session_state:
    st.session_state.user_city = ""
if 'user_crop' not in st.session_state:
    st.session_state.user_crop = ""
if 'user_language' not in st.session_state:
    st.session_state.user_language = "English"

def initialize_bot():
    """Initialize the agricultural advisor bot"""
    try:
        with st.spinner("Initializing Agricultural Advisor Bot..."):
            bot = AgriculturalAdvisorBot()
            st.session_state.bot = bot
            st.success("✅ Bot initialized successfully!")
            return bot
    except Exception as e:
        st.error(f"❌ Error initializing bot: {e}")
        return None

def get_database_stats():
    """Get statistics from the database"""
    try:
        conn = sqlite3.connect('agri_data.db')
        cursor = conn.cursor()
        
        # Get price records count
        cursor.execute('SELECT COUNT(*) FROM mandi_prices')
        price_count = cursor.fetchone()[0]
        
        # Get soil records count
        cursor.execute('SELECT COUNT(*) FROM soil_health')
        soil_count = cursor.fetchone()[0]
        
        # Get unique crops
        cursor.execute('SELECT COUNT(DISTINCT Commodity) FROM mandi_prices')
        unique_crops = cursor.fetchone()[0]
        
        # Get unique markets
        cursor.execute('SELECT COUNT(DISTINCT Market) FROM mandi_prices')
        unique_markets = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'price_records': price_count,
            'soil_records': soil_count,
            'unique_crops': unique_crops,
            'unique_markets': unique_markets
        }
    except Exception as e:
        st.error(f"Error accessing database: {e}")
        return None

def get_policy_stats():
    """Get policy database statistics"""
    try:
        if os.path.exists('improved_vector_db/metadata.json'):
            with open('improved_vector_db/metadata.json', 'r') as f:
                metadata = json.load(f)
            return {
                'total_sections': metadata.get('total_sections', 0),
                'total_documents': metadata.get('total_documents', 0)
            }
        return None
    except Exception as e:
        st.error(f"Error accessing policy database: {e}")
        return None

def create_price_chart(city, crop):
    """Create a price trend chart"""
    try:
        conn = sqlite3.connect('agri_data.db')
        query = """
        SELECT Arrival_Date, AVG(Modal_Price) as avg_price, COUNT(*) as count
        FROM mandi_prices 
        WHERE District LIKE ? AND Commodity LIKE ?
        GROUP BY Arrival_Date
        ORDER BY Arrival_Date DESC
        LIMIT 30
        """
        
        df = pd.read_sql_query(query, conn, params=(f"%{city}%", f"%{crop}%"))
        conn.close()
        
        if df.empty:
            return None
            
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
        df = df.sort_values('Arrival_Date')
        
        fig = px.line(df, x='Arrival_Date', y='avg_price', 
                     title=f'Price Trend for {crop} in {city}',
                     labels={'avg_price': 'Average Price (₹/quintal)', 'Arrival_Date': 'Date'})
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🌾 Agricultural Advisor Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Empowering Indian Farmers with AI-Powered Agricultural Intelligence</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")
        
        # Language selection
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        language = st.selectbox(
            "🌍 Select Language",
            ["English", "Hindi"],
            index=0 if st.session_state.user_language == "English" else 1
        )
        st.session_state.user_language = language
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User preferences
        st.markdown("### 👤 User Preferences")
        city = st.text_input("🏙️ Your City", value=st.session_state.user_city, placeholder="e.g., Kanpur, Mumbai")
        if city:
            st.session_state.user_city = city
            
        crop = st.text_input("🌾 Primary Crop", value=st.session_state.user_crop, placeholder="e.g., Wheat, Rice")
        if crop:
            st.session_state.user_crop = crop
        
        # Initialize bot button
        if st.button("🚀 Initialize Bot", type="primary"):
            bot = initialize_bot()
            if bot:
                st.session_state.bot = bot
        
        # System status
        st.markdown("### 📊 System Status")
        if st.session_state.bot:
            st.success("✅ Bot Active")
        else:
            st.error("❌ Bot Not Initialized")
        
        # Database stats
        db_stats = get_database_stats()
        if db_stats:
            st.metric("Price Records", f"{db_stats['price_records']:,}")
            st.metric("Soil Records", f"{db_stats['soil_records']:,}")
            st.metric("Unique Crops", db_stats['unique_crops'])
            st.metric("Markets", db_stats['unique_markets'])
        
        # Policy stats
        policy_stats = get_policy_stats()
        if policy_stats:
            st.metric("Policy Sections", policy_stats['total_sections'])
            st.metric("Documents", policy_stats['total_documents'])
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📊 Analytics", "🌤️ Weather", "📋 Policies", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">💬 Interactive Chat</h2>', unsafe_allow_html=True)
        
        if not st.session_state.bot:
            st.warning("Please initialize the bot first using the sidebar button.")
            return
        
        # Chat interface
        st.markdown("### Ask your agricultural questions:")
        
        # Query input
        if language == "Hindi":
            placeholder = "उदाहरण: गेहूं का भाव क्या है? मौसम कैसा है?"
        else:
            placeholder = "Example: What is the wheat price? How is the weather?"
        
        user_query = st.text_area("Your Question", placeholder=placeholder, height=100)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Ask", type="primary"):
                if user_query.strip():
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_query,
                        'timestamp': datetime.now()
                    })
                    
                    # Get bot response
                    with st.spinner("🤖 Processing your query..."):
                        try:
                            # Set user preferences
                            if st.session_state.user_city:
                                st.session_state.bot.set_user_city(st.session_state.user_city)
                            if st.session_state.user_crop:
                                st.session_state.bot.set_user_crop(st.session_state.user_crop)
                            if st.session_state.user_language:
                                st.session_state.bot.set_user_language(st.session_state.user_language)
                            
                            response = st.session_state.bot.process_query(user_query)
                            
                            # Add bot response to chat history
                            st.session_state.chat_history.append({
                                'role': 'bot',
                                'content': response,
                                'timestamp': datetime.now()
                            })
                            
                        except Exception as e:
                            error_msg = f"Error processing query: {e}"
                            st.session_state.chat_history.append({
                                'role': 'bot',
                                'content': error_msg,
                                'timestamp': datetime.now()
                            })
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        st.markdown("### Chat History")
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong> {message['content']}
                        <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Bot:</strong> {message['content']}
                        <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No messages yet. Start a conversation!")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Agricultural Analytics</h2>', unsafe_allow_html=True)
        
        # Price analytics
        st.markdown("### 💰 Price Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_city = st.text_input("City for Chart", value=st.session_state.user_city or "Kanpur")
        with col2:
            chart_crop = st.text_input("Crop for Chart", value=st.session_state.user_crop or "Wheat")
        
        if st.button("📈 Generate Price Chart"):
            chart = create_price_chart(chart_city, chart_crop)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected city and crop.")
        
        # Database insights
        st.markdown("### 📊 Database Insights")
        db_stats = get_database_stats()
        if db_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Price Records", f"{db_stats['price_records']:,}")
            with col2:
                st.metric("Soil Health Records", f"{db_stats['soil_records']:,}")
            with col3:
                st.metric("Unique Crops", db_stats['unique_crops'])
            with col4:
                st.metric("Markets Covered", db_stats['unique_markets'])
    
    with tab3:
        st.markdown('<h2 class="sub-header">🌤️ Weather Information</h2>', unsafe_allow_html=True)
        
        if not st.session_state.bot:
            st.warning("Please initialize the bot first.")
        else:
            weather_city = st.text_input("City for Weather", value=st.session_state.user_city or "Kanpur")
            
            if st.button("🌤️ Get Weather"):
                try:
                    with st.spinner("Fetching weather data..."):
                        weather_service = st.session_state.bot.weather_service
                        weather_data = weather_service.get_weather_data(weather_city)
                        
                        if weather_data:
                            # Current weather
                            st.markdown("### Current Weather")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Temperature", f"{weather_data.get('current_temp', 'N/A')}°C")
                            with col2:
                                st.metric("Humidity", f"{weather_data.get('current_humidity', 'N/A')}%")
                            with col3:
                                st.metric("Wind Speed", f"{weather_data.get('current_wind_speed', 'N/A')} km/h")
                            with col4:
                                st.metric("Condition", weather_data.get('current_condition', 'N/A'))
                            
                            # Weather forecast
                            if 'forecast' in weather_data:
                                st.markdown("### 7-Day Forecast")
                                forecast_df = pd.DataFrame(weather_data['forecast'])
                                st.dataframe(forecast_df, use_container_width=True)
                        else:
                            st.error("Could not fetch weather data.")
                except Exception as e:
                    st.error(f"Error fetching weather: {e}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">📋 Government Policies</h2>', unsafe_allow_html=True)
        
        if not st.session_state.bot:
            st.warning("Please initialize the bot first.")
        else:
            policy_query = st.text_input("Search Policy Information", placeholder="e.g., PM Kisan scheme, crop insurance")
            
            if st.button("🔍 Search Policies"):
                try:
                    with st.spinner("Searching policy documents..."):
                        policy_chatbot = st.session_state.bot.policy_chatbot
                        if policy_chatbot.is_loaded:
                            results = policy_chatbot.search_policies(policy_query)
                            if results:
                                st.markdown("### Policy Search Results")
                                for i, result in enumerate(results[:5]):  # Show top 5 results
                                    with st.expander(f"Result {i+1}: {result.get('document', 'Unknown')}"):
                                        st.write(result.get('content', 'No content available'))
                                        st.caption(f"Relevance: {result.get('relevance', 0):.2f}")
                            else:
                                st.info("No relevant policy documents found.")
                        else:
                            st.error("Policy database not loaded.")
                except Exception as e:
                    st.error(f"Error searching policies: {e}")
            
            # Policy statistics
            policy_stats = get_policy_stats()
            if policy_stats:
                st.markdown("### Policy Database Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Sections", policy_stats['total_sections'])
                with col2:
                    st.metric("Documents Processed", policy_stats['total_documents'])
    
    with tab5:
        st.markdown('<h2 class="sub-header">ℹ️ About the Agricultural Advisor Bot</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🌾 Mission</h3>
        <p>Empowering Indian farmers with AI-powered agricultural intelligence through multilingual support, 
        real-time market data, weather insights, and government policy information.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Key Features")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - 🌍 **Multilingual Support**: Hindi & English
            - 🎯 **Smart Query Classification**: 6 intent categories
            - 💰 **Intelligent Price Queries**: LLM-based SQL generation
            - 🌤️ **Weather-Based Advice**: Agricultural insights
            """)
        
        with col2:
            st.markdown("""
            - 📋 **Policy Support**: 12 government schemes
            - 🤖 **AI Integration**: Groq LLM for responses
            - 📊 **Real-time Data**: 35,522+ price records
            - 🔄 **Fallback Mechanisms**: Robust error handling
            """)
        
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        - **Price Data**: mandi_prices.csv (35,522 records)
        - **Weather Service**: Open-Meteo API
        - **Policy Documents**: 12 PDF files with vector search
        - **Soil Health**: 5 districts data
        """)
        
        st.markdown("### 🛠️ Technical Stack")
        st.markdown("""
        - **Python 3.8+**: Core language
        - **SQLite**: Local database
        - **FAISS**: Vector database
        - **Groq API**: LLM services
        - **Transformers**: Hugging Face models
        - **Streamlit**: Web interface
        """)

if __name__ == "__main__":
    main()
