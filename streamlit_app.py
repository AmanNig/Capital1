import streamlit as st
from agricultural_advisor_bot import AgriculturalAdvisorBot

st.set_page_config(page_title="Agricultural Advisor Bot", page_icon="🌾")

st.title("🌾 Agricultural Advisor Bot")
st.markdown(
    """
    Welcome! This is an interactive UI for your Agricultural Advisor Bot.
    - Set your city, crop, and language.
    - Ask any question about weather, policy, prices, technical support, or general agriculture.
    """
)

# Session state for bot
if "bot" not in st.session_state:
    st.session_state.bot = AgriculturalAdvisorBot()

bot = st.session_state.bot

# Sidebar for user preferences
st.sidebar.header("User Preferences")
city = st.sidebar.text_input("City", value=bot.get_user_city())
crop = st.sidebar.text_input("Primary Crop", value=bot.get_user_crop())
language = st.sidebar.selectbox(
    "Language",
    ["English", "Hindi", "Marathi", "Gujarati", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam", "Punjabi"],
    index=0 if bot.get_user_language() == "English" else 1
)

if st.sidebar.button("Update Preferences"):
    st.success(bot.set_user_city(city))
    st.success(bot.set_user_crop(crop))
    st.success(bot.set_user_language(language))

st.markdown("#### Ask your question below:")

user_query = st.text_area("Your Question", height=80, placeholder="E.g. What is the price of wheat in Kanpur?")

if st.button("Get Advice"):
    if not city or city == "Not set":
        st.warning("Please set your city in the sidebar first.")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            response = bot.process_query(user_query)
        st.markdown("**Advisor Response:**")
        st.markdown(response, unsafe_allow_html=True)

st.markdown("---")
st.markdown("ℹ️ Powered by Llama3-8b-8192 via Groq API. For best results, ensure your API keys and data files are set up.")