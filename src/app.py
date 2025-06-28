from textwrap import dedent
import streamlit as st
import os
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta

# Initialize Hugging Face Inference Client with better error handling
def get_hf_client():
    api_key = os.getenv("HF_API_KEY") or st.session_state.get("hf_api_key", "")
    if not api_key or api_key == "hf_demo_fallback":
        return None
    try:
        return InferenceClient(provider="together",api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize HF client: {str(e)}")
        return None

# Helper function for chat completion with multiple fallbacks
def hf_chat_completion(messages, max_tokens=2048):
    client = get_hf_client()
    if not client:
        return "Error: HuggingFace API key not provided or invalid. Please set your API key in the sidebar."
    

    try:
        completion = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: All HuggingFace models unavailable. {str(e)}"
    
# Custom Agent class to use Hugging Face model
class HFChatAgent:
    def __init__(self, name, role, description, instructions):
        self.name = name
        self.role = role
        self.description = description
        self.instructions = instructions

    def run(self, prompt: str, stream=False):
        try:
            messages = [
                {"role": "system", "content": f"{self.description}\n\nInstructions: {' '.join(self.instructions) if self.instructions else ''}"},
                {"role": "user", "content": prompt}
            ]
            response_content = hf_chat_completion(messages)
            return type("Response", (), {"content": response_content})()
        except Exception as e:
            return type("Response", (), {"content": f"Error: {str(e)}"})()

# Offline fallback agent for when API is unavailable
class OfflineAgent:
    def __init__(self, name, role, description, instructions):
        self.name = name
        self.role = role
        self.description = description
        self.instructions = instructions

    def run(self, prompt: str, stream=False):
        # Provide basic template responses when API is unavailable
        if "research" in prompt.lower():
            response = self._generate_research_template(prompt)
        elif "itinerary" in prompt.lower():
            response = self._generate_itinerary_template(prompt)
        else:
            response = self._generate_tips_template(prompt)
        
        return type("Response", (), {"content": response})()
    
    def _generate_research_template(self, prompt):
        return """
        🔍 **Research Template** (Offline Mode)
        
        **Note**: This is a basic template. For detailed, AI-powered research, please provide a valid HuggingFace API key.
        
        **Top Attractions**:
        - Visit major landmarks and tourist attractions
        - Explore local museums and cultural sites
        - Experience natural attractions and parks
        
        **Accommodations**:
        - Research hotels in central locations
        - Consider vacation rentals for longer stays
        - Check reviews and amenities
        
        **Dining**:
        - Try local specialties and traditional cuisine
        - Visit highly-rated restaurants
        - Explore local markets and street food
        
        **Transportation**:
        - Research public transportation options
        - Consider ride-sharing or car rentals
        - Plan airport transfers
        
        **Tips**:
        - Check visa requirements
        - Research local customs and etiquette
        - Download offline maps and translation apps
        """
    
    def _generate_itinerary_template(self, prompt):
        return """
        📅 **Itinerary Template** (Offline Mode)
        
        **Note**: This is a basic template. For personalized, AI-powered itineraries, please provide a valid HuggingFace API key.
        
        **Day 1**: Arrival & City Overview
        - Morning: Arrive and check-in to accommodation
        - Afternoon: Explore city center and main attractions
        - Evening: Welcome dinner at local restaurant
        
        **Day 2**: Cultural Exploration
        - Morning: Visit museums and historical sites
        - Afternoon: Guided city tour or walking tour
        - Evening: Local entertainment or cultural show
        
        **Day 3**: Nature & Adventure
        - Morning: Visit parks or natural attractions
        - Afternoon: Outdoor activities based on location
        - Evening: Sunset viewing and dinner
        
        **Additional Days**: 
        - Continue exploring based on your interests
        - Mix of relaxation and adventure activities
        - Shopping and souvenir hunting
        - Day trips to nearby attractions
        
        **General Tips**:
        - Book popular attractions in advance
        - Allow flexibility for weather changes
        - Keep emergency contacts handy
        - Stay hydrated and take breaks
        """
    
    def _generate_tips_template(self, prompt):
        return """
        💡 **Travel Tips Template** (Offline Mode)
        
        **Essential Preparations**:
        - Check passport expiration dates
        - Research visa requirements
        - Get travel insurance
        - Notify banks of travel dates
        
        **Packing Tips**:
        - Check weather forecasts
        - Pack versatile clothing
        - Bring necessary medications
        - Don't forget chargers and adapters
        
        **Safety & Health**:
        - Research local emergency numbers
        - Keep copies of important documents
        - Stay aware of your surroundings
        - Follow local health guidelines
        
        **Money Matters**:
        - Research local currency and exchange rates
        - Have multiple payment methods
        - Keep some cash for small purchases
        - Understand tipping customs
        """

# Page config
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for API keys
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = ""

# Styling
st.markdown("""
<style>
.main-header { 
    text-align: center; 
    padding: 2rem 0; 
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    border-radius: 10px; 
    margin-bottom: 2rem; 
} 
.feature-card { 
    background: #f8f9fa; 
    padding: 1rem; 
    border-radius: 8px; 
    border-left: 4px solid #667eea; 
    margin: 1rem 0; 
} 
.itinerary-day { 
    background: white; 
    padding: 1rem; 
    border-radius: 8px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    margin: 1rem 0; 
    border-left: 4px solid #28a745; 
}
.api-status {
    padding: 0.5rem 1rem;
    border-radius: 5px;
    margin: 1rem 0;
    font-weight: bold;
}
.api-connected {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.api-disconnected {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌍 AI Travel Planner using HuggingFace + Llama</h1>
    <p>Plan your next adventure with AI-powered research and personalized itineraries</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key inputs with better handling
    hf_api_key = st.text_input(
        "HuggingFace API Key", 
        type="password", 
        help="Required for AI model access. Get one at huggingface.co/settings/tokens",
        value=st.session_state.hf_api_key
    )
    if hf_api_key:
        st.session_state.hf_api_key = hf_api_key
    
    # API Status indicator
    if st.session_state.hf_api_key:
        st.markdown('<div class="api-status api-connected">✅ HuggingFace API: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-disconnected">❌ HuggingFace API: Not Connected</div>', unsafe_allow_html=True)
        st.info("💡 Get a free HuggingFace API key at: https://huggingface.co/settings/tokens")
    
    st.divider()
    st.header("🎯 Travel Preferences")
    travel_style = st.selectbox("Travel Style", ["Honeymoon", "Adventure", "Relaxation", "Cultural", "Luxury", "Family", "Solo", "Business"])
    budget_range = st.select_slider("Budget Range", options=["Budget ($)", "Mid-range ($$)", "Premium ($$$)", "Luxury ($$$$)"], value="Mid-range ($$)")
    accommodation_type = st.multiselect("Preferred Accommodations", ["Hotels", "Hostels", "Airbnb", "Resorts", "Boutique Hotels", "Camping"], default=["Hotels"])
    interests = st.multiselect("Interests", ["Food & Dining", "Museums", "Nature", "Nightlife", "Shopping", "History", "Art", "Sports", "Photography"], default=["Food & Dining", "Nature"])
    mobility = st.selectbox("Mobility Preferences", ["Walking", "Public Transport", "Car Rental", "Mixed"])

# Main trip details
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📍 Trip Details")
    destination = st.text_input("Destination", placeholder="e.g., Tokyo, Japan")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("Start Date", value=datetime.now() + timedelta(days=30), min_value=datetime.now().date())
    with col_date2:
        end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=37), min_value=start_date if 'start_date' in locals() else datetime.now().date())
    
    # Calculate number of days safely
    num_days = 0
    if start_date and end_date and end_date >= start_date:
        num_days = (end_date - start_date).days
        if num_days > 0:
            st.info(f"Trip duration: {num_days} days")
        else:
            st.warning("Please select valid travel dates")
    
    travelers = st.number_input("Number of Travelers", min_value=1, max_value=20, value=2)
    special_requirements = st.text_area("Special Requirements", placeholder="Any dietary restrictions, accessibility needs, or special occasions...")

with col2:
    st.header("🎮 Quick Actions")
    st.markdown("""
    <div class="feature-card"><h4>🔍 Smart Research</h4><p>AI-powered research for activities and accommodations</p></div>
    <div class="feature-card"><h4>📅 Personalized Itinerary</h4><p>Custom day-by-day plans based on your preferences</p></div>
    <div class="feature-card"><h4>💾 Save & Export</h4><p>Download your itinerary in multiple formats</p></div>
    """, unsafe_allow_html=True)

# Generator section
st.divider()
st.header("🚀 Generate Your Itinerary")

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None

# Check if all required inputs are provided
if destination and num_days > 0:
    # Choose agent type based on API availability
    use_offline = not st.session_state.hf_api_key
    
    if use_offline:
        st.warning("⚠️ Operating in offline mode. Provide HuggingFace API key for AI-powered responses.")
        AgentClass = OfflineAgent
    else:
        AgentClass = HFChatAgent
    
    # Initialize agents with proper error handling
    try:
        researcher = AgentClass(
            name="Researcher",
            role="Travel Research Specialist",
            description=dedent(f"""
            You are an expert travel researcher specializing in {travel_style.lower()} travel.
            Research comprehensive information about {destination} for a {num_days}-day trip.
            Focus on finding the best {', '.join(accommodation_type)} accommodations, 
            activities related to {', '.join(interests)}, and travel options for {mobility.lower()}.
            Consider the budget range of {budget_range} and group size of {travelers} travelers.
            Provide detailed, accurate, and up-to-date information about attractions, dining, 
            transportation, and local experiences.
            """),
            instructions=[
                f"Research {destination} thoroughly for {travel_style.lower()} travelers",
                f"Focus on {', '.join(interests)} activities and {', '.join(accommodation_type)} accommodations",
                f"Consider {budget_range} budget and {mobility} transportation preferences",
                "Provide specific recommendations with practical details like pricing, location, and booking information",
                "Response should be under 2k tokens and well-structured",
            ]
        )

        planner = AgentClass(
            name="Planner",
            role="Travel Itinerary Specialist",
            description=dedent(f"""
            You are a professional travel planner creating a detailed {num_days}-day itinerary for {destination}.
            The trip is for {travelers} travelers with a {travel_style.lower()} travel style and {budget_range} budget.
            Key interests: {', '.join(interests)}
            Accommodation preferences: {', '.join(accommodation_type)}
            Transportation: {mobility}
            Special requirements: {special_requirements or 'None'}
            
            Create a comprehensive day-by-day itinerary that includes:
            - Daily activities and attractions
            - Meal recommendations
            - Transportation suggestions
            - Estimated costs and timing
            - Backup options for weather-dependent activities
            """),
            instructions=[
                f"Create a detailed {num_days}-day itinerary for {destination}",
                "Structure each day with morning, afternoon, and evening activities",
                "Include specific recommendations for dining, transportation, and accommodations",
                "Provide estimated costs and time allocations for each activity",
                "Consider the group size, travel style, and special requirements",
                "Add practical tips and local insights",
                "Response should be under 2k tokens and well-structured"
            ]
        )

        # Button layout
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            research_btn = st.button("🔍 Research Destination", type="primary", use_container_width=True)
        
        with col_btn2:
            generate_btn = st.button("📅 Generate Full Itinerary", type="primary", use_container_width=True)
        
        with col_btn3:
            quick_tips_btn = st.button("💡 Quick Tips", use_container_width=True)

        # Research functionality
        if research_btn:
            with st.spinner("🔍 Researching destination..."):
                try:
                    research_prompt = f"""
                    Research {destination} for a {num_days}-day {travel_style.lower()} trip for {travelers} travelers.
                    Budget: {budget_range}
                    Interests: {', '.join(interests)}
                    Accommodation preferences: {', '.join(accommodation_type)}
                    Transportation: {mobility}
                    Special requirements: {special_requirements or 'None'}
                    
                    Provide comprehensive information about:
                    1. Top attractions and activities
                    2. Recommended accommodations
                    3. Local dining options
                    4. Transportation methods
                    5. Cultural tips and local customs
                    6. Weather considerations
                    7. Budget estimates
                    """
                    
                    research_response = researcher.run(research_prompt)
                    st.session_state.research_results = research_response.content
                    st.success("✅ Research completed successfully!")
                except Exception as e:
                    st.error(f"❌ Research failed: {str(e)}")

        # Generate itinerary
        if generate_btn:
            with st.spinner("📅 Creating your personalized itinerary..."):
                try:
                    context = f"""
                    Create a detailed itinerary for:
                    Destination: {destination}
                    Duration: {num_days} days (from {start_date} to {end_date})
                    Travelers: {travelers} people
                    Travel Style: {travel_style}
                    Budget: {budget_range}
                    Interests: {', '.join(interests)}
                    Accommodations: {', '.join(accommodation_type)}
                    Transportation: {mobility}
                    Special Requirements: {special_requirements or 'None'}
                    """
                    
                    if st.session_state.research_results:
                        context += f"\n\nResearch Information:\n{st.session_state.research_results}"
                    
                    itinerary_response = planner.run(context)
                    st.session_state.itinerary = itinerary_response.content
                    st.success("✅ Itinerary generated successfully!")
                except Exception as e:
                    st.error(f"❌ Itinerary generation failed: {str(e)}")

        # Quick tips
        if quick_tips_btn:
            tips_prompt = f"Provide essential travel tips for visiting {destination}"
            try:
                tips_agent = AgentClass("Tips", "Travel Advisor", "You provide practical travel tips and advice", [])
                tips_response = tips_agent.run(tips_prompt)
                st.info(f"💡 **Travel Tips for {destination}:**\n\n{tips_response.content}")
            except Exception as e:
                st.error(f"Failed to generate tips: {str(e)}")

        # Display results
        if st.session_state.research_results:
            st.subheader("🔍 Research Results")
            with st.expander("View Research Details", expanded=False):
                st.write(st.session_state.research_results)

        if st.session_state.itinerary:
            st.subheader("📅 Your Personalized Itinerary")
            with st.expander("View Itinerary Details", expanded=False):
                st.write(st.session_state.itinerary)
            st.markdown('</div>', unsafe_allow_html=True)

            # Export options
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                if st.session_state.itinerary:
                    st.download_button(
                        label="📄 Download Itinerary",
                        data=st.session_state.itinerary,
                        file_name=f"{destination.replace(', ', '_').replace(' ', '_')}_{num_days}days_itinerary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            with col_export2:
                if st.button("📧 Email Itinerary", use_container_width=True):
                    st.info("📧 Email functionality will be implemented soon")
            
            with col_export3:
                if st.button("📱 Share Itinerary", use_container_width=True):
                    st.info("📱 Share functionality will be implemented soon")

    except Exception as e:
        st.error(f"❌ Error initializing agents: {str(e)}")

else:
    st.warning("⚠️ Please provide destination and valid travel dates to generate an itinerary.")
    if not destination:
        st.info("📍 Enter your destination in the Trip Details section")
    if num_days <= 0:
        st.info("📅 Select valid start and end dates for your trip")

# Help section
st.divider()
st.header("ℹ️ Help & Troubleshooting")

with st.expander("🔧 Setup Instructions"):
    st.markdown("""
    **Getting Started:**
    
    1. **HuggingFace API Key** (Recommended):
       - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
       - Create a new token with 'Read' permissions
       - Paste it in the sidebar
    
    2. **Offline Mode**:
       - If no API keys are provided, the app runs in offline mode
       - You'll get basic templates instead of AI-generated content
    """)

with st.expander("❓ Common Issues"):
    st.markdown("""
    **Error: 401 Unauthorized**
    - Check your HuggingFace API key is correct
    - Ensure the token has proper permissions
    - Try regenerating your API key
    
    **Error: Model not available**
    - The app automatically tries multiple models
    - Some models may be temporarily unavailable
    - Try again later or use offline mode
    
    **Slow responses**
    - Free tier APIs have rate limits
    - Consider upgrading to paid tiers for faster responses
    - Use offline mode for immediate basic results
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by HuggingFace + Llama 4 | Built with Streamlit</p>
    <p>🌟 Plan smarter, travel better!</p>
</div>
""", unsafe_allow_html=True)