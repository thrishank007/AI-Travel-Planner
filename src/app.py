from textwrap import dedent
import streamlit as st
import os
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
# Initialize Hugging Face Inference Client with better error handling
def get_hf_client():
    api_key = st.session_state.get("hf_api_key") or os.getenv("HF_API_KEY")
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

# Initialize session state for API keys, correctly prioritizing .env file
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = os.getenv("HF_API_KEY", "")

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
    
    # API Key input that does NOT display the loaded key
    user_provided_key = st.text_input(
        "HuggingFace API Key", 
        type="password", 
        placeholder="Enter key to override .env",
        help="Required for AI model access. Overrides .env key for this session."
    )
    
    # Update session state ONLY if the user enters a new key
    if user_provided_key:
        st.session_state.hf_api_key = user_provided_key
        st.success("API key updated for this session.")

    # API Status indicator
    if st.session_state.hf_api_key:
        st.markdown('<div class="api-status api-connected">✅ HuggingFace API: Connected</div>', unsafe_allow_html=True)
        # Inform the user if the key is from the .env file
        if os.getenv("HF_API_KEY") == st.session_state.hf_api_key and not user_provided_key:
             st.caption("Using API key from .env file.")
    else:
        st.markdown('<div class="api-status api-disconnected">❌ HuggingFace API: Not Connected</div>', unsafe_allow_html=True)
        st.info("💡 You can set the HF_API_KEY in a .env file or enter it above.")
    
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
    destination = st.text_input(
        "Destination(s)", 
        placeholder="e.g., Tokyo, Japan OR Tokyo, Kyoto, Osaka (separate multiple destinations with commas)",
        help="Enter one destination or multiple destinations separated by commas for a multi-city trip"
    )
    
    # Parse and display destinations
    destinations = []
    if destination:
        destinations = [dest.strip() for dest in destination.split(',') if dest.strip()]
        if len(destinations) > 1:
            st.info(f"🌍 Multi-destination trip detected: {len(destinations)} destinations")
            for i, dest in enumerate(destinations, 1):
                st.caption(f"  {i}. {dest}")
        elif len(destinations) == 1:
            st.info(f"📍 Single destination: {destinations[0]}")
    
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
            if len(destinations) > 1:
                days_per_destination = num_days // len(destinations)
                remaining_days = num_days % len(destinations)
                st.info(f"Trip duration: {num_days} days total")
                st.caption(f"📊 Approximate time per destination: {days_per_destination} days each" + 
                          (f" (+ {remaining_days} extra days to distribute)" if remaining_days > 0 else ""))
            else:
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

# Enhanced function to create multi-destination prompts
def create_multi_destination_research_prompt(destinations, num_days, travel_style, travelers, budget_range, interests, accommodation_type, mobility, special_requirements):
    """Create comprehensive research prompt for single or multiple destinations"""
    
    if len(destinations) == 1:
        # Single destination prompt
        return f"""
        Research {destinations[0]} for a {num_days}-day {travel_style.lower()} trip for {travelers} travelers.
        Budget: {budget_range}
        Interests: {', '.join(interests)}
        Accommodation preferences: {', '.join(accommodation_type)}
        Transportation: {mobility}
        Special requirements: {special_requirements or 'None'}
        
        Provide comprehensive information about:
        1. Top attractions and activities with current status and pricing
        2. Recommended accommodations with availability and rates
        3. Local dining options with recent reviews
        4. Transportation methods and current schedules/costs
        5. Cultural tips and local customs
        6. Weather considerations for travel dates
        7. Current budget estimates and cost breakdowns
        8. Best neighborhoods to stay in
        9. Day trip opportunities from the city
        10. Shopping and entertainment options
        """
    else:
        # Multi-destination prompt
        destinations_list = ', '.join(destinations)
        days_per_destination = num_days // len(destinations)
        
        return f"""
        Research a {num_days}-day multi-destination {travel_style.lower()} trip covering: {destinations_list}
        For {travelers} travelers with {budget_range} budget.
        Interests: {', '.join(interests)}
        Accommodation preferences: {', '.join(accommodation_type)}
        Transportation: {mobility}
        Special requirements: {special_requirements or 'None'}
        
        MULTI-DESTINATION RESEARCH REQUIREMENTS:
        
        FOR EACH DESTINATION ({destinations_list}):
        1. **Key Attractions & Activities** (prioritize based on {', '.join(interests)})
        2. **Recommended Stay Duration** (suggest optimal days out of {days_per_destination} average per destination)
        3. **Best Accommodations** ({', '.join(accommodation_type)}) with location recommendations
        4. **Must-try Local Dining** and signature dishes
        5. **Transportation within the city** ({mobility} preferences)
        
        INTER-CITY LOGISTICS:
        1. **Transportation between destinations** (flights, trains, buses, cars)
        2. **Travel time and costs** between each destination
        3. **Optimal route planning** (most efficient order to visit)
        4. **Luggage considerations** for multi-city travel
        
        BUDGET BREAKDOWN:
        1. **Per-destination costs** (accommodation, food, activities)
        2. **Transportation costs** between cities
        3. **Total estimated budget** for the entire trip
        
        PRACTICAL MULTI-CITY TIPS:
        1. **Packing strategies** for multiple destinations
        2. **Weather variations** across different cities/regions
        3. **Cultural differences** and customs for each destination
        4. **Best booking strategies** for multi-city accommodations
        5. **Time zone considerations** if applicable
        
        Please structure the response clearly for each destination and provide a comparative analysis.
        """

def create_multi_destination_itinerary_prompt(destinations, num_days, start_date, end_date, travelers, travel_style, budget_range, interests, accommodation_type, mobility, special_requirements, research_results=None):
    """Create comprehensive itinerary prompt for single or multiple destinations"""
    
    if len(destinations) == 1:
        # Single destination itinerary
        context = f"""
        Create a detailed {num_days}-day itinerary for {destinations[0]}:
        Duration: {num_days} days (from {start_date} to {end_date})
        Travelers: {travelers} people
        Travel Style: {travel_style}
        Budget: {budget_range}
        Interests: {', '.join(interests)}
        Accommodations: {', '.join(accommodation_type)}
        Transportation: {mobility}
        Special Requirements: {special_requirements or 'None'}
        """
    else:
        # Multi-destination itinerary
        destinations_list = ', '.join(destinations)
        days_per_destination = num_days // len(destinations)
        
        context = f"""
        Create a detailed {num_days}-day MULTI-DESTINATION itinerary covering: {destinations_list}
        Duration: {num_days} days total (from {start_date} to {end_date})
        Travelers: {travelers} people
        Travel Style: {travel_style}
        Budget: {budget_range}
        Interests: {', '.join(interests)}
        Accommodations: {', '.join(accommodation_type)}
        Transportation: {mobility}
        Special Requirements: {special_requirements or 'None'}
        
        MULTI-DESTINATION ITINERARY REQUIREMENTS:
        
        1. **OPTIMAL ROUTE PLANNING**:
           - Determine the best order to visit destinations
           - Consider geographical proximity and transportation efficiency
           - Factor in arrival/departure logistics
        
        2. **TIME ALLOCATION**:
           - Distribute {num_days} days across {len(destinations)} destinations
           - Suggest ideal duration for each destination based on attractions
           - Account for travel days between destinations
        
        3. **DAY-BY-DAY BREAKDOWN**:
           For each day, specify:
           - **Location**: Which destination you're in
           - **Morning**: Specific activities with timing
           - **Afternoon**: Attractions/experiences with travel time
           - **Evening**: Dining and entertainment recommendations
           - **Travel days**: Detailed transportation between cities
        
        4. **TRANSITION PLANNING**:
           - **Check-out/Check-in logistics**
           - **Transportation booking details** (flights, trains, etc.)
           - **Luggage handling** during city transitions
           - **Buffer time** for unexpected delays
        
        5. **DESTINATION-SPECIFIC HIGHLIGHTS**:
           For each destination, prioritize:
           - Must-see attractions based on {', '.join(interests)}
           - Local experiences unique to that location
           - Best dining spots for authentic cuisine
           - Cultural activities and local customs
        
        6. **PRACTICAL CONSIDERATIONS**:
           - **Weather-appropriate activities** for travel dates
           - **Booking priorities** (what to reserve in advance)
           - **Emergency contacts** for each destination
           - **Communication tips** (language, currency, customs)
        
        Please structure as:
        - **Overview & Route Summary**
        - **Day 1-X: [Destination 1]**
        - **Day X: Travel Day (Destination 1 → Destination 2)**
        - **Day X-Y: [Destination 2]**
        - **Continue for all destinations...**
        - **Final Tips & Recommendations**
        """
    
    if research_results:
        context += f"\n\nRESEARCH INFORMATION TO INCORPORATE:\n{research_results}"
    
    return context
if destination and num_days > 0 and destinations:
    # Choose agent type based on API availability
    use_offline = not st.session_state.hf_api_key
    
    if use_offline:
        st.warning("⚠️ Operating in offline mode. Provide HuggingFace API key for AI-powered responses.")
        AgentClass = OfflineAgent
    else:
        AgentClass = HFChatAgent
    
    # Display trip summary
    if len(destinations) > 1:
        st.success(f"✅ Multi-destination trip planned: {len(destinations)} destinations in {num_days} days")
    else:
        st.success(f"✅ Single destination trip planned: {destinations[0]} for {num_days} days")
    
    # Initialize agents with proper error handling
    try:
        # Enhanced researcher for multi-destination support
        researcher = AgentClass(
            name="Multi-Destination Researcher",
            role="Travel Research Specialist for Single and Multi-City Trips",
            description=dedent(f"""
            You are an expert travel researcher specializing in {travel_style.lower()} travel.
            You excel at planning both single-destination and multi-destination trips.
            
            Current assignment: Research {'multi-destination trip covering: ' + ', '.join(destinations) if len(destinations) > 1 else 'single destination: ' + destinations[0]}
            Trip duration: {num_days} days for {travelers} travelers
            Budget: {budget_range} | Interests: {', '.join(interests)}
            
            For multi-destination trips, you provide comparative analysis, optimal routing, 
            and comprehensive logistics planning. For single destinations, you provide 
            in-depth local expertise and detailed recommendations.
            """),
            instructions=[
                f"Research {'all destinations: ' + ', '.join(destinations) if len(destinations) > 1 else destinations[0]} for {travel_style.lower()} travelers",
                f"Focus on {', '.join(interests)} activities and {', '.join(accommodation_type)} accommodations",
                f"Consider {budget_range} budget and {mobility} transportation preferences",
                "For multi-destination trips: provide inter-city logistics, optimal routing, and time allocation",
                "For single destinations: provide comprehensive local insights and day trip options",
                "Include practical details like pricing, booking info, and travel times",
                "Structure response clearly by destination with comparative insights",
                "Response should be comprehensive but well-organized (under 3k tokens)"
            ]
        )

        # Enhanced planner for multi-destination support
        planner = AgentClass(
            name="Multi-Destination Planner",
            role="Travel Itinerary Specialist for Complex Multi-City Trips",
            description=dedent(f"""
            You are a professional travel planner specializing in creating detailed itineraries 
            for both single and multi-destination trips.
            
            Current assignment: Create {num_days}-day itinerary for {'multi-city trip: ' + ', '.join(destinations) if len(destinations) > 1 else destinations[0]}
            Travelers: {travelers} with {travel_style.lower()} style | Budget: {budget_range}
            Interests: {', '.join(interests)} | Transport: {mobility}
            Special requirements: {special_requirements or 'None'}
            
            For multi-destination trips, you excel at optimal routing, time allocation, 
            and seamless transitions between cities while maximizing experiences.
            """),
            instructions=[
                f"Create detailed {num_days}-day itinerary covering {'all destinations: ' + ', '.join(destinations) if len(destinations) > 1 else destinations[0]}",
                "For multi-destination: plan optimal route, allocate time efficiently, include travel days",
                "For single destination: create comprehensive daily schedules with variety and depth",
                "Structure each day with morning, afternoon, evening activities",
                "Include specific recommendations for dining, transportation, and accommodations",
                "Provide estimated costs and time allocations for each activity",
                "Consider group size, travel style, and special requirements",
                "Add practical tips, booking priorities, and local insights",
                "For multi-city: include detailed transition logistics and luggage handling",
                "Response should be detailed but well-structured (under 3k tokens)"
            ]
        )

        # Button layout with updated labels
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        destination_type = "Multi-Destination" if len(destinations) > 1 else "Destination"
        
        with col_btn1:
            research_btn = st.button(f"🔍 Research {destination_type}", type="primary", use_container_width=True)
        
        with col_btn2:
            generate_btn = st.button(f"📅 Generate {destination_type} Itinerary", type="primary", use_container_width=True)
        
        with col_btn3:
            quick_tips_btn = st.button("💡 Quick Tips", use_container_width=True)

        # Enhanced research functionality
        if research_btn:
            research_type = "multi-destination" if len(destinations) > 1 else "destination"
            with st.spinner(f"🔍 Researching {research_type}..."):
                try:
                    research_prompt = create_multi_destination_research_prompt(
                        destinations, num_days, travel_style, travelers, 
                        budget_range, interests, accommodation_type, mobility, special_requirements
                    )
                    
                    research_response = researcher.run(research_prompt)
                    st.session_state.research_results = research_response.content
                    st.success(f"✅ {destination_type} research completed successfully!")
                except Exception as e:
                    st.error(f"❌ Research failed: {str(e)}")

        # Enhanced itinerary generation
        if generate_btn:
            itinerary_type = "multi-destination" if len(destinations) > 1 else ""
            with st.spinner(f"📅 Creating your personalized {itinerary_type} itinerary..."):
                try:
                    context = create_multi_destination_itinerary_prompt(
                        destinations, num_days, start_date, end_date, travelers, 
                        travel_style, budget_range, interests, accommodation_type, 
                        mobility, special_requirements, st.session_state.research_results
                    )
                    
                    itinerary_response = planner.run(context)
                    st.session_state.itinerary = itinerary_response.content
                    st.success(f"✅ {destination_type} itinerary generated successfully!")
                except Exception as e:
                    st.error(f"❌ Itinerary generation failed: {str(e)}")

        # Enhanced quick tips
        if quick_tips_btn:
            if len(destinations) > 1:
                tips_prompt = f"Provide essential multi-destination travel tips for visiting {', '.join(destinations)} in a single trip. Include inter-city travel logistics, packing strategies, and destination-specific advice."
            else:
                tips_prompt = f"Provide essential travel tips for visiting {destinations[0]}"
                
            try:
                tips_agent = AgentClass("Tips", "Travel Advisor", "You provide practical travel tips and advice", [])
                tips_response = tips_agent.run(tips_prompt)
                destination_list = ', '.join(destinations)
                st.info(f"💡 **Travel Tips for {destination_list}:**\n\n{tips_response.content}")
            except Exception as e:
                st.error(f"Failed to generate tips: {str(e)}")

        # Display results with enhanced formatting
        if st.session_state.research_results:
            research_title = f"🔍 Research Results - {', '.join(destinations)}"
            st.subheader(research_title)
            with st.expander("View Research Details", expanded=False):
                st.write(st.session_state.research_results)

        if st.session_state.itinerary:
            itinerary_title = f"📅 Your Personalized {'Multi-Destination ' if len(destinations) > 1 else ''}Itinerary"
            st.subheader(itinerary_title)
            with st.expander("View Itinerary Details", expanded=False):
                st.write(st.session_state.itinerary)

            # Enhanced export options
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                if st.session_state.itinerary:
                    # Create more descriptive filename
                    filename_destinations = '_'.join([dest.replace(', ', '_').replace(' ', '_') for dest in destinations[:3]])  # Limit to first 3 destinations
                    if len(destinations) > 3:
                        filename_destinations += f"_plus{len(destinations)-3}more"
                    
                    filename = f"{filename_destinations}_{num_days}days_itinerary.txt"
                    
                    st.download_button(
                        label="📄 Download Itinerary",
                        data=st.session_state.itinerary,
                        file_name=filename,
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
    st.warning("⚠️ Please provide destination(s) and valid travel dates to generate an itinerary.")
    if not destination:
        st.info("📍 Enter your destination(s) in the Trip Details section")
    if num_days <= 0:
        st.info("📅 Select valid start and end dates for your trip")

# Enhanced help section
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

with st.expander("🌍 Multi-Destination Planning"):
    st.markdown("""
    **How to plan multi-destination trips:**
    
    **Input Format:**
    - Single destination: `Tokyo, Japan`
    - Multiple destinations: `Tokyo, Kyoto, Osaka` or `Paris, London, Amsterdam`
    - Separate destinations with commas
    
    **Multi-Destination Features:**
    - **Automatic route optimization** for efficient travel
    - **Time allocation** across destinations based on attractions
    - **Inter-city transportation** planning and costs
    - **Comparative analysis** of destinations
    - **Seamless transitions** with luggage and check-in logistics
    
    **Best Practices:**
    - **2-4 destinations** work best for most trip lengths
    - **Allow travel days** in your total duration
    - **Consider geographical proximity** for efficient routing
    - **Research visa requirements** for multiple countries
    
    **Example Multi-City Trips:**
    - **Japan Circuit**: Tokyo, Kyoto, Osaka (7-10 days)
    - **Europe Tour**: Paris, Amsterdam, Berlin (10-14 days)
    - **Southeast Asia**: Bangkok, Singapore, Kuala Lumpur (8-12 days)
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
    - Multi-destination planning requires more processing time
    - Free tier APIs have rate limits
    - Consider upgrading to paid tiers for faster responses
    - Use offline mode for immediate basic results
    
    **Multi-destination planning tips**
    - Keep destination count reasonable (2-4 destinations)
    - Ensure adequate trip duration for multiple destinations
    - Consider visa and transportation requirements between countries
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by HuggingFace + Llama 4 | Built with Streamlit</p>
    <p>🌟 Plan smarter, travel better - single or multi-destination!</p>
</div>
""", unsafe_allow_html=True)