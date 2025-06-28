# 🌍 AI Travel Planner

An intelligent travel planning application that uses HuggingFace's Llama 4 model to generate personalized travel itineraries and research destinations. Built with Streamlit for an intuitive web interface.

## ✨ Features

- **🔍 Smart Destination Research**: AI-powered research for attractions, accommodations, dining, and transportation
- **📅 Personalized Itinerary Generation**: Custom day-by-day travel plans based on your preferences
- **💡 Travel Tips**: Essential tips and local insights for your destination
- **📱 Multiple Travel Styles**: Support for Honeymoon, Adventure, Cultural, Luxury, Family, Solo, and Business travel
- **💰 Budget-Aware Planning**: Tailored recommendations for different budget ranges
- **🚶 Transportation Preferences**: Plans adapted to walking, public transport, car rental, or mixed mobility
- **📄 Export Options**: Download your itinerary as a text file
- **🔄 Offline Mode**: Basic templates available when API is unavailable

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- HuggingFace API key (optional but recommended for AI features)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thrishank007/AI-Travel-Planner
   cd AI-Travel-Planner
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Setting up HuggingFace API

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with 'Read' permissions
3. Enter the token in the sidebar of the application

## 🎯 Usage

### Basic Workflow

1. **Configure your preferences** in the sidebar:
   - Enter your HuggingFace API key
   - Select travel style, budget range, and interests
   - Choose accommodation and transportation preferences

2. **Enter trip details**:
   - Destination (e.g., "Tokyo, Japan")
   - Start and end dates
   - Number of travelers
   - Special requirements

3. **Generate your plan**:
   - **🔍 Research Destination**: Get comprehensive information about your destination
   - **📅 Generate Full Itinerary**: Create a detailed day-by-day plan
   - **💡 Quick Tips**: Get essential travel tips

4. **Export and save**:
   - Download your itinerary as a text file
   - View detailed research results and itinerary in expandable sections

### Travel Preferences

- **Travel Styles**: Honeymoon, Adventure, Relaxation, Cultural, Luxury, Family, Solo, Business
- **Budget Ranges**: Budget ($), Mid-range ($$), Premium ($$$), Luxury ($$$$)
- **Accommodations**: Hotels, Hostels, Airbnb, Resorts, Boutique Hotels, Camping
- **Interests**: Food & Dining, Museums, Nature, Nightlife, Shopping, History, Art, Sports, Photography
- **Transportation**: Walking, Public Transport, Car Rental, Mixed

## 🏗️ Project Structure

```
gt_hk/
├── src/
│   └── app.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Technical Details

### Core Components

- **[`get_hf_client()`](src/app.py)**: Initializes HuggingFace Inference Client
- **[`hf_chat_completion()`](src/app.py)**: Handles chat completion with error handling
- **[`HFChatAgent`](src/app.py)**: Custom agent class for AI-powered responses
- **[`OfflineAgent`](src/app.py)**: Fallback agent for offline mode with templates

### AI Model

- **Primary Model**: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **Provider**: Together AI via HuggingFace
- **Max Tokens**: 2048 per response

### Dependencies

```txt
streamlit              # Web application framework
huggingface_hub       # HuggingFace API integration
textwrap              # Text formatting utilities
datetime              # Date and time handling
os                    # Operating system interface
```

## 🎨 User Interface

### Main Sections

1. **Configuration Sidebar**:
   - API key input
   - Travel preferences
   - Budget and style selection

2. **Trip Details**:
   - Destination input
   - Date selection
   - Traveler count
   - Special requirements

3. **Quick Actions**:
   - Feature overview cards
   - Action buttons

4. **Results Display**:
   - Research results
   - Generated itinerary
   - Export options

### Styling Features

- Gradient header design
- Responsive card layouts
- API status indicators
- Color-coded sections
- Mobile-friendly interface

## ⚠️ Error Handling

The application includes comprehensive error handling for:

- **API Connection Issues**: Graceful fallback to offline mode
- **Invalid API Keys**: Clear error messages and setup instructions
- **Model Unavailability**: Automatic retry mechanisms
- **Input Validation**: Checks for required fields and valid dates
- **Network Errors**: Timeout and connection error handling

## 🔄 Offline Mode

When no API key is provided, the application operates in offline mode with:

- **Research Templates**: Basic destination research structure
- **Itinerary Templates**: Generic day-by-day planning templates
- **Travel Tips Templates**: Essential travel preparation guidelines

## 🚧 Troubleshooting

### Common Issues

**Error: 401 Unauthorized**
- Verify your HuggingFace API key is correct
- Ensure the token has proper permissions
- Try regenerating your API key

**Error: Model not available**
- Some models may be temporarily unavailable
- Try again later or use offline mode
- Check HuggingFace status page

**Slow responses**
- Free tier APIs have rate limits
- Consider upgrading to paid tiers for faster responses
- Use offline mode for immediate basic results

### Setup Issues

**Dependencies not installing**
- Ensure Python 3.1+ is installed
- Try using a virtual environment
- Update pip: `pip install --upgrade pip`

**Streamlit not starting**
- Check if port 8501 is available
- Try: `streamlit run src/app.py --port 8502`

## 🔮 Future Enhancements

- 📧 Email itinerary functionality
- 📱 Social sharing features
- 🗺️ Interactive maps integration
- 💾 Save/load itinerary profiles
- 🌐 Multi-language support
- 📊 Budget tracking and expense estimation
- 🎨 Custom itinerary themes
- 🔄 Real-time weather integration

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the HuggingFace documentation
3. Open an issue in the repository

---

**Built with ❤️ using Streamlit and HuggingFace**

*🌟 Plan smarter, travel better!*