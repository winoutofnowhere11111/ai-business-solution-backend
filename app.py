# ================================================================
# app.py - Advanced AI Search Backend with RAG
# ================================================================
# This is a Flask-based backend that provides an AI-powered search
# with Retrieval-Augmented Generation (RAG) capabilities.
# ================================================================

import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import re
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ================================================================
# CONFIGURATION
# ================================================================

# Try to import openai, but don't crash if not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Using fallback responses.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY')
GOOGLE_SEARCH_CX = os.getenv('GOOGLE_SEARCH_CX')
USE_FREE_ALTERNATIVES = os.getenv('USE_FREE_ALTERNATIVES', 'true').lower() == 'true'

# ================================================================
# SEARCH FUNCTIONS
# ================================================================

def search_web(query, max_results=5):
    """
    Search the web for relevant articles using Google Programmable Search
    or free alternatives.
    """
    results = []
    
    # Try using Google Programmable Search API first
    if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX:
        try:
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_SEARCH_CX}&q={quote_plus(query)}&num={max_results}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', 'No title'),
                        'link': item.get('link', '#'),
                        'snippet': item.get('snippet', 'No description available'),
                        'source': item.get('displayLink', 'Unknown source')
                    })
                return results
        except Exception as e:
            print(f"Google Search API error: {e}")
    
    # Try using Serper API as alternative
    if SERPER_API_KEY:
        try:
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "q": query,
                "num": max_results
            })
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            data = response.json()
            
            if 'organic' in data:
                for item in data['organic'][:max_results]:
                    results.append({
                        'title': item.get('title', 'No title'),
                        'link': item.get('link', '#'),
                        'snippet': item.get('snippet', 'No description available'),
                        'source': item.get('source', 'Unknown source')
                    })
                return results
        except Exception as e:
            print(f"Serper API error: {e}")
    
    # Fallback: Use demo search results
    if USE_FREE_ALTERNATIVES:
        try:
            return [
                {
                    'title': f'Business Strategy for: {query[:40]}...',
                    'link': f'https://example.com/search?q={quote_plus(query)}',
                    'snippet': f'This is a simulated search result for "{query}". To get real results, configure Google or Serper API keys in your environment variables.',
                    'source': 'Demo Source'
                },
                {
                    'title': f'Operational Solutions for: {query[:40]}...',
                    'link': f'https://example.com/search?q={quote_plus(query)}&topic=operations',
                    'snippet': f'Simulated operational insights for "{query}". Configure API keys for real-world data.',
                    'source': 'Demo Source'
                },
                {
                    'title': f'Technology Implementation for: {query[:40]}...',
                    'link': f'https://example.com/search?q={quote_plus(query)}&topic=technology',
                    'snippet': f'Simulated technology solutions for "{query}". Add real API keys for production use.',
                    'source': 'Demo Source'
                }
            ][:min(max_results, 3)]
        except Exception as e:
            print(f"Free search fallback error: {e}")
    
    return results

def generate_ai_solution(query, search_results):
    """
    Generate a comprehensive AI solution using RAG based on search results.
    """
    # Prepare context from search results
    context = ""
    if search_results:
        context = "Based on the following research sources:\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"Source {i}: {result['title']} ({result['source']})\n"
            context += f"Summary: {result['snippet']}\n\n"
    else:
        context = "No specific sources were found. Generating a comprehensive response based on business knowledge.\n\n"
    
    # Generate response using OpenAI if available
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            
            # Prepare the prompt
            prompt = f"""You are an expert business consultant and AI assistant. You need to provide a comprehensive, well-researched solution to the following business problem:

**Business Problem:** {query}

{context}

**Your Task:** Provide a detailed solution that:
1. Analyzes the problem from multiple perspectives
2. Provides actionable steps to solve it
3. Uses best practices and proven frameworks
4. Includes relevant examples and case studies
5. Mentions specific tools, technologies, or methods

**Output Format:**
- Provide the solution in **English**, **Hindi**, and **German**
- Each language section should be 700+ words
- Total solution should be 2000+ words
- Include a "Sources" section with references

Be comprehensive, practical, and research-backed in your response.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are an expert business consultant providing comprehensive solutions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            ai_response = response.choices[0].message.content.strip()
            
            # Parse the response to extract language sections
            return parse_ai_response(ai_response, search_results)
        except Exception as e:
            print(f"OpenAI API error: {e}")
    
    # Fallback: Generate a template-based response
    return generate_fallback_response(query, search_results)

def parse_ai_response(response, search_results):
    """
    Parse the AI response to extract English, Hindi, German sections.
    """
    # Default structure
    result = {
        'english': '',
        'hindi': '',
        'german': '',
        'sources': [],
        'query': '',
        'timestamp': datetime.now().isoformat()
    }
    
    # Try to extract language sections
    sections = {
        'english': ['English:', '**English**', 'English Solution'],
        'hindi': ['Hindi:', '**Hindi**', 'हिंदी:', 'Hindi Solution'],
        'german': ['German:', '**German**', 'Deutsch:', 'German Solution']
    }
    
    current_section = None
    current_text = []
    
    for line in response.split('\n'):
        line_lower = line.lower().strip()
        
        # Check if this line starts a new section
        for lang, markers in sections.items():
            for marker in markers:
                if marker.lower() in line_lower and len(line) < 50:
                    # Save previous section if any
                    if current_section:
                        result[current_section] = '\n'.join(current_text).strip()
                    current_section = lang
                    current_text = []
                    break
            if current_section:
                break
        
        # If we're in a section, accumulate text
        if current_section:
            current_text.append(line)
        else:
            # Auto-detect based on language patterns
            if any(char in line for char in 'अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'):
                if not result['hindi']:
                    result['hindi'] += line + '\n'
            elif any(char in line for char in 'äöüßÄÖÜ'):
                if not result['german']:
                    result['german'] += line + '\n'
            else:
                if not result['english']:
                    result['english'] += line + '\n'
    
    # Save last section
    if current_section and current_text:
        result[current_section] = '\n'.join(current_text).strip()
    
    # Add sources
    if search_results:
        for item in search_results:
            result['sources'].append({
                'title': item.get('title', 'Unknown'),
                'link': item.get('link', '#'),
                'source': item.get('source', 'Unknown source')
            })
    else:
        # Add a default source
        result['sources'].append({
            'title': 'Business Knowledge Base',
            'link': '#',
            'source': 'AI Assistant'
        })
    
    # Add query
    result['query'] = query
    
    # Ensure each section has content
    for lang in ['english', 'hindi', 'german']:
        if not result[lang] or len(result[lang]) < 100:
            result[lang] = generate_language_fallback(query, lang, search_results)
    
    return result

def generate_language_fallback(query, language, search_results):
    """
    Generate fallback content for a specific language.
    """
    templates = {
        'english': f"""
# Comprehensive Solution for: {query}

## Problem Analysis
This is a detailed analysis of the business problem. The solution involves multiple approaches including strategic planning, operational improvements, and stakeholder engagement.

## Action Plan
1. **Step 1**: Assess the current situation thoroughly.
   - Analyze current processes and performance metrics
   - Identify gaps and bottlenecks
   - Gather stakeholder feedback

2. **Step 2**: Develop a clear, data-driven strategy.
   - Set SMART goals and objectives
   - Create detailed implementation roadmap
   - Allocate resources effectively

3. **Step 3**: Implement solutions with regular monitoring.
   - Execute the action plan
   - Track progress against KPIs
   - Make adjustments as needed

4. **Step 4**: Adjust based on feedback and results.
   - Collect and analyze performance data
   - Implement continuous improvement
   - Document lessons learned

## Best Practices
- Use data-driven decision making
- Engage all stakeholders early
- Iterate based on feedback
- Document all processes
- Maintain clear communication

## Tools & Technologies
- AI/ML for predictive analytics
- CRM systems for customer management
- Automation tools for operational efficiency
- Project management software for tracking
- Business intelligence tools for reporting

## Case Study Example
A mid-sized manufacturing company faced a similar challenge and achieved success by following these principles. They implemented a phased approach, starting with pilot projects and scaling successful initiatives.

## Conclusion
A structured, step-by-step approach will lead to successful outcomes. Regular review and adaptation are key to long-term success.
""",
        'hindi': f"""
# {query} के लिए व्यापक समाधान

## समस्या विश्लेषण
यह व्यावसायिक समस्या का विस्तृत विश्लेषण है। समाधान में कई दृष्टिकोण शामिल हैं।

## कार्य योजना
1. **चरण 1**: वर्तमान स्थिति का आकलन करें।
   - मौजूदा प्रक्रियाओं और प्रदर्शन मेट्रिक्स का विश्लेषण करें
   - अंतराल और बाधाओं की पहचान करें
   - हितधारकों से फीडबैक प्राप्त करें

2. **चरण 2**: एक रणनीति विकसित करें।
   - SMART लक्ष्य निर्धारित करें
   - विस्तृत कार्यान्वयन रोडमैप बनाएं
   - संसाधनों का प्रभावी आवंटन करें

3. **चरण 3**: समाधान लागू करें।
   - कार्य योजना को क्रियान्वित करें
   - KPI के विरुद्ध प्रगति ट्रैक करें
   - आवश्यकतानुसार समायोजन करें

4. **चरण 4**: निगरानी और समायोजन करें।
   - प्रदर्शन डेटा एकत्र और विश्लेषण करें
   - निरंतर सुधार लागू करें
   - सीखे गए सबक दस्तावेज़ करें

## सर्वोत्तम अभ्यास
- डेटा-संचालित निर्णय लेना
- हितधारकों को शामिल करना
- फीडबैक के आधार पर पुनरावृत्ति करना
- सभी प्रक्रियाओं को दस्तावेज़ करना
- स्पष्ट संचार बनाए रखना

## उपकरण और प्रौद्योगिकियाँ
- AI/ML भविष्यवाणी विश्लेषण के लिए
- ग्राहक प्रबंधन के लिए CRM प्रणाली
- दक्षता के लिए स्वचालन उपकरण
- ट्रैकिंग के लिए परियोजना प्रबंधन सॉफ्टवेयर
- रिपोर्टिंग के लिए बिजनेस इंटेलिजेंस उपकरण

## केस स्टडी उदाहरण
एक मध्यम आकार की विनिर्माण कंपनी ने इसी तरह की चुनौती का सामना किया और इन सिद्धांतों का पालन करके सफलता प्राप्त की।

## निष्कर्ष
एक संरचित दृष्टिकोण सफल परिणामों की ओर ले जाएगा।
""",
        'german': f"""
# Umfassende Lösung für: {query}

## Problemanalyse
Dies ist eine detaillierte Analyse des Geschäftsproblems. Die Lösung umfasst mehrere Ansätze.

## Aktionsplan
1. **Schritt 1**: Bewertung der aktuellen Situation.
2. **Schritt 2**: Entwicklung einer Strategie.
3. **Schritt 3**: Umsetzung der Lösungen.
4. **Schritt 4**: Überwachung und Anpassung.

## Best Practices
- Datengesteuerte Entscheidungsfindung
- Einbeziehung von Stakeholdern
- Iteration basierend auf Feedback
- Dokumentation aller Prozesse
- Klare Kommunikation

## Werkzeuge und Technologien
- KI/ML für prädiktive Analysen
- CRM-Systeme für das Kundenmanagement
- Automatisierungstools für Effizienz
- Projektmanagementsoftware für die Verfolgung
- Business-Intelligence-Tools für Berichte

## Fazit
Ein strukturierter Ansatz wird zu erfolgreichen Ergebnissen führen.
"""
    }
    
    return templates.get(language, templates['english'])

def generate_fallback_response(query, search_results):
    """
    Generate a fallback response when AI is not available.
    """
    result = {
        'english': generate_language_fallback(query, 'english', search_results),
        'hindi': generate_language_fallback(query, 'hindi', search_results),
        'german': generate_language_fallback(query, 'german', search_results),
        'sources': [{'title': 'Business Knowledge Base', 'link': '#', 'source': 'AI Assistant'}],
        'query': query,
        'timestamp': datetime.now().isoformat()
    }
    return result

# ================================================================
# API ROUTES
# ================================================================

@app.route('/')
def home():
    """Root endpoint to verify the backend is running."""
    return jsonify({
        'status': 'AI Business Solution Backend is running',
        'message': 'Use POST /api/search with JSON {"query": "your problem"}',
        'endpoints': {
            '/': 'GET - This status message',
            '/api/search': 'POST - AI-powered search',
            '/api/health': 'GET - Health check'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search', methods=['POST'])
def search():
    """
    API endpoint for AI-powered search with RAG.
    Expects JSON: { "query": "business problem description" }
    Returns: { "success": true, "data": { ... } }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Please provide a query'
            }), 400
        
        # Step 1: Search the web for relevant sources
        search_results = search_web(query)
        
        # Step 2: Generate AI solution using RAG
        solution = generate_ai_solution(query, search_results)
        
        # Step 3: Return the result
        return jsonify({
            'success': True,
            'data': solution
        })
        
    except Exception as e:
        print(f"Search endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

# ================================================================
# RUN THE APP
# ================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
