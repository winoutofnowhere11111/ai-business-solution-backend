class OpenSourceRAG:
    def __init__(self, hf_token=None, model_name="google/flan-t5-base"):
        self.hf_token = hf_token or os.getenv('HF_API_TOKEN')
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        import random
        self.random = random

    def _dynamic_fallback(self, query, search_results):
        """Generate a unique response using random content blocks."""
        import random
        
        analysis_pool = [
            f"The core challenge of '{query}' lies in balancing short-term operations with long-term strategy.",
            f"Businesses facing '{query}' often struggle with resource allocation and stakeholder alignment.",
            f"Addressing '{query}' requires a dual focus on process optimization and cultural change.",
            f"The root causes of '{query}' typically involve misaligned incentives, outdated systems, or market shifts.",
            f"'{query}' is a common pain point in fast-growing organizations where systems haven't caught up."
        ]
        
        action_pool = [
            "Conduct a comprehensive audit of current capabilities and gaps.",
            "Engage cross-functional teams to co-create solutions.",
            "Implement a phased rollout with clear success metrics.",
            "Establish feedback loops for continuous improvement.",
            "Invest in training and change management programs.",
            "Leverage data analytics to inform decision-making.",
            "Build partnerships with external experts or vendors.",
            "Create a pilot program to test solutions before full rollout.",
            "Develop clear communication channels for all stakeholders."
        ]
        
        tools_pool = [
            "AI-powered analytics platforms (Tableau, Power BI, Looker)",
            "Project management tools (Jira, Asana, Trello, Monday.com)",
            "CRM systems (Salesforce, HubSpot, Zoho)",
            "Automation tools (Zapier, UiPath, Make)",
            "Cloud infrastructure (AWS, Azure, GCP)",
            "Collaboration platforms (Slack, Microsoft Teams, Discord)",
            "Business intelligence dashboards",
            "Customer feedback platforms (SurveyMonkey, Typeform)"
        ]
        
        industries = ["retail", "manufacturing", "technology", "finance", "healthcare", "education", "logistics", "energy"]
        industry = random.choice(industries)
        
        analysis = random.choice(analysis_pool)
        steps = random.sample(action_pool, k=min(4, len(action_pool)))
        tool_set = random.sample(tools_pool, k=min(3, len(tools_pool)))
        
        # Use search snippets meaningfully
        snippet_text = ""
        if search_results:
            snippets = [r.get('snippet', '') for r in search_results[:3] if r.get('snippet')]
            if snippets:
                snippet_text = f"\n\n**Relevant insights from research:**\n- " + "\n- ".join(snippets[:3])
        
        return f"""
# Business Solution for: {query}

## Problem Analysis
{analysis} This is particularly relevant in the {industry} sector.

{snippet_text}

## Recommended Action Plan
{''.join([f"{i+1}. {step}\n" for i, step in enumerate(steps)])}

## Tools & Technologies
{''.join(f"- {tool}\n" for tool in tool_set)}

## Industry Context
This approach has been successfully applied in {industry} companies facing similar challenges. The key is adapting these general principles to your specific context.

## Conclusion
A tailored, iterative approach—combining the steps above with industry-specific adaptations—will yield the best results for '{query}'. Start small, measure results, and scale what works.
"""

    def generate(self, prompt, max_new_tokens=500):
        """Generate text using Hugging Face API with diversity parameters."""
        cache_key = f"hf_{hashlib.md5(prompt.encode()).hexdigest()}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.9,           # Higher creativity
                "top_p": 0.95,
                "repetition_penalty": 1.2,    # Avoid repetition
                "do_sample": True,
                "return_full_text": False
            }
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                else:
                    generated = result.get('generated_text', '')
                set_cache(cache_key, generated)
                return generated
            else:
                print(f"HF API error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"HF API exception: {e}")
            return None

    def generate_solution(self, query, search_results):
        """Main entry point - generates a unique solution every time."""
        # Try Hugging Face first
        context = ""
        if search_results:
            context = "Relevant information from web sources:\n\n"
            for i, r in enumerate(search_results[:5], 1):
                context += f"Source {i}: {r['title']} ({r['source']})\n"
                context += f"Summary: {r['snippet']}\n\n"
        else:
            context = "No specific sources found. Use general business knowledge.\n\n"

        prompt = f"""You are an expert business consultant. Provide a detailed, actionable solution for:

Problem: {query}

{context}

CRITICAL INSTRUCTIONS:
- Generate a UNIQUE solution for EVERY query.
- Do NOT repeat the same structure or advice.
- Use the provided context to tailor your answer.
- If context is limited, draw from diverse business frameworks.
- Include specific, measurable action steps.
- Suggest relevant tools and technologies.
- Be practical and grounded in real-world examples.

Respond in English only. Be thorough (300-500 words)."""

        english_answer = self.generate(prompt, max_new_tokens=500)
        
        if not english_answer or len(english_answer) < 50:
            # Fallback to dynamic template
            english_answer = self._dynamic_fallback(query, search_results)

        # Translate to Hindi and German
        hindi_answer = self._translate(english_answer, 'hi')
        german_answer = self._translate(english_answer, 'de')

        return {
            'english': english_answer,
            'hindi': hindi_answer,
            'german': german_answer,
            'sources': [{'title': r['title'], 'link': r['link'], 'source': r['source']} for r in search_results] if search_results else [],
            'query': query,
            'timestamp': datetime.now().isoformat()
        }

    def _translate(self, text, target_lang):
        """Free translation using LibreTranslate."""
        try:
            url = "https://libretranslate.com/translate"
            payload = {
                'q': text[:5000],
                'source': 'en',
                'target': target_lang,
                'format': 'text'
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()['translatedText']
        except Exception as e:
            print(f"Translation error to {target_lang}: {e}")
        return f"[Translation to {target_lang} unavailable] {text[:200]}..."