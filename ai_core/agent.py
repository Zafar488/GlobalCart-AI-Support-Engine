import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 1. API Key Setup (Kept Exactly as Requested)
os.environ["GROQ_API_KEY"] = "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"

# 2. Define the exact JSON structure we want the LLM to output
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Must be strictly 'Positive', 'Negative', or 'Neutral'. Any complaint about delays, safety, or damage is 'Negative'.")
    key_aspect: str = Field(description="The main feature discussed (e.g., Battery, Shipping, Price, Safety)")
    requires_escalation: bool = Field(description="True ONLY if the sentiment is Negative or if there is a safety risk.")
    auto_response: str = Field(description="A professional 2-line draft reply to the customer from the Support Team.")

# 3. Initialize the ultra-fast Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1, # Lowered temperature for more deterministic output
)

# Bind the Pydantic model to force structured output
structured_llm = llm.with_structured_output(ReviewAnalysis)

# 4. Create the core logic
def analyze_review_with_agent(review_text: str):
    print(f"\nAnalyzing: '{review_text}'")
    print("-" * 50)
    
    # Prompt the LLM with stricter persona instructions
    prompt = f"""
    You are an expert Customer Success AI Agent.
    Analyze the following customer review and extract the required information.
    Pay close attention to mixed sentiments (e.g., "Good product, but terrible battery" is NEGATIVE).
    
    Review: "{review_text}"
    """
    
    # Execute the agent
    result = structured_llm.invoke(prompt)
    
    # Print the structured results
    print(f"🧠 Sentiment:   {result.sentiment}")
    print(f"🎯 Key Aspect:  {result.key_aspect}")
    print(f"🚨 Escalate?:   {'YES - Routing to Support' if result.requires_escalation else 'NO - System Normal'}")
    print(f"✉️ Draft Reply: {result.auto_response}")
    print("\n")

# --- Let's Test The Agent ---
if __name__ == "__main__":
    test_reviews = [
        "The camera quality is absolutely stunning, but the battery dies in just 3 hours. It's frustrating.",
        "Terrible experience. The package arrived two weeks late and the screen was scratched. I want a refund.",
        "Smooth interface and very fast delivery. Will definitely buy again!"
    ]
    
    for review in test_reviews:
        analyze_review_with_agent(review)