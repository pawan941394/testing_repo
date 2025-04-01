import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from agno.agent import Agent
from agno.models.openai.like import OpenAILike

# Page configuration
st.set_page_config(page_title="Finsocial Digital System - Intelligent Query Processing with Enhanced Model Responses", layout="wide")

# Company branding
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #34495e;">Intelligent Query Processing with Enhanced Model Responses</h2>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h3>Finsocial Digital System</h3>
</div>
""", unsafe_allow_html=True)
st.sidebar.header("Model Configuration")
model_name = st.sidebar.text_input("Model Name", value="openthinker:32b")
api_key = st.sidebar.text_input("API Key", value="sk-no-key-needed", type="password")
base_url = st.sidebar.text_input(
    "Base URL", 
    value=""
)

# Static template file path
template_file = "merged.json"

# Load JSON template file
@st.cache_data
def load_templates(file_path):
    if not file_path.endswith(".json"):
        st.error("Unsupported file format. Use JSON.")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        
        st.error(f"Error loading template file: {e}")
        return []

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Agent with dynamic configuration
def get_agent(model_name, api_key, base_url):
    return Agent(
        model=OpenAILike(
            id=model_name,
            api_key=api_key,
            base_url=base_url
        )
    )

# Load templates and prepare embeddings
@st.cache_data
def prepare_embeddings(_template_file):
    templates = load_templates(_template_file)
    if not templates:
        return [], None
    
    template_texts = [t["general"] for t in templates]
    embedding_model = get_embedding_model()
    template_embeddings = embedding_model.encode(template_texts, convert_to_tensor=True)
    return templates, template_embeddings

templates, template_embeddings = prepare_embeddings(template_file)

def rank_templates(query, templates, template_embeddings):
    """Rank templates based on cosine similarity."""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, template_embeddings)[0].cpu().numpy()
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
    return [(templates[i], float(scores[i])) for i in ranked_indices]  # Convert float32 to float

def process_query(query, agent, templates, template_embeddings):
    """Processes the query without caching."""
    threshold = 0.5  # Cosine similarity threshold
    ranked_templates = rank_templates(query, templates, template_embeddings)
    top_templates = [(template, score) for template, score in ranked_templates if score > threshold]
    top_10 = top_templates[:10] if len(top_templates) >= 10 else top_templates

    if not top_10:
        st.warning("⚠️ There is no such requirement to enhance the answer for the following query.")
        # Fallback: Generate a direct answer without templates
        fallback_prompt = f"""
        The user has asked the following question, but no matching templates were found in our knowledge base.
        Please provide a comprehensive, step-by-step answer using your general knowledge:
        
        {query}
        
        Respond with a detailed, structured answer that breaks down the solution clearly.
        """
        fallback_response = agent.run(fallback_prompt).content.strip()
        
        return {
            "top_templates": [],
            "cot_summary": "No relevant templates found",
            "direct_summary": "Direct fallback response generated",
            "full_direct_response": fallback_response,
            "template_direct_response": fallback_response
          }  # Using the same response in both fields
           
    # Enhance low scores
    avg_score = np.mean([t[1] for t in top_10])
    enhanced_templates = [
        (t, float(avg_score * 1.1)) if s < avg_score * 0.8 else (t, float(s))
        for t, s in top_10
    ]
    
    # Extract unique content (handle missing 'content' key safely)
    unique_content = list(set(
        [content for template, _ in enhanced_templates for content in template.get('content', [])]
    ))
    ensemble_text = "\n".join(unique_content)

    with st.spinner("Understanding your query..."):
        # Generate Chain-of-Thought summary
        summary_response = agent.run(f"Summarize this content concisely: {ensemble_text}").content.strip()
    
    with st.spinner("Processing direct response..."):
        # Generate direct model response for the query
        direct_response = agent.run(query).content.strip()
        summarized_direct_response = agent.run(f"Summarize this response concisely: {direct_response}").content.strip()
    
    with st.spinner("Generating Advanced Answer..."):
        # Generate a template direct response using the top-10 template summary and the user query.
        template_direct_prompt = (
            f'''
            The following summary contains key insights extracted from various templates to serve as the model’s reasoning base:
            {ensemble_text}

            Using the above summary as the foundation, please provide a detailed, step-by-step answer for the following query:
            {query}
            '''
        )
        template_direct_response = agent.run(template_direct_prompt).content.strip()
        print("*"*50, template_direct_response)
        

    response_data = {
        "top_templates": [{"general": t["general"], "score": float(s)} for t, s in top_10],
        "cot_summary": summary_response,
        "direct_summary": summarized_direct_response,
        "full_direct_response": direct_response,
        "template_direct_response": template_direct_response
    }

    return response_data

# Main interface
if templates:
    query = st.text_area("Enter your query:", height=100)
    process_button = st.button("Process Query")
    
    if process_button and query:
        agent = get_agent(model_name, api_key, base_url)
        with st.spinner("Processing your query..."):
            result = process_query(query, agent, templates, template_embeddings)
        
        if "error" not in result:
            # Display top templates
            #st.subheader("🔹 Top Matching Templates")
            #template_data = []
            #for i, template in enumerate(result["top_templates"]):
                #template_data.append({
                    #"Rank": i+1,
                    #"Template": template['general'],
                    #"Score": f"{template['score']:.4f}"
                #})
            
            #st.dataframe(template_data, use_container_width=True)
            
            # Display summary in a column and responses in dropdowns
            col1 = st.columns(1)[0]
            
            
            # Display answers in dropdowns/expanders
            with st.expander("📌 Direct Response From Model", expanded=True):
                st.write(result["full_direct_response"])
            
            # Full direct response in expander
            with st.expander("Enhanced Response",expanded=True):
                st.write(result.get("template_direct_response", "No full response available"))
            
            # Add footer with company branding
            st.markdown("""
            <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #f0f2f6; padding: 10px; display: flex; justify-content: center; align-items: center; border-top: 2px solid #2c3e50;">
                <p style="color: #2c3e50; margin: 0; text-align: center;">
                    © 2024 Finsocial Digital System. All rights reserved.<br>
                    <small>Empowering Financial Intelligence</small>
                </p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.error("No templates loaded. Please check your template file path and format.")
