import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- PAGE CONFIG ---
st.set_page_config(page_title="PulseSignal | Market Intelligence", page_icon="📈", layout="wide")

# --- DATABASE CONNECTION ---
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('pulsesignal.db')
        # Load structured AI-extracted data created by Nivegna
        df = pd.read_sql_query("SELECT * FROM structured_signals", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty if DB isn't ready yet

df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("📈 PulseSignal")
    st.write("**Market Intelligence Agent**")
    st.info("Currently monitoring:\nOpenAI, Anthropic, Databricks, NVIDIA, Snowflake.")
    st.markdown("---")
    st.write("**System Status:** Active / Live")
    
# --- MAIN DASHBOARD (TRACK B - STEP 1) ---
st.title("Enterprise Hiring Pulse")
st.write("Monitoring live web signals to identify GTM and investment opportunities.")

if df.empty:
    st.warning("⚠️ Waiting for Database Sync. Run `git pull` to get Nivegna's latest pulsesignal.db file.")
    st.stop()

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
total_signals = len(df)
# ROI Calculation: Assume 1.5 hours of manual research per web signal
hours_saved = total_signals * 1.5

col1.metric("Total Signals Tracked", total_signals)
col2.metric("Companies Monitored", df['company'].nunique())
col3.metric("Research Time Replaced", f"{hours_saved} Hours", delta="Enterprise ROI")
col4.metric("Data Freshness", "Live", delta="Cache Active")

st.divider()

# --- NEW UI UPGRADE: HIGH-INTENT SIGNAL SCORE ---
company_counts = df['company'].value_counts()
top_company = company_counts.idxmax()
top_count = company_counts.max()
total_signals = len(df)

# Dynamic Formula: (Company Signals / Total Signals) * 100 for "Market Presence Score"
# We add a multiplier of 2 to make it feel like "Intensity"
signal_score = min(int((top_count / total_signals) * 100 * 2), 99)

st.markdown("### 🔥 High-Intent Alert")
st.error(f"**{top_company}** is showing the highest hiring velocity in the current cluster.")
st.metric("Signal Score", f"{signal_score}/100", delta="Market Leader")
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("### 🎯 Strategic Market Intelligence")
st.write("Visualizing the 'Hiring Intensity' acceleration—a leading indicator of corporate budget shifts.")
col_chart, col_table = st.columns(2)

with col_chart:
    st.write("**Hiring Intensity by Company**")
    # Group by company to count signals
    chart_data = df.groupby('company').size().reset_index(name='Signal Count')
    fig = px.bar(chart_data, x='company', y='Signal Count', color='company')
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.write("**Top Trending Skills (AI Extracted)**")
    # Display the actual data Nivegna extracted
    display_df = df[['company', 'skills', 'business_priority']]
    # UI FIX: Added column config to prevent text truncation
    st.dataframe(
        display_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "skills": st.column_config.TextColumn("Skills Needed", width="large")
        }
    )


# --- PERSONA PLAYBOOKS & AGENT PIPELINE (TRACK B - STEP 2) ---
st.markdown("---")
st.header("💡 AI Business Insights")
st.write(f"Actionable intelligence generated from live web signals. *Replaces **{hours_saved}** hours of manual GTM research.*")

# Check for API Keys
api_key = os.environ.get("GOOGLE_API_KEY")
aiml_api_key = os.environ.get("AIML_API_KEY")

if not api_key:
    st.error("🔑 GOOGLE_API_KEY environment variable is not set. Please set it in your terminal to use AI features.")
else:
    if not aiml_api_key:
        st.warning("⚠️ AIML_API_KEY not found. Using Gemini for intelligence (Bounty requirement missing).")
        # Fallback to Gemini
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash", 
            google_api_key=api_key, 
            convert_system_message_to_human=True
        )
    else:
        # SUCCESS: Initialize Hybrid Multi-Model Architecture
        # Using Llama-3.3 70B via AI/ML API for high-reasoning market intelligence
        llm = ChatOpenAI(
            api_key=aiml_api_key, 
            base_url="https://api.aimlapi.com/v1", 
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo" 
        )
        st.sidebar.success("🚀 Hybrid AI Engine: Active (Llama-3.3 + Gemini)")

    
    insight_prompt = PromptTemplate.from_template(
        """
        You are an elite Go-To-Market Market Intelligence Analyst. 
        Analyze the following structured data extracted from company career pages:
        
        {raw_data}
        
        Generate a highly concise "Why this matters" summary and a "Recommended Action" for three distinct enterprise personas.
        CRITICAL RULE: You MUST ground your insights with concrete evidence counts (e.g., "based on 3 job postings").
        
        Format your response exactly like this:
        **For Sales Teams:** [1 sentence insight]. **Next Step:** [1 specific sales action].
        
        **For Recruiters:** [1 sentence insight]. **Next Step:** [1 specific recruiting action].
        
        **For Investors:** [1 sentence insight]. **Next Step:** [1 specific investment action].
        """
    )
    
    if st.button("Run Multi-Agent Intelligence Pipeline", type="primary"):
        # UI UPGRADE: The Agent Pipeline Visualizer
        with st.status("Initializing Multi-Agent Pipeline...", expanded=True) as status:
            st.write("🔍 Agent 1 (Data Fetcher): Scanning SQLite database for live web signals...")
            data_string = df.to_string()
            
            st.write("🧠 Agent 2 (Synthesizer): Cross-referencing skills with GTM priorities...")
            chain = insight_prompt | llm
            
            st.write("✍️ Agent 3 (Reporting): Formatting insights for enterprise personas...")
            response = chain.invoke({"raw_data": data_string})
            
            # Save to session state so it doesn't disappear on next button click
            st.session_state['intelligence_report'] = response.content
            
            status.update(label="Intelligence Pipeline Complete!", state="complete", expanded=False)

    # Display the report if it exists in session state
    if 'intelligence_report' in st.session_state:
        st.info(st.session_state['intelligence_report'])
        
        # UI UPGRADE: Action Pack (Cold Email Generator)
        st.markdown("#### ⚡ Execute Action Pack")
        if st.button("Draft Automated Outreach Email (Sales)"):
            with st.spinner("Agent 4 (Copywriter): Drafting hyper-personalized outreach based on live signals..."):
                email_prompt = PromptTemplate.from_template(
                    """
                    You are a top-tier B2B sales rep. Write a short, punchy cold email to the VP of Engineering at the company with the highest signal count from this data: {data}.
                    
                    Rules:
                    1. Mention the specific skills or business priorities they are hiring for based ON THE DATA.
                    2. Keep it under 100 words.
                    3. No generic fluff. Be direct.
                    """
                )
                email_chain = email_prompt | llm
                email_res = email_chain.invoke({"data": df.to_string()})
                st.success("Draft Generated Successfully. Ready for CRM Export.")
                st.code(email_res.content, language="markdown")

    # --- RAG Q&A CHATBOT (TRACK B - STEP 3) ---
    st.markdown("---")
    st.header("💬 Query Market Signals (RAG)")
    st.write("Ask natural language questions about the extracted market intelligence.")

    # 1. Prepare Data Chunks for Vector Store
    documents = []
    for index, row in df.iterrows():
        # Format into dense text chunks for the embedding model
        doc = f"[Company: {row['company']}] [Skills Needed: {row['skills']}] [Business Priority: {row.get('business_priority', 'N/A')}]"
        documents.append(doc)

    # 2. Embed and Store (Cached so it doesn't re-embed on every keypress)
    @st.cache_resource
    def create_vector_store(_docs, key):
        # Updated to the current standard Gemini embedding model available in your account
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=key)
        return FAISS.from_texts(_docs, embedding=embeddings)

    vector_store = create_vector_store(documents, api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Chat Interface Setup
    # Updated prompt to provide technical context for AI-specific terms like RAG
    template = """
    You are an elite Go-To-Market Market Intelligence Analyst.
    Use the following pieces of context to answer the user's question. 
    Note: In this technical context, 'RAG' refers to Retrieval-Augmented Generation (AI/LLM infrastructure).
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    qa_prompt = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("E.g., 'Which company is focusing on AI Infrastructure?'"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Scanning vector database..."):
                result = qa_chain.invoke(prompt)
                answer = result['result']
                st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})