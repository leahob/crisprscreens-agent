import streamlit as st
st.set_page_config(page_title="CRISPR LLM Agent App", layout="wide")

import pandas as pd
import duckdb
from rapidfuzz import process, fuzz
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
import re
from llms import llm

# --- Configuration ---
S3_BUCKET = "bioinf-trailbiomed"
S3_PREFIX = "data/dev-crispr/TEST-SCREEN/SCREENS/"
INDEX_KEY = "data/dev-crispr/TEST-SCREEN/index.txt"
REGION = "us-west-2"

# --- DuckDB S3 Connection ---
def get_duckdb_s3_connection(region=REGION):
    con = duckdb.connect()
    con.execute("INSTALL aws;")
    con.execute("LOAD aws;")
    con.execute(f"SET s3_region = '{region}';")
    con.execute("""
        CREATE SECRET (
            TYPE S3,
            PROVIDER credential_chain
        );
    """)
    return con

# --- Load Index Metadata ---
def load_index_df(index_s3_path=f's3://{S3_BUCKET}/{INDEX_KEY}', region=REGION):
    with get_duckdb_s3_connection(region=region) as con:
        index_df = con.execute(f"SELECT * FROM read_csv_auto('{index_s3_path}', header=True, sep='\t')").df()
    return index_df

index_df = load_index_df()

# --- Controlled Vocabularies ---
virus_options = index_df['CONDITION_NAME'].dropna().unique().tolist() if 'CONDITION_NAME' in index_df.columns else []
cell_line_options = index_df['CELL_LINE'].dropna().unique().tolist() if 'CELL_LINE' in index_df.columns else []
species_options = index_df['ORGANISM_OFFICIAL'].dropna().unique().tolist() if 'ORGANISM_OFFICIAL' in index_df.columns else []

# --- Query Functions ---
def load_screen_tab_filtered(
    screen_id,
    bucket=S3_BUCKET,
    prefix=S3_PREFIX,
    gene_symbols=None,
    hit_value=None,
    score2_max=None
):
    s3_path = f's3://{bucket}/{prefix}TEST-SCREEN_{screen_id}-0.0.0.screen.tab.txt'
    where_clauses = []
    if gene_symbols is not None:
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]
        gene_list = ','.join([f"'{g}'" for g in gene_symbols])
        where_clauses.append(f"OFFICIAL_SYMBOL IN ({gene_list})")
    if hit_value is not None:
        where_clauses.append(f"HIT = '{hit_value}'")
    if score2_max is not None:
        where_clauses.append(f'CAST("SCORE.2 (pos_fdr)" AS DOUBLE) <= {score2_max}')
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"SELECT *, {screen_id} AS SCREEN_ID FROM read_csv_auto('{s3_path}', header=True, sep='\t') {where_sql}"
    with get_duckdb_s3_connection() as con:
        return con.execute(query).df()

def get_results_by_genes(gene_symbols, index_df=index_df, bucket=S3_BUCKET, prefix=S3_PREFIX):
    if isinstance(gene_symbols, str):
        gene_symbols = [gene_symbols]
    results = []
    for screen_id in index_df['SCREEN_ID'].unique():
        df = load_screen_tab_filtered(screen_id, bucket, prefix, gene_symbols=gene_symbols)
        if not df.empty:
            meta = index_df[index_df['SCREEN_ID'] == screen_id]
            for col in meta.columns:
                if col not in df.columns:
                    df.loc[:, col] = meta.iloc[0][col]
            results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def get_results_by_screens(screen_ids, bucket=S3_BUCKET, prefix=S3_PREFIX):
    if isinstance(screen_ids, (str, int)):
        screen_ids = [screen_ids]
    results = []
    for screen_id in screen_ids:
        df = load_screen_tab_filtered(screen_id, bucket, prefix)
        if not df.empty:
            results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def get_results_by_condition(virus_query, index_df=index_df, bucket=S3_BUCKET, prefix=S3_PREFIX, virus_col='CONDITION_NAME', threshold=70):
    exact = index_df[index_df[virus_col].str.lower() == virus_query.lower()]
    if not exact.empty:
        screen_ids = exact['SCREEN_ID'].unique()
    else:
        choices = index_df[virus_col].dropna().unique()
        best_match, score, _ = process.extractOne(virus_query, choices, scorer=fuzz.token_sort_ratio)
        if score > threshold:
            screen_ids = index_df[index_df[virus_col] == best_match]['SCREEN_ID'].unique()
        else:
            return pd.DataFrame()  # No match found
    return get_results_by_screens(screen_ids, bucket, prefix)


gene_tool = Tool(
    name="GeneQuery",
    func=lambda gene: get_results_by_genes(gene),
    description=(
        "Retrieve CRISPR screen results for a given gene symbol or list of symbols. "
        f"Valid viruses: {', '.join(virus_options[:5])}... "
        f"Valid cell lines: {', '.join(cell_line_options[:5])}... "
        f"Valid species: {', '.join(species_options[:5])}... "
        "Use the List tools for full options."
    )
)

screen_tool = Tool(
    name="ScreenQuery",
    func=lambda screen_id: get_results_by_screens(screen_id).head(100).to_markdown(index=False),
    description="Retrieve up to the first 100 CRISPR screen results for a given SCREEN_ID or list of SCREEN_IDs, formatted as a markdown table."
)

condition_tool = Tool(
    name="ConditionQuery",
    func=lambda virus: get_results_by_condition(virus),
    description=(
        "Retrieve CRISPR screen results for a given virus or condition name (fuzzy matching supported). "
        f"Valid viruses: {', '.join(virus_options[:5])}..."
    )
)

def list_viruses():
    return virus_options

def list_cell_lines():
    return cell_line_options

def list_species():
    return species_options

list_viruses_tool = Tool(
    name="ListViruses",
    func=lambda: list_viruses(),
    description="List all valid virus names for CRISPR screens."
)
list_cell_lines_tool = Tool(
    name="ListCellLines",
    func=lambda: list_cell_lines(),
    description="List all valid cell line names for CRISPR screens."
)
list_species_tool = Tool(
    name="ListSpecies",
    func=lambda: list_species(),
    description="List all valid species names for CRISPR screens."
)

system_prompt = (
    "You are an expert CRISPR screen data agent. "
    f"The valid options for virus are: {', '.join(virus_options[:10])}... "
    f"The valid options for cell line are: {', '.join(cell_line_options[:10])}... "
    f"The valid options for species are: {', '.join(species_options[:10])}... "
    "When a user asks for a result, always map their input to the closest valid value. "
    "If unsure, use the List tools to clarify the options. "
    "If a user asks a question that is not about CRISPR screen data, or is unsafe, politely refuse to answer."
)

agent = create_react_agent(
    model=llm,
    tools=[gene_tool, screen_tool, condition_tool, list_viruses_tool, list_cell_lines_tool, list_species_tool]
)

# --- Streamlit UI ---
st.title("CRISPR LLM Agent App")
st.markdown("Interactively query CRISPR screen data using natural language and an LLM agent.")

with st.expander("Show controlled vocabularies"):
    st.write(f"**Viruses (first 10):** {', '.join(virus_options[:10])}")
    st.write(f"**Cell lines (first 10):** {', '.join(cell_line_options[:10])}")
    st.write(f"**Species (first 10):** {', '.join(species_options[:10])}")

# --- Chat history state ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [("system", system_prompt)]
if 'last_raw_response' not in st.session_state:
    st.session_state['last_raw_response'] = None

# --- Chat input and controls ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Enter your message", "", key="user_input")
    col1, col2 = st.columns([1, 1])
    submit = col1.form_submit_button("Send")
    reset = col2.form_submit_button("Reset Conversation")

if reset:
    st.session_state['chat_history'] = [("system", system_prompt)]
    st.rerun()

# --- Display chat history ---
st.subheader("Conversation")
for role, msg in st.session_state['chat_history']:
    if role == "human":
        st.markdown(f"**You:** {msg}")
    elif role == "ai":
        st.markdown(f"**Agent:** {msg}")
    elif role == "trace":
        st.markdown(msg)
    # Optionally show system prompt at top only

# --- Show last raw agent response persistently ---
if st.session_state.get('last_raw_response'):
    st.markdown(f"<details><summary>Raw agent response</summary><pre>{st.session_state['last_raw_response']}</pre></details>", unsafe_allow_html=True)

# --- Handle new user input ---
if submit and user_input.strip():
    st.session_state['chat_history'].append(("human", user_input.strip()))
    response = agent.invoke({
        "messages": st.session_state['chat_history'],
        "return_intermediate_steps": True
    })
    # Store the full agent response in session state for persistent display
    st.session_state['last_raw_response'] = str(response)
    # Extract agent reply
    if isinstance(response, dict) and 'messages' in response and response['messages']:
        agent_reply = response['messages'][-1].content
    else:
        agent_reply = str(response)
    st.session_state['chat_history'].append(("ai", agent_reply))
    # Extract and display intermediate steps if present
    if isinstance(response, dict) and 'intermediate_steps' in response and response['intermediate_steps']:
        for step in response['intermediate_steps']:
            tool_name = step.get('tool', 'Unknown Tool')
            tool_input = step.get('tool_input', '')
            tool_output = step.get('output', '')
            trace_msg = f"**Tool Trace:**\n- Tool: `{tool_name}`\n- Input: `{tool_input}`\n- Output: `{tool_output}`"
            st.session_state['chat_history'].append(("trace", trace_msg))
    st.rerun()

