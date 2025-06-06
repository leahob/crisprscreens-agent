import streamlit as st
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from llms import llm
from crispr_data_tools import (
    virus_options, cell_line_options, species_options,
    get_results_by_genes, get_results_by_screens, get_results_by_condition,
    list_viruses, list_cell_lines, list_species
)

# --- Tool Definitions ---
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

