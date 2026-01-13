
import streamlit as st
import logging

# Ensure imports work relative to the script execution or installed package
try:
    from .config import DEFAULT_NUM_GENERATIONS, DEFAULT_NUM_SOLUTIONS, DEFAULT_TAVILY_RESULTS
    from .llm_client import GeminiClient
    from .tavily_client import TavilyClient
    from .solver import EvoGenSolver_Tavily
except ImportError:
    # If running directly from directory
    from config import DEFAULT_NUM_GENERATIONS, DEFAULT_NUM_SOLUTIONS, DEFAULT_TAVILY_RESULTS
    from llm_client import GeminiClient
    from tavily_client import TavilyClient
    from solver import EvoGenSolver_Tavily

# Configure logging to show up in terminal
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="EvoGen AI + Tavily (Refactored)", layout="wide")
st.title("EvoGen AI 🧬 (Refactored)")
st.markdown("Evolutionary Generative AI Framework (v16.0 Logic)")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    st.subheader("Parameters")
    num_generations = st.slider("Generations", 1, 20, DEFAULT_NUM_GENERATIONS)
    num_solutions = st.slider("Solutions per Generation", 3, 10, DEFAULT_NUM_SOLUTIONS)
    tavily_results_per_search = st.slider("Tavily Results (per query)", 1, 10, DEFAULT_TAVILY_RESULTS) 
    st.markdown("---")
    st.info("Full-text web research & Evolutionary algorithm.")

default_problem = """
# Challenge
Propose a new AI solution to drastically improve invoice processing efficiency for SME accounting departments.

# Requirements
- Low implementation cost (under 50k JPY/month).
- Usable without specialized IT knowledge.
- Integration with existing software (freee, etc.) desirable.
"""
problem_statement = st.text_area("Enter Challenge", value=default_problem.strip(), height=200)

if st.button("Start Generation", type="primary"):
    if not gemini_key or not tavily_key:
        st.error("Please provide both API keys.")
    elif not problem_statement.strip():
        st.warning("Please enter a challenge.")
    else:
        # Initialize session state for consistent labels
        if 'output_labels' not in st.session_state:
            st.session_state.output_labels = {"main_label": "Proposal", "details_label": "Details"}

        # Placeholders
        status_placeholder = st.empty()
        team_placeholder = st.empty()
        augmented_placeholder = st.container()
        tavily_placeholder = st.container()
        results_area = st.container()
        final_result_area = st.container()

        try:
            gemini_client = GeminiClient(api_key=gemini_key)
            tavily_client = TavilyClient(api_key=tavily_key)
            solver = EvoGenSolver_Tavily(
                llm_client=gemini_client,
                tavily_client=tavily_client,
                num_solutions_per_generation=num_solutions,
                tavily_results_per_search=tavily_results_per_search
            )
        except Exception as e:
            st.error(f"Initialization Failed: {e}")
            st.stop()

        # Helper to display tavily results
        def display_tavily(results, title):
            with tavily_placeholder.container():
                st.subheader(title)
                if results:
                    for r in results:
                        t = r.get("title", "No Title")
                        u = r.get("url", "")
                        st.markdown(f"- [{t}]({u})")
                else:
                    st.caption("No results found.")
                st.markdown("---")

        with st.spinner("AI is thinking..."):
            # Main Loop
            for event in solver.solve(problem_statement, generations=num_generations):
                
                # --- Log handling ---
                if isinstance(event, str):
                    if event.startswith("ERROR:"):
                        status_placeholder.error(event)
                    elif event.startswith("WARNING:"):
                        status_placeholder.warning(event)
                    elif event.startswith("INFO:") or "---" in event:
                        status_placeholder.info(event)
                    elif event.startswith("LOG:"):
                        status_placeholder.caption(event)
                    else:
                        status_placeholder.write(event)
                
                # --- Data handling ---
                elif isinstance(event, dict):
                    # Tavily Results
                    if "tavily_info_analysis" in event or "tavily_info_solution" in event:
                        tavily_placeholder.empty()
                        a_res = event.get("tavily_info_analysis", [])
                        s_res = event.get("tavily_info_solution", [])
                        if a_res: display_tavily(a_res, "Phase 1: Analysis Research")
                        if s_res: display_tavily(s_res, "Phase 2: Solution Research")

                    # Augmented Problem
                    elif "augmented_problem" in event:
                        with augmented_placeholder.container():
                            st.subheader("🔍 Augmented Problem Statement")
                            with st.expander("Show Details"):
                                st.markdown(event["augmented_problem"])
                            st.markdown("---")

                    # Agent Team
                    elif "agent_team" in event or "agent_team_updated" in event:
                        key = "agent_team_updated" if "agent_team_updated" in event else "agent_team"
                        team = event[key]
                        
                        if "output_labels" in team:
                            st.session_state.output_labels = team["output_labels"]
                        
                        lb = st.session_state.output_labels
                        
                        with team_placeholder.container():
                            st.subheader("🤖 Agent Swarm")
                            st.markdown(f"**Labels:** `{lb.get('main_label')}` / `{lb.get('details_label')}`")
                            
                            is_updated = (key == "agent_team_updated")
                            with st.expander("Show Team Details", expanded=is_updated):
                                st.markdown("##### Solver Agents")
                                for i, ag in enumerate(team.get("solver_agents", [])):
                                    st.markdown(f"**{i+1}. {ag.get('role', 'N/A')}**")
                                    st.caption(f"Instr: {ag.get('instructions')}")
                                    insights = ag.get("agent_research_insights")
                                    if insights:
                                        st.markdown("**Insights:**")
                                        for ins in insights:
                                            st.markdown(f"  - {ins}")
                                    st.markdown("---")
                                
                                st.markdown("##### Evaluators")
                                for i, ev in enumerate(team.get("evaluators", [])):
                                    st.markdown(f"**{i+1}. {ev.get('role', 'N/A')}**")
                                    st.caption(f"Guide: {ev.get('evaluation_guideline')}")

                    # Generation Results
                    elif "generation" in event:
                        gen_data = event
                        current_labels = st.session_state.output_labels
                        with results_area.container():
                            st.subheader(f"Generation {gen_data['generation']} Results")
                            with st.container(border=True):
                                if not gen_data.get('results'):
                                    st.warning("No valid proposals.")
                                    continue
                                
                                for item in gen_data.get('results', []):
                                    sol = item.get('solution', {})
                                    eva = item.get('evaluation', {})
                                    score = eva.get('total_score', 0)
                                    
                                    st.markdown(f"**{current_labels.get('main_label')}:** {sol.get('proposal_main', 'N/A')} (Score: {score})")
                                    st.caption(f"**{current_labels.get('details_label')}:** {sol.get('proposal_details', 'N/A')}")
                                    st.markdown("---")

            # Final Results Logic (Similar to original)
            if solver.history:
                all_sols = [
                    item for gen in solver.history
                    for item in gen.get("results", [])
                    if item.get("evaluation") and "total_score" in item["evaluation"]
                ]
                
                if all_sols:
                    top_5 = sorted(all_sols, key=lambda x: x["evaluation"]["total_score"], reverse=True)[:5]
                    labels = st.session_state.output_labels
                    
                    st.balloons()
                    with final_result_area:
                        st.success("🏆 Top 5 Solutions")
                        for i, item in enumerate(top_5):
                            sol = item.get('solution', {})
                            eva = item.get('evaluation', {})
                            
                            st.header(f"🏅 Rank {i + 1} (Score: {eva.get('total_score')})")
                            st.info(f"**{labels.get('main_label')}**\n\n{sol.get('proposal_main')}")
                            st.write(f"**{labels.get('details_label')}**\n\n{sol.get('proposal_details')}")
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                st.success("**Strengths**")
                                st.write(eva.get('strengths'))
                            with c2:
                                st.warning("**Weaknesses**")
                                st.write(eva.get('weaknesses'))
                            st.markdown("---")
                else:
                    status_placeholder.warning("Process finished but no solutions found.")

