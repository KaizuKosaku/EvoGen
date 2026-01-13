
import random
from typing import List, Dict, Any, Generator, Optional
import logging

from .llm_client import LLMClient
from .tavily_client import TavilyClient
from .prompts import PromptManager

# Logger customization can be done in main app, here we just use what we get
logger = logging.getLogger(__name__)

class EvoGenSolver:
    """Base EvoGenSolver logic."""
    def __init__(self, llm_client: LLMClient, num_solutions_per_generation: int = 10):
        self.client = llm_client
        self.num_solutions = num_solutions_per_generation 
        self.prompter = PromptManager()
        self.history = []

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        return self.client.call(prompt) 

    def _generate_agent_personas(self, problem_statement: str) -> Dict:
        prompt = self.prompter.get_agent_personas_prompt(problem_statement)
        return self._call_llm(prompt)

    def _generate_initial_solutions(self, problem_statement: str, context: List[Dict]) -> Generator[str, None, List[Dict[str, str]]]:
        """
        Generates initial solutions. Yields log messages.
        """
        initial_agent_list = context 
        if not isinstance(initial_agent_list, list) or len(initial_agent_list) == 0:
            yield "WARNING: [EvoGenSolver] Agent list is invalid."
            return []
        
        num_initial_agents = len(initial_agent_list)
        yield f"INFO: 💡 {num_initial_agents} individual agents are generating initial proposals..."
        
        all_solutions = []
        for i, agent_context in enumerate(initial_agent_list):
            yield f"LOG:   - Agent {i+1}/{num_initial_agents} ({agent_context.get('role', 'N/A')}) is working..."
            
            prompt = self.prompter.get_initial_generation_prompt(
                problem_statement, 
                1, 
                agent_context 
            )
            response = self._call_llm(prompt) 
            
            if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list) and len(response["solutions"]) > 0:
                all_solutions.append(response["solutions"][0])
            else:
                yield f"WARNING: [EvoGenSolver] Agent {i+1} returned invalid format. Debug info available in logs."
                logger.warning(f"Agent {i+1} invalid response: {response}")
                
        return all_solutions

    def _evaluate_solutions(self, solutions: List[Dict[str, str]], problem_statement: str, context: Dict) -> Generator[str | List[Dict], None, None]:
        evaluator_agent_list = context
        if not isinstance(evaluator_agent_list, list) or len(evaluator_agent_list) == 0:
            yield "ERROR: [EvoGenSolver] Evaluator list is invalid."
            yield []
            return

        evaluated_solutions = []
        if not solutions:
            yield []
            return

        num_evaluators = len(evaluator_agent_list)

        for i, solution in enumerate(solutions):
            if not isinstance(solution, dict) or "proposal_main" not in solution:
                yield f"LOG:   - Evaluation skipped: Invalid proposal format."
                continue
            
            yield f"LOG:   - Evaluating {i+1}/{len(solutions)}: {solution.get('proposal_main', 'Unknown')} ({num_evaluators} evaluators)"

            individual_evaluations = []
            
            for j, eval_context in enumerate(evaluator_agent_list):
                # yield f"LOG:     - Evaluator {j+1}/{num_evaluators} ({eval_context.get('role', 'N/A')}) evaluating..."
                
                prompt = self.prompter.get_evaluation_prompt(solution, problem_statement, eval_context)
                evaluation = self._call_llm(prompt)
                
                if isinstance(evaluation, dict) and "total_score" in evaluation and "error" not in evaluation:
                    individual_evaluations.append(evaluation)
                else:
                    logger.warning(f"Evaluator {j+1} invalid response: {evaluation}")

            if not individual_evaluations:
                yield f"WARNING: [EvoGenSolver] No valid evaluation for '{solution.get('proposal_main', 'N/A')}'."
                continue

            total_score_sum = sum(e.get('total_score', 0) for e in individual_evaluations)
            aggregated_score = round(total_score_sum / len(individual_evaluations))
            
            agg_strengths = "\n---\n".join([f"Evaluator {k+1} ({e.get('role', 'N/A')}):\n{e.get('strengths', 'N/A')}" for k, e in enumerate(individual_evaluations)])
            agg_weaknesses = "\n---\n".join([f"Evaluator {k+1} ({e.get('role', 'N/A')}):\n{e.get('weaknesses', 'N/A')}" for k, e in enumerate(individual_evaluations)])
            agg_comment = "\n---\n".join([f"Evaluator {k+1} ({e.get('role', 'N/A')}):\n{e.get('overall_comment', 'N/A')}" for k, e in enumerate(individual_evaluations)])

            aggregated_evaluation = {
                "total_score": aggregated_score,
                "strengths": agg_strengths,
                "weaknesses": agg_weaknesses,
                "overall_comment": agg_comment,
                "individual_evals": individual_evaluations 
            }
            
            evaluated_solutions.append({"solution": solution, "evaluation": aggregated_evaluation})
            
        # Sort by score
        evaluated_solutions.sort(key=lambda x: x.get("evaluation", {}).get("total_score", 0), reverse=True)
        yield evaluated_solutions

    def _generate_next_generation(self, evaluated_solutions: List[Dict], problem_statement: str, context: List[Dict]) -> Generator[str, None, List[Dict[str, str]]]:
        solver_agent_list = context 
        if not isinstance(solver_agent_list, list) or len(solver_agent_list) == 0:
            yield "WARNING: [EvoGenSolver] Solver agent list is invalid."
            return []

        num_elites = max(1, int(len(evaluated_solutions) * 0.4))
        elite_solutions = evaluated_solutions[:num_elites]
        failed_solutions = evaluated_solutions[num_elites:]

        yield f"INFO: 🚀 Selecting {self.num_solutions} agents for next generation..."

        new_solutions = []
        for i in range(self.num_solutions):
            
            if random.random() < 0.20:
                # Mutation
                yield f"LOG:   - ⚡ (Mutation) Agent {i+1}/{self.num_solutions} is innovating..."
                
                existing_roles = [a.get('role', 'N/A') for a in solver_agent_list]
                
                prompt = self.prompter.get_revolutionary_generation_prompt(
                    problem_statement, 
                    1, 
                    existing_roles 
                )
            else:
                # Evolution
                selected_agent_context = random.choice(solver_agent_list) 
                yield f"LOG:   - 🧬 (Evolution) Agent {i+1}/{self.num_solutions} ({selected_agent_context.get('role', 'N/A')}) is refining..."
                
                prompt = self.prompter.get_next_generation_prompt(
                    elite_solutions, 
                    failed_solutions, 
                    problem_statement, 
                    1, 
                    selected_agent_context
                )
            
            response = self._call_llm(prompt) 
            
            if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list) and len(response["solutions"]) > 0:
                new_solutions.append(response["solutions"][0])
            else:
                yield f"WARNING: [EvoGenSolver] Agent {i+1} returned invalid format."
                logger.warning(f"Agent {i+1} invalid response: {response}")

        return new_solutions

    def solve_internal(self, problem_statement: str, agent_personas: Dict, generations: int) -> Generator[str | Dict, None, None]:
        """
        Execution cycle: Generation -> Evaluation -> Evolution.
        """
        if self.history: 
             pass
        else:
             self.history = []
             
        yield "\n--- 💡 Generation 0: Generating initial proposals (10) ... ---"
        # Since _generate_initial_solutions is a generator now (to yield logs), we need to iterate it
        solutions = []
        gen = self._generate_initial_solutions(problem_statement, agent_personas["solver_agents"])
        # If it was a normal function we'd just call it, but it yields logs.
        # We need to capture the return value (the solutions list) from the generator.
        # Python generators return value is in StopIteration exception or we can use `yield from` if we are in a generator.
        # But `yield from` returns the value.
        solutions = yield from gen
        
        if not solutions:
             yield "ERROR: Failed to generate initial proposals. Ending process."
             return

        yield "--- 🧐 Evaluating proposals ... ---"
        # _evaluate_solutions yields logs then the final list
        evaluated_solutions = []
        eval_gen = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluators"])
        
        for item in eval_gen:
            if isinstance(item, str):
                yield item
            else:
                evaluated_solutions = item

        if not evaluated_solutions:
             yield "ERROR: Failed to evaluate proposals. Ending process."
             return

        self.history.append({"generation": 0, "results": evaluated_solutions})
        yield self.history[-1] # Yield generation results for UI to display

        # Evolution cycles
        for i in range(1, generations):
            yield f"\n--- 🚀 Generation {i}: Evolving to next stage ... ---"
            previous_generation_results = self.history[-1]["results"]
            
            if not previous_generation_results:
                yield f"ERROR: No valid results from Generation {i-1}. Stopping."
                break
            
            gen_next = self._generate_next_generation(previous_generation_results, problem_statement, agent_personas["solver_agents"])
            solutions = yield from gen_next

            if not solutions:
                yield f"ERROR: Failed to generate proposals for Generation {i}. Stopping."
                break

            yield f"--- 🧐 Evaluating Generation {i} ... ---"
            evaluated_solutions_next = []
            eval_gen_next = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluators"])
            
            for item in eval_gen_next:
                if isinstance(item, str):
                    yield item
                else:
                    evaluated_solutions_next = item

            if not evaluated_solutions_next:
                 yield f"ERROR: Failed to evaluate Generation {i}. Stopping."
                 break

            self.history.append({"generation": i, "results": evaluated_solutions_next})
            yield self.history[-1]

        yield "\n--- ✅ Evolution Process Complete ---"


class EvoGenSolver_Tavily(EvoGenSolver):
    """
    Extended Solver with Tavily Research capabilities.
    """
    def __init__(self, llm_client: LLMClient, tavily_client: TavilyClient, num_solutions_per_generation: int = 10, tavily_results_per_search: int = 5):
        super().__init__(llm_client, num_solutions_per_generation)
        self.tavily = tavily_client
        self.tavily_results_per_agent_query = max(1, tavily_results_per_search // 2) 
        self.tavily_results_for_augmentation = tavily_results_per_search 

    def _format_raw_content_for_llm(self, results: List[Dict[str, Any]], context_tag: str, max_items: int = 3, truncate_chars: int = 4000) -> str:
        content_blocks = []
        if not results:
            return f"({context_tag}: No content found.)\n"
        
        for i, r in enumerate(results[:max_items]): 
            url = r.get("url", "Unknown URL")
            title = r.get("title", "No Title")
            raw_content = r.get("raw_content")
            
            content_blocks.append(f"--- START {context_tag} SOURCE {i+1} ({title}) ---\n")
            content_blocks.append(f"URL: {url}\n")
            
            if raw_content:
                truncated_content = raw_content[:truncate_chars] 
                content_blocks.append(f"CONTENT (first {truncate_chars} chars):\n{truncated_content}\n")
            else:
                snippet = r.get("snippet", "") or r.get("description", "")
                content_blocks.append(f"CONTENT: (No raw content available, using snippet)\n{snippet}\n")
            
            content_blocks.append(f"--- END {context_tag} SOURCE {i+1} ---\n")
        
        return "\n".join(content_blocks)

    def _summarize_multi_phase_results_with_llm(self, problem_statement: str, analysis_results: List[Dict[str, Any]], solution_results: List[Dict[str, Any]]) -> str:
        if not analysis_results and not solution_results:
            return problem_statement

        analysis_content_text = self._format_raw_content_for_llm(analysis_results, "ANALYSIS CONTENT", max_items=3)
        solution_content_text = self._format_raw_content_for_llm(solution_results, "SOLUTION CONTENT", max_items=3)

        prompt = f"""
        # 役割
        あなたは、第一線のリサーチ戦略家です。あなたの仕事は、大量の調査資料（Webページの全文）を読み解き、
        単なる要約ではなく、「戦略的な洞察」を抽出することです。

        # 元の課題
        {problem_statement}

        # 調査資料 1: 現状・背景分析 (Webページ全文)
        {analysis_content_text if analysis_content_text else "なし"}

        # 調査資料 2: 解決策の事例・技術 (Webページ全文)
        {solution_content_text if solution_content_text else "なし"}

        # タスク
        あなたは今、上記の「調査資料1」と「調査資料2」の*全文*（またはその冒頭）を読み終えました。
        これらの詳細な情報に基づき、元の課題をより深く、より具体的に補強するための分析を行ってください。

        # 出力形式 (JSON)
        分析結果を以下のJSON形式で出力してください。
        {{
          "summary_analysis": "「調査資料1（現状・背景）」を深く分析した*戦略的洞察*。(1〜3文)",
          "summary_solution": "「調査資料2（解決策・事例）」から抽出した*重要な傾向*。(1〜3文)",
          "key_points": [
            "その他、考慮するべきと思われる観点1",
            "その他、考慮するべきと思われる観点2"
          ],
          "top_sources": [
             {{ "title": "...", "url": "..." }}
          ]
        }}
        """
        
        llm_ret = self._call_llm(prompt) 
        
        if isinstance(llm_ret, dict) and any(k in llm_ret for k in ["summary_analysis", "summary_solution", "key_points"]):
            try:
                summary_analysis_text = llm_ret.get("summary_analysis", "現状分析の要約なし")
                summary_solution_text = llm_ret.get("summary_solution", "解決策事例の要約なし")
                kp = llm_ret.get("key_points", [])
                top = llm_ret.get("top_sources", [])
                top_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in top]) if isinstance(top, list) else ""
                
                composed = f"""
## Tavily Research Summary (LLM Analysis)
### Current Status / Background
{summary_analysis_text}
### Solutions / Case Studies
{summary_solution_text}

### Key Points
""" + "\n".join([f"- {p}" for p in kp]) + "\n\n" + \
"### Top Sources\n" + top_text + "\n\n" + \
"--- (Original Problem) ---\n" + problem_statement
                
                return composed
            except Exception:
                pass 

        # Fallback
        return problem_statement

    def _run_agent_specific_research(self, problem_statement: str, solver_agents: List[Dict]) -> Generator[str, None, List[Dict]]:
        if not solver_agents:
            yield "WARNING: No solver agents defined. Skipping individual research."
            return []
            
        yield f"--- 🤖 Generating search queries for {len(solver_agents)} agents (Batch)... ---"
        
        all_queries_prompt = self.prompter.get_all_agent_queries_prompt(problem_statement, solver_agents)
        all_queries_response = self._call_llm(all_queries_prompt)
        
        all_queries_dict = {}
        if isinstance(all_queries_response, dict) and "agent_queries" in all_queries_response:
            all_queries_dict = all_queries_response["agent_queries"]
        else:
            yield f"WARNING: Failed to batch generate queries. Skipping individual research."
            return solver_agents 
            
        yield f"--- ✔️ Queries generated. Starting deep research for {len(solver_agents)} agents... ---"
        
        updated_agents = []
        num_agents = len(solver_agents)

        for i, agent_context in enumerate(solver_agents):
            role = agent_context.get("role", "Unknown Role")
            instructions = agent_context.get("instructions", "")
            
            queries = all_queries_dict.get(role, [])
            
            if not queries:
                yield f"LOG:   - {i+1}/{num_agents}: '{role}' has no queries. Skipping."
                updated_agents.append(agent_context)
                continue

            yield f"LOG:   - {i+1}/{num_agents}: '{role}' researching (Queries: {', '.join(queries)})..."
            agent_search_results = []
            for q in queries:
                if not q.strip(): continue
                tavily_resp = self.tavily.search(q, num_results=self.tavily_results_per_agent_query)
                if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                    agent_search_results.extend(tavily_resp["results"])
                elif isinstance(tavily_resp, dict) and "error" in tavily_resp:
                     yield f"WARNING: Tavily error for '{q}': {tavily_resp['error']}"

            if not agent_search_results:
                yield f"LOG:   - '{role}' found no results."
                updated_agents.append(agent_context) 
                continue

            raw_content_text = self._format_raw_content_for_llm(
                agent_search_results,
                f"AGENT {i+1} RESEARCH",
                max_items=self.tavily_results_per_agent_query * 2, 
                truncate_chars=3000 
            )

            # yield f"LOG:   - {i+1}/{num_agents}: Analyzing research for '{role}'..."
            analysis_prompt = self.prompter.get_agent_specific_analysis_prompt(
                problem_statement,
                role,
                instructions,
                raw_content_text
            )
            analysis_response = self._call_llm(analysis_prompt)
            
            insights = []
            if isinstance(analysis_response, dict) and "key_insights" in analysis_response and isinstance(analysis_response["key_insights"], list):
                insights = analysis_response["key_insights"]
            
            agent_context["agent_research_insights"] = insights
            updated_agents.append(agent_context)
            yield f"LOG:   - {i+1}/{num_agents}: '{role}' gained {len(insights)} insights."

        yield f"--- ✔️ Individual agent research complete ---"
        return updated_agents 

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        self.history = []

        # --- Step 1: Problem Augmentation ---
        yield "--- 💡 generating research queries for problem augmentation... ---"
        prompt = self.prompter.get_tavily_multi_phase_query_prompt(problem_statement)
        query_response = self._call_llm(prompt)

        augmented_problem = problem_statement 

        if not isinstance(query_response, dict) or ("analysis_queries" not in query_response and "solution_queries" not in query_response):
            yield f"ERROR: Failed to generate augmented queries."
        else:
            analysis_queries = query_response.get("analysis_queries", [])
            solution_queries = query_response.get("solution_queries", [])
            
            yield f"--- ✔️ Queries generated ---"
            
            analysis_results_list = []
            solution_results_list = []
            
            if analysis_queries:
                yield "--- 🌐 Phase 1: Context Analysis (Full Text) ... ---"
                for q in analysis_queries:
                    if not q.strip(): continue
                    yield f"LOG:   - Researching (Analysis): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_for_augmentation)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        analysis_results_list.extend(tavily_resp["results"])
            
            if solution_queries:
                yield "--- 🌐 Phase 2: Solution Research (Full Text) ... ---"
                for q in solution_queries:
                    if not q.strip(): continue
                    yield f"LOG:   - Researching (Solution): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_for_augmentation)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        solution_results_list.extend(tavily_resp["results"])

            yield {"tavily_info_analysis": analysis_results_list, "tavily_info_solution": solution_results_list}

            yield "--- ✍️ Analyzing full web content with LLM... ---"
            try:
                augmented_problem = self._summarize_multi_phase_results_with_llm(
                    problem_statement, 
                    analysis_results_list, 
                    solution_results_list
                )
            except Exception as e:
                yield f"WARNING: Augmented analysis failed: {e}"
        
        yield {"augmented_problem": augmented_problem}

        # --- Step 2: Swarm Formation ---
        yield "--- 🧠 Forming Agent Swarm based on augmented problem... ---"
        agent_personas = self._generate_agent_personas(augmented_problem) 

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["solver_agents", "evaluators", "output_labels"]):
            yield "ERROR: Failed to form team."
            return

        yield f"--- ✔️ Team formed ---"
        yield {"agent_team": agent_personas} 

        # --- Step 3: Agent Specific Research ---
        updated_agents_list = yield from self._run_agent_specific_research(
            augmented_problem, 
            agent_personas["solver_agents"]
        )

        if not updated_agents_list:
             updated_agents_list = agent_personas["solver_agents"] 

        agent_personas["solver_agents"] = updated_agents_list
        yield {"agent_team_updated": agent_personas}

        # --- Step 4: Execution Cycle ---
        yield from self.solve_internal(augmented_problem, agent_personas, generations)
