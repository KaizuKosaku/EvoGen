import streamlit as st
import os
import json
import abc
from typing import List, Dict, Any, Generator

# --- LLMライブラリのインポート ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- 1. LLMとの通信を担当する部分 ---
class LLMClient(abc.ABC):
    """LLMクライアントの基本となる設計図"""
    @abc.abstractmethod
    def call(self, prompt: str) -> Dict[str, Any]:
        """LLMを呼び出すための命令"""
        pass

class GeminiClient(LLMClient):
    """Google Gemini API を使用するためのクラス"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if genai is None:
            raise ImportError("`google-generativeai`ライブラリが未インストールです。コマンドプロンプトで `pip install google-generativeai` を実行してください。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )

    def call(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"[GeminiClient Error] API呼び出し中にエラーが発生しました: {e}")
            return {"error": str(e)}

# --- 2. AIへの指示書（プロンプト）を作成する部分 ---
class PromptManager:
    """AIへの指示書（プロンプト）を管理するクラス"""

    def get_agent_personas_prompt(self, problem_statement: str) -> str:
        """AIエージェントチームの役割を決めさせるための指示書"""
        return f"""
        # 役割
        あなたは、非常に複雑な課題を解決するために、AIエージェントからなるドリームチームを編成する「マスタープランナー」です。
        # タスク
        以下の「課題」を深く分析し、この課題を解決するために最も効果的な思考チームを編成してください。
        チームは以下の3体のAIエージェントで構成されます。それぞれのエージェントについて、その役割（ペルソナ）と具体的な行動指示を定義してください。

        1. **initial_generator (初期アイデア生成担当):**
            - **role:** どのような専門性や性格を持つべきか？
            - **instructions:** どのような観点から、どのように多様なアイデアを出すべきか？

        2. **evaluator (評価担当):**
            - **role:** どのような視点からアイデアを評価すべきか？
            - **criteria:** この課題に特化した評価基準を3つ定義し、重要度に応じて合計100点になるように配点してください。

        3. **synthesizer (進化・統合担当):**
            - **role:** どのようにしてアイデアをより優れたものに進化させるべきか？
            - **instructions:** 高評価案と低評価案をどのように分析し、次世代のアイデアを生成すべきか具体的な指示を与えてください。

        # 課題
        {problem_statement}

        # 出力形式
        あなたの回答は、必ず以下の構造を持つJSONオブジェクトでなければなりません。
        {{
          "initial_generator": {{"role": "...", "instructions": "..."}},
          "evaluator": {{"role": "...", "criteria": [{{"criterion": "...", "weight": 10}}]}},
          "synthesizer": {{"role": "...", "instructions": "..."}}
        }}
        """

    def get_initial_generation_prompt(self, problem_statement: str, num_solutions: int, context: Dict[str, str]) -> str:
        """最初のアイデアを生成させるための指示書"""
        return f"""
        # 役割: {context.get('role', 'あなたは一流のイノベーターです。')}
        # 指示: {context.get('instructions', f'以下の課題に対し、互いに全く異なるアプローチからの解決策を{num_solutions}個提案してください。')}
        # 課題文: {problem_statement}
        # 出力形式: 各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。
        {{ "solutions": [ {{ "name": "解決策1", "summary": "概要1", "specific_method": "具体的方法1" }} ] }}
        """

    def get_evaluation_prompt(self, solution: Dict[str, str], problem_statement: str, context: Dict[str, Any]) -> str:
        """アイデアを評価させるための指示書"""
        criteria_text = []
        scores_json_structure = []
        if "criteria" in context and isinstance(context["criteria"], list):
            for item in context["criteria"]:
                criterion = item.get("criterion", "不明な基準")
                weight = item.get("weight", 0)
                criteria_text.append(f"- {criterion}: {weight}点")
                scores_json_structure.append(f'"{criterion}": 点数(整数)')

        criteria_prompt_part = "\\n".join(criteria_text)
        scores_json_prompt_part = f"{{ {', '.join(scores_json_structure)} }}"

        return f"""
        # 役割: {context.get('role', 'あなたは客観的で厳しい批評家です。')}
        # タスク: 提示された課題に対し、解決案を評価基準に基づいて厳密に評価してください。
        # 課題文: {problem_statement}
        # 評価対象の解決案:
        - 名称: {solution.get('name', '名称不明')}
        - 概要: {solution.get('summary', '概要なし')}
        # 評価基準:
        {criteria_prompt_part}
        # 出力形式: 評価結果を必ず以下のJSON形式で出力してください。
        {{
          "total_score": 合計点(整数),
          "scores": {scores_json_prompt_part},
          "strengths": "この解決案が優れている点",
          "weaknesses": "この解決案の懸念点や改善が必要な点",
          "overall_comment": "評価の総括"
        }}
        """

    def get_next_generation_prompt(self, elite_solutions: List[Dict], failed_solutions: List[Dict], problem_statement: str, num_solutions: int, context: Dict[str, str]) -> str:
        """次の世代のアイデアを進化・生成させるための指示書"""
        elite_text = "\\n".join([f"- {s['solution'].get('name', 'N/A')} (スコア: {s['evaluation'].get('total_score', 0)})" for s in elite_solutions])
        failed_text = "\\n".join([f"- {s['solution'].get('name', 'N/A')} (弱点: {s['evaluation'].get('weaknesses', 'N/A')})" for s in failed_solutions])

        return f"""
        # 役割: {context.get('role', 'あなたは優れた戦略家であり編集者です。')}
        # タスク: 前世代の分析に基づき、次世代の新しい解決策を{num_solutions}個生成してください。
        # 分析対象1：高評価だった解決案（優れた遺伝子）: 
        {elite_text}
        # 分析対象2：低評価だった解決案（学ぶべき教訓）: 
        {failed_text}
        # 新しい解決策の生成指示: {context.get('instructions', '高評価案の良い点を組み合わせ、低評価案の失敗から学び、新しい解決策を生成してください。')}
        
        # 出力形式: 各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。あなたの回答は、必ず以下の構造を持つJSONオブジェクトでなければなりません。
        {{ "solutions": [ {{ "name": "新しい解決策1", "summary": "概要1", "specific_method": "具体的方法1" }} ] }}
        """

# --- 3. 全体の処理を管理するメイン部分 ---
class EvoGenSolver:
    """EvoGenフレームワークの処理全体を管理するクラス"""
    def __init__(self, llm_client: LLMClient, num_solutions_per_generation: int = 5):
        self.client = llm_client
        self.num_solutions = num_solutions_per_generation
        self.prompter = PromptManager()
        self.history = []

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        return self.client.call(prompt)

    def _generate_agent_personas(self, problem_statement: str) -> Dict:
        prompt = self.prompter.get_agent_personas_prompt(problem_statement)
        return self._call_llm(prompt)

    def _generate_initial_solutions(self, problem_statement: str, context: Dict) -> List[Dict[str, str]]:
        prompt = self.prompter.get_initial_generation_prompt(problem_statement, self.num_solutions, context)
        response = self._call_llm(prompt)
        return response.get("solutions", []) if isinstance(response, dict) else []

    def _evaluate_solutions(self, solutions: List[Dict[str, str]], problem_statement: str, context: Dict) -> Generator[str | List[Dict], None, None]:
        evaluated_solutions = []
        if not solutions:
            yield []
            return

        for i, solution in enumerate(solutions):
            yield f"  - 評価中 {i+1}/{len(solutions)}: {solution.get('name', '名称不明')}"
            prompt = self.prompter.get_evaluation_prompt(solution, problem_statement, context)
            evaluation = self._call_llm(prompt)
            if evaluation and "error" not in evaluation:
                evaluated_solutions.append({"solution": solution, "evaluation": evaluation})
        
        evaluated_solutions.sort(key=lambda x: x.get("evaluation", {}).get("total_score", 0), reverse=True)
        yield evaluated_solutions

    def _generate_next_generation(self, evaluated_solutions: List[Dict], problem_statement: str, context: Dict) -> List[Dict[str, str]]:
        num_elites = max(1, int(len(evaluated_solutions) * 0.4))
        elite_solutions = evaluated_solutions[:num_elites]
        failed_solutions = evaluated_solutions[num_elites:]
        prompt = self.prompter.get_next_generation_prompt(elite_solutions, failed_solutions, problem_statement, self.num_solutions, context)
        response = self._call_llm(prompt)
        return response.get("solutions", []) if isinstance(response, dict) else []

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, Dict]:
        self.history = []

        # STEP 1: AIエージェントチームの編成
        yield "--- 🧠 課題を分析し、最適なAIエージェントチームを編成中... ---"
        agent_personas = self._generate_agent_personas(problem_statement)

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["initial_generator", "evaluator", "synthesizer"]):
            yield "エラー: チーム編成に失敗しました。処理を中断します。"
            yield f"**デバッグ情報:** AIからの応答が不正です。APIキーが正しいか確認してください。\n```\n{agent_personas}\n```"
            return {"best_solution": None}

        yield f"--- ✔️ チーム編成完了 ---"
        # --- ▼▼▼ 修正箇所 ▼▼▼ ---
        # チーム編成の詳細をUIに送信
        yield {"agent_team": agent_personas}
        # --- ▲▲▲ 修正箇所 ▲▲▲ ---

        # STEP 2: 最初のアイデア生成と評価
        yield "\n--- 💡 Generation 0: 最初のアイデアを生成中... ---"
        solutions = self._generate_initial_solutions(problem_statement, agent_personas["initial_generator"])
        
        yield "--- 🧐 アイデアを評価中... ---"
        eval_generator = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluator"])
        evaluated_solutions = []
        for item in eval_generator:
            if isinstance(item, str):
                yield item
            else:
                evaluated_solutions = item
        
        self.history.append({"generation": 0, "results": evaluated_solutions})
        yield self.history[-1]

        # STEP 3: 世代の進化
        for i in range(1, generations):
            yield f"\n--- 🚀 Generation {i}: 次のアイデアへ進化中... ---"
            previous_generation_results = self.history[-1]["results"]
            
            solutions = self._generate_next_generation(previous_generation_results, problem_statement, agent_personas["synthesizer"])
            
            if not solutions:
                yield f"エラー: Generation {i} の解決策生成に失敗しました。AIが適切な応答を返さなかった可能性があります。処理を終了します。"
                break

            yield f"--- 🧐 Generation {i} のアイデアを評価中... ---"
            eval_generator_next = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluator"])
            evaluated_solutions_next = []
            for item in eval_generator_next:
                if isinstance(item, str):
                    yield item
                else:
                    evaluated_solutions_next = item
            
            self.history.append({"generation": i, "results": evaluated_solutions_next})
            yield self.history[-1]

        # STEP 4: 最終的な最適解の決定
        if not self.history or not any(gen.get("results") for gen in self.history):
             yield "エラー: 評価済みの解決策が一つもありませんでした。"
             return {"best_solution": None}

        best_solution_overall = max(
            (item for gen in self.history for item in gen.get("results", []) if item.get("evaluation")),
            key=lambda x: x.get("evaluation", {}).get("total_score", 0),
            default=None
        )
        
        yield "\n--- ✅ 進化プロセス完了 ---"
        return {"best_solution": best_solution_overall}


# --- 4. 画面表示（UI）を担当する部分 ---

st.set_page_config(page_title="EvoGen AI", layout="wide")

st.title("EvoGen AI: 進化的問題解決フレームワーク 🧬")
st.markdown("AIエージェントチームが、与えられた課題に対して**進化的**に解決策を探求するプロセスを可視化します。")

# --- サイドバーの設定画面 ---
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Google Gemini API Key", type="password", help="APIキーが保存されることはありません。")
    
    st.subheader("パラメータ")
    num_generations = st.slider("世代数", 1, 5, 2, help="解決策を進化させる回数です。")
    num_solutions = st.slider("世代ごとの解決策の数", 3, 10, 4, help="1世代あたりに生成・評価する解決策の数です。")

    st.markdown("---")
    st.info("このアプリは、課題に応じて専門家チームを自動編成し、アイデアの生成→評価→淘汰・進化のサイクルを繰り返すことで、より良い解決策を見つけ出します。")


# --- メイン画面 ---
default_problem = """
# 課題
都市部におけるカラスによるゴミ集積所の被害が深刻化している。
カラスを傷つけることなく、かつ低コストで持続可能な方法で、
ゴミが荒らされるのを防ぐための画期的な解決策を提案せよ。

# 要件・制約条件
- カラスや他の動物に危害を加えないこと。
- 住民が簡単に利用・管理できること。
- 導入および維持コストが低いこと。
- 景観を大きく損なわないこと。
"""
problem_statement = st.text_area("解決したい課題を入力してください", value=default_problem, height=250)

if st.button("解決策の生成を開始", type="primary"):
    if not api_key:
        st.error("サイドバーでGoogle Gemini APIキーを入力してください。")
    elif not problem_statement.strip():
        st.warning("課題を入力してください。")
    else:
        status_placeholder = st.empty()
        # --- ▼▼▼ 修正箇所 ▼▼▼ ---
        team_placeholder = st.empty() # チーム情報を表示するための場所を確保
        # --- ▲▲▲ 修正箇所 ▲▲▲ ---
        results_area = st.container()
        final_result_placeholder = st.empty()
        
        final_result = {}

        with st.spinner("🌀 AIが思考中です...しばらくお待ちください..."):
            solver = EvoGenSolver(llm_client=GeminiClient(api_key=api_key), num_solutions_per_generation=num_solutions)
            for result in solver.solve(problem_statement, generations=num_generations):
                if isinstance(result, str):
                    status_placeholder.info(result)
                
                # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                # チーム情報を受け取って表示する処理
                elif isinstance(result, dict) and "agent_team" in result:
                    with team_placeholder.container():
                        st.subheader("🤖 編成されたAIエージェントチーム")
                        team = result["agent_team"]
                        with st.expander("チームの詳細を表示する"):
                            # アイデア生成担当
                            gen = team.get("initial_generator", {})
                            st.markdown("##### 💡 アイデア生成担当")
                            st.markdown(f"**役割:** {gen.get('role', '未定義')}")
                            st.markdown(f"**指示:** {gen.get('instructions', '未定義')}")
                            
                            # 評価担当
                            eva = team.get("evaluator", {})
                            st.markdown("##### 🧐 評価担当")
                            st.markdown(f"**役割:** {eva.get('role', '未定義')}")
                            criteria_list = eva.get('criteria', [])
                            criteria_md = ""
                            if criteria_list:
                                for c in criteria_list:
                                    criteria_md += f"- **{c.get('criterion', '項目名なし')}:** {c.get('weight', 0)}点\n"
                            st.markdown(f"**評価基準:**\n{criteria_md or '未定義'}")

                            # 進化・統合担当
                            syn = team.get("synthesizer", {})
                            st.markdown("##### 🧬 進化・統合担当")
                            st.markdown(f"**役割:** {syn.get('role', '未定義')}")
                            st.markdown(f"**指示:** {syn.get('instructions', '未定義')}")
                # --- ▲▲▲ 修正箇所 ▲▲▲ ---

                elif isinstance(result, dict) and "generation" in result:
                    gen_data = result
                    with results_area.container():
                        st.subheader(f"第 {gen_data['generation']} 世代の結果")
                        with st.expander(f"世代 {gen_data['generation']} の詳細を表示", expanded=True):
                            for item in gen_data.get('results', []):
                                sol = item.get('solution', {})
                                eva = item.get('evaluation', {})
                                score = eva.get('total_score', 0)
                                st.markdown(f"**{sol.get('name', 'N/A')}** (スコア: {score})")
                                with st.container(border=True):
                                    st.markdown(f"**概要:** {sol.get('summary', 'N/A')}")
                                    st.markdown(f"**長所:** {eva.get('strengths', 'N/A')}")
                                    st.markdown(f"**懸念点:** {eva.get('weaknesses', 'N/A')}")
                
                elif isinstance(result, dict) and "best_solution" in result:
                    final_result = result
                    break
        
        if final_result and final_result.get("best_solution"):
            status_placeholder.empty()
            st.balloons()
            best = final_result["best_solution"]
            sol = best.get('solution', {})
            eva = best.get('evaluation', {})
            
            with final_result_placeholder.container():
                st.success("🏆 最適解が発見されました！")
                st.header(f"【名称】: {sol.get('name', 'N/A')}")
                st.metric(label="最終スコア", value=f"{eva.get('total_score', 'N/A')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**概要**\n\n{sol.get('summary', 'N/A')}")
                    st.warning(f"**懸念点・改善点**\n\n{eva.get('weaknesses', 'N/A')}")
                with col2:
                    st.success(f"**優れた点**\n\n{eva.get('strengths', 'N/A')}")
                    st.info(f"**総評**\n\n{eva.get('overall_comment', 'N/A')}")
        else:
            status_placeholder.warning("処理が完了しましたが、最終的な最適解は見つかりませんでした。")

