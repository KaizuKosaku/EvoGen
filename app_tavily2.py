# app_tavily.py (修正版)
"""
EvoGen AI with Tavily integration

使い方:
  - 必要ライブラリ:
      pip install streamlit requests google-generativeai
  - 実行:
      streamlit run app_tavily.py
"""

import streamlit as st
import os
import json
import abc
from typing import List, Dict, Any, Generator, Optional
import time

# --- 外部ライブラリの読み込み ---
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import requests
except ImportError:
    requests = None

# ----------------------------
# 1) LLMクライアント層 (既存)
# ----------------------------
class LLMClient(abc.ABC):
    """LLMクライアントの基本インタフェース"""
    @abc.abstractmethod
    def call(self, prompt: str) -> Dict[str, Any]:
        pass

class GeminiClient(LLMClient):
    """Google Gemini 用のクライアント（既存実装を踏襲）"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if genai is None:
            raise ImportError("`google-generativeai`ライブラリが未インストールです。pip install google-generativeai を実行してください。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # ここでは generative model に JSON を返させる想定で設定を使う
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )

    def call(self, prompt: str) -> Dict[str, Any]:
        """
        prompt -> LLM 呼び出し -> JSON パースを試みる
        返り値: dict（失敗時は {"error": "...", "raw": "<text>"} を返す）
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            # Gemini の場合 response.text に文字列がある想定
            text = getattr(response, "text", None) or getattr(response, "response", None) or str(response)
            try:
                return json.loads(text)
            except Exception:
                # JSON パースに失敗した場合は raw テキストとして返す
                return {"raw_text": text}
        except Exception as e:
            # Streamlit上でも見えるようにログ出力するが、戻り値は dict で
            st.error(f"[GeminiClient Error] API 呼び出し中にエラーが発生しました: {e}")
            return {"error": str(e)}

# ----------------------------
# 2) Tavily クライアント (既存)
# ----------------------------
class TavilyClient:
    """
    Tavily Search API とのやり取りを行うシンプルなクライアント。
    """
    DEFAULT_ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT, timeout: int = 15):
        if requests is None:
            raise ImportError("`requests`ライブラリが未インストールです。pip install requests を実行してください。")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, num_results: int = 5, domain: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {"query": query, "max_results": num_results}
        if domain:
            payload["domain"] = domain
        if lang:
            payload["language"] = lang

        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP error: {e}"}
        except ValueError as e:
            return {"error": f"JSON parse error: {e}", "raw": resp.text if 'resp' in locals() else None}
        except Exception as e:
            return {"error": str(e)}

# ----------------------------
# 3) PromptManager（修正）
# ----------------------------
class PromptManager:
    """AIへの指示書（プロンプト）を管理するクラス"""
    
    # --- 新規追加メソッド ---
    def get_tavily_query_generation_prompt(self, problem_statement: str) -> str:
        """
        課題解決に必要な情報を検索するためのクエリをLLMに生成させるプロンプト。
        """
        return f"""
        あなたは、提示された「課題」の解決策を生成するために、インターネットで最新情報や事例を検索する専門の調査員です。
        
        以下の「課題」を分析し、**その解決策を考案するために**最も有用な情報を取得できる、具体的な**日本語の検索クエリ**を3つ生成してください。
        
        生成するクエリは、単なる課題の言い換えではなく、
        「最新の対策事例」「関連する技術の動向」「具体的なデータや制約条件」といった、解決策の質を高めるための**ファクトベースの情報**に焦点を当てたものにしてください。

        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "queries": [
            "課題解決のための情報検索クエリ1 (日本語)",
            "課題解決のための情報検索クエリ2 (日本語)",
            "課題解決のための情報検索クエリ3 (日本語)"
          ]
        }}
        """
    # -----------------------

    def get_agent_personas_prompt(self, problem_statement: str) -> str:
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
            - **role:** どのようにしてアイデアをより優れたものへと進化させるべきか？
            - **instructions:** 高評価案と低評価案をどのように分析し、次世代のアイデアを生成すべきか具体的な指示を与えてください。

        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "initial_generator": {{"role": "...", "instructions": "..."}},
          "evaluator": {{"role": "...", "criteria": [{{"criterion": "...", "weight": 10}}]}},
          "synthesizer": {{"role": "...", "instructions": "..."}}
        }}
        """

    def get_initial_generation_prompt(self, problem_statement: str, num_solutions: int, context: Dict[str, str]) -> str:
        return f"""
        # 役割: {context.get('role', 'あなたは一流のイノベーターです。')}
        # 指示: {context.get('instructions', f'以下の課題に対し、互いに全く異なるアプローチからの解決策を{num_solutions}個提案してください。')}
        # 課題文: {problem_statement}
        # 出力形式: 各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。
        {{ "solutions": [ {{ "name": "解決策1", "summary": "概要1", "specific_method": "具体的方法1" }} ] }}
        """

    def get_evaluation_prompt(self, solution: Dict[str, str], problem_statement: str, context: Dict[str, Any]) -> str:
        criteria_text = []
        scores_json_structure = []
        if "criteria" in context and isinstance(context["criteria"], list):
            for item in context["criteria"]:
                criterion = item.get("criterion", "不明な基準")
                weight = item.get("weight", 0)
                criteria_text.append(f"- {criterion}: {weight}点")
                scores_json_structure.append(f'"{criterion}": 点数(整数)')

        criteria_prompt_part = "\n".join(criteria_text)
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
        elite_text = "\n".join([f"- {s['solution'].get('name', 'N/A')} (スコア: {s['evaluation'].get('total_score', 0)})" for s in elite_solutions])
        failed_text = "\n".join([f"- {s['solution'].get('name', 'N/A')} (弱点: {s['evaluation'].get('weaknesses', 'N/A')})" for s in failed_solutions])

        return f"""
        # 役割: {context.get('role', 'あなたは優れた戦略家であり編集者です。')}
        # タスク: 前世代の分析に基づき、次世代の新しい解決策を{num_solutions}個生成してください。
        # 分析対象1：高評価だった解決案（優れた遺伝子）: 
        {elite_text}
        # 分析対象2：低評価だった解決案（学ぶべき教訓）: 
        {failed_text}
        # 新しい解決策の生成指示: {context.get('instructions', '高評価案の良い点を組み合わせ、低評価案の失敗から学び、新しい解決策を生成してください。')}
        
        # 出力形式: 各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。
        {{ "solutions": [ {{ "name": "新しい解決策1", "summary": "概要1", "specific_method": "具体的方法1" }} ] }}
        """

# ----------------------------
# 4) EvoGenSolver（既存） + Tavily 拡張
# ----------------------------
class EvoGenSolver:
    """元の EvoGenSolver（主要ロジック）"""
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

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        self.history = []

        # STEP 1: AIエージェントチームの編成
        yield "--- 🧠 課題を分析し、最適なAIエージェントチームを編成中... ---"
        agent_personas = self._generate_agent_personas(problem_statement)

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["initial_generator", "evaluator", "synthesizer"]):
            yield "エラー: チーム編成に失敗しました。処理を中断します。"
            yield f"**デバッグ情報:** AIからの応答が不正です。APIキーが正しいか確認してください。\n```\n{agent_personas}\n```"
            return

        yield f"--- ✔️ チーム編成完了 ---"
        yield {"agent_team": agent_personas}

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

        yield "\n--- ✅ 進化プロセス完了 ---"

# EvoGenSolver_Tavily: Tavily 統合版（修正）
class EvoGenSolver_Tavily(EvoGenSolver):
    """
    Tavily を用いて課題に関連する最新情報を収集し、その情報を
    問題文に組み込んで EvoGen のフローを回す拡張版。
    """
    def __init__(self, llm_client: LLMClient, tavily_client: TavilyClient, num_solutions_per_generation: int = 5, tavily_results_per_search: int = 5):
        super().__init__(llm_client, num_solutions_per_generation)
        self.tavily = tavily_client
        self.tavily_results_per_search = tavily_results_per_search

    # --- 新規追加メソッド ---
    def _generate_tavily_query(self, problem_statement: str) -> str:
        """
        LLMを呼び出し、Tavily検索に最適なクエリを生成させる。
        最初のクエリを返す。失敗したら元の問題文を返す。
        """
        prompt = self.prompter.get_tavily_query_generation_prompt(problem_statement)
        response = self._call_llm(prompt)
        
        if isinstance(response, dict) and "queries" in response and isinstance(response["queries"], list) and response["queries"]:
            # 最初のクエリを返す
            return response["queries"][0]
        
        # エラー時または不正な形式の場合は元の問題文を返す
        st.warning(f"LLMによる検索クエリ生成に失敗したため、問題文の最初の1行を使用します。\nデバッグ情報: {response}")
        # 問題文の最初の行（簡潔なクエリとして機能する可能性が高い部分）をフォールバックとして使用
        return problem_statement.strip().split('\n')[0].replace('# 課題', '').strip()
    # -----------------------

    def _summarize_tavily_results_with_llm(self, tavily_results: Dict[str, Any], problem_statement: str) -> str:
        """
        Tavily の検索結果を LLM に要約させ、問題文に統合する。
        - 失敗したら簡易フォールバック要約を返す。
        """
        results = tavily_results.get("results", []) if isinstance(tavily_results, dict) else []
        if not results:
            return problem_statement

        snippet_texts = []
        for r in results[:min(len(results), 8)]:
            title = r.get("title", "")
            snippet = r.get("snippet", "") or r.get("description", "")
            url = r.get("url", "")
            snippet_texts.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n---")

        combined = "\n".join(snippet_texts)
        prompt = f"""
        以下は、Tavily によって取得されたウェブ検索結果の抜粋です。各結果は出典（URL）を持ちます。
        あなたはこの情報を3点セットで要約し、課題にとって「特に重要な事実/データ」「潜在的な制約やリスク」「引用すべき出典(最大3つ)」を簡潔に整理して下さい。
        出力は必ず JSON 形式で以下のキーを持ってください:
        {{
          "summary": "簡潔な要約（日本語、3-4文）",
          "key_points": ["重要な事実1", "重要な事実2"],
          "risks": ["リスク1", "リスク2"],
          "top_sources": [{{"title":"...", "url":"..."}}]
        }}

        ### Tavily Results (抜粋)
        {combined}

        ### 元の課題:
        {problem_statement}
        """
        llm_ret = self._call_llm(prompt)
        
        if isinstance(llm_ret, dict):
            if any(k in llm_ret for k in ["summary", "key_points", "top_sources", "risks"]):
                try:
                    summary_text = llm_ret.get("summary", "")
                    kp = llm_ret.get("key_points", [])
                    risks = llm_ret.get("risks", [])
                    top = llm_ret.get("top_sources", [])
                    top_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in top]) if isinstance(top, list) else ""
                    composed = f"## Tavily要約（LLM生成）\n{summary_text}\n\n重要点:\n" + "\n".join([f"- {p}" for p in kp]) + "\n\nリスク:\n" + "\n".join([f"- {r}" for r in risks]) + "\n\n出典:\n" + top_text + "\n\n" + problem_statement
                    return composed
                except Exception:
                    pass
            if "raw_text" in llm_ret:
                return f"## Tavily要約（raw）\n{llm_ret['raw_text']}\n\n{problem_statement}"
            if "error" in llm_ret:
                pass

        fallback_sources = []
        for r in results[:3]:
            fallback_sources.append(f"- {r.get('title','No title')} ({r.get('url','')})")
        fallback = "## Tavily要約（フォールバック）\n" + \
                   "最新のウェブ情報を参照しました。上位出典:\n" + "\n".join(fallback_sources) + "\n\n" + problem_statement
        return fallback

    # --- solveメソッドの修正 ---
    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        
        # LLMにTavily検索クエリを生成させる
        yield "--- 💡 LLMによる最適なTavily検索クエリの生成中... ---"
        tavily_query = self._generate_tavily_query(problem_statement)
        yield f"--- ✔️ 生成された検索クエリ: **{tavily_query}** ---"
        
        # 生成されたクエリでTavily検索を実行
        yield "--- 🌐 Tavily による関連情報の検索を開始しています... ---"
        tavily_resp = self.tavily.search(tavily_query, num_results=self.tavily_results_per_search)

        if not isinstance(tavily_resp, dict) or "error" in tavily_resp:
            err = tavily_resp.get("error", "Unknown error") if isinstance(tavily_resp, dict) else "Unknown Tavily response"
            yield f"エラー: Tavily API の呼び出しに失敗しました: {err}"
            return

        yield {"tavily_info": tavily_resp}

        yield "--- ✍️ Tavily 結果を要約し、問題文に統合します... ---"
        try:
            augmented_problem = self._summarize_tavily_results_with_llm(tavily_resp, problem_statement)
        except Exception as e:
            augmented_problem = problem_statement
            yield f"警告: Tavily 要約中にエラーが発生しました: {e}"

        yield from super().solve(augmented_problem, generations)
    # -----------------------

# ----------------------------
# 5) Streamlit UI
# ----------------------------
st.set_page_config(page_title="EvoGen AI + Tavily", layout="wide")
st.title("EvoGen AI (Tavily 統合版) 🧬🌐")
st.markdown("Tavily による最新ウェブ情報を参照しながら、AIエージェントチームが進化的に解決策を探索します。")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_key = st.text_input("Google Gemini API Key", type="password", help="Gemini の API キーを入力してください（保存されません）。")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Tavily の API キーを入力してください（保存されません）。")
    st.subheader("パラメータ")
    num_generations = st.slider("世代数", 1, 5, 2, help="解決策を進化させる回数です。")
    num_solutions = st.slider("世代ごとの解決策の数", 3, 10, 4, help="1世代あたりに生成・評価する解決策の数です。")
    tavily_results_per_search = st.slider("Tavily 検索結果数", 1, 10, 5, help="Tavily から取得する検索結果数。")
    st.markdown("---")
    st.info("Tavily を使って課題に関連する最新情報を取得し、それを参考に解決策を生成します。")

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
problem_statement = st.text_area("解決したい課題を入力してください", value=default_problem, height=260)

if st.button("解決策の生成を開始", type="primary"):
    if not gemini_key:
        st.error("サイドバーでGoogle Gemini APIキーを入力してください。")
    elif not tavily_key:
        st.error("サイドバーでTavily APIキーを入力してください。")
    elif not problem_statement.strip():
        st.warning("課題を入力してください。")
    else:
        status_placeholder = st.empty()
        team_placeholder = st.empty()
        tavily_placeholder = st.empty()
        results_area = st.container()
        final_result_placeholder = st.container()

        with st.spinner("🌀 AIが思考中です..."):
            try:
                gemini_client = GeminiClient(api_key=gemini_key)
                tavily_client = TavilyClient(api_key=tavily_key)
            except Exception as e:
                st.error(f"クライアントの初期化に失敗しました: {e}")
                st.stop()

            solver = EvoGenSolver_Tavily(
                llm_client=gemini_client,
                tavily_client=tavily_client,
                num_solutions_per_generation=num_solutions,
                tavily_results_per_search=tavily_results_per_search
            )

            # --- Solverを実行し、結果をUIにストリーミング表示 ---
            for result in solver.solve(problem_statement, generations=num_generations):
                if isinstance(result, str):
                    status_placeholder.info(result)

                elif isinstance(result, dict) and "tavily_info" in result:
                    tavily_data = result["tavily_info"]
                    with tavily_placeholder.container():
                        st.subheader("🌐 Tavily 検索結果（出典付き）")
                        if "results" in tavily_data and isinstance(tavily_data["results"], list):
                            for r in tavily_data["results"]:
                                title = r.get("title", "No title")
                                url = r.get("url", "")
                                snippet = r.get("snippet", "") or r.get("description", "")
                                st.markdown(f"- [{title}]({url})")
                                if snippet:
                                    st.caption(snippet)
                        else:
                            st.warning("Tavily から想定外のレスポンスが返ってきました。")
                            st.text(json.dumps(tavily_data, ensure_ascii=False, indent=2))

                elif isinstance(result, dict) and "agent_team" in result:
                    with team_placeholder.container():
                        st.subheader("🤖 編成されたAIエージェントチーム")
                        team = result["agent_team"]
                        with st.expander("チームの詳細を表示"):
                            gen = team.get("initial_generator", {})
                            st.markdown("##### 💡 アイデア生成担当")
                            st.markdown(f"**役割:** {gen.get('role', '未定義')}")
                            st.markdown(f"**指示:** {gen.get('instructions', '未定義')}")
                            eva = team.get("evaluator", {})
                            st.markdown("##### 🧐 評価担当")
                            st.markdown(f"**役割:** {eva.get('role', '未定義')}")
                            criteria_list = eva.get('criteria', [])
                            criteria_md = ""
                            if criteria_list:
                                for c in criteria_list:
                                    criteria_md += f"- **{c.get('criterion', '項目名なし')}:** {c.get('weight', 0)}点\n"
                            st.markdown(f"**評価基準:**\n{criteria_md or '未定義'}")
                            syn = team.get("synthesizer", {})
                            st.markdown("##### 🧬 進化・統合担当")
                            st.markdown(f"**役割:** {syn.get('role', '未定義')}")
                            st.markdown(f"**指示:** {syn.get('instructions', '未定義')}")

                # === 修正箇所 (1) のまま (表示ロジック) ===
                elif isinstance(result, dict) and "generation" in result:
                    gen_data = result
                    with results_area.container():
                        st.subheader(f"第 {gen_data['generation']} 世代の結果")
                        with st.container(border=True):
                            if not gen_data.get('results'):
                                st.write("この世代では有効な解決策が生成されませんでした。")
                                continue
                            
                            for item in gen_data.get('results', []):
                                sol = item.get('solution', {})
                                eva = item.get('evaluation', {})
                                score = eva.get('total_score', 0)
                                
                                # 依頼に沿った「題名」と「内容」の形式で表示
                                st.markdown(f"**題名:** {sol.get('name', 'N/A')} (スコア: {score})")
                                content = sol.get('summary', 'N/A')
                                st.markdown(f"**内容:** {content}")
                                
                                # 各解決案の区切り線
                                if item != gen_data.get('results', [])[-1]:
                                    st.markdown("---")
                # === 修正箇所 (1) のまま ===

        # === 修正箇所 (2) のまま (最終結果の表示ロジック) ===
        # --- 最終結果の表示（トップ5ランキング） ---
        
        # すべての世代から評価済みの解決策を収集
        all_solutions = [
            item for gen in solver.history
            for item in gen.get("results", [])
            if item.get("evaluation") and "total_score" in item["evaluation"]
        ]

        if all_solutions:
            # スコアで降順にソート
            sorted_solutions = sorted(
                all_solutions,
                key=lambda x: x["evaluation"]["total_score"],
                reverse=True
            )
            
            top_5_solutions = sorted_solutions[:5]

            status_placeholder.empty()
            st.balloons()

            with final_result_placeholder:
                st.success("🏆 処理完了！スコアトップ5の解決策はこちらです。")
                
                for i, item in enumerate(top_5_solutions):
                    sol = item.get('solution', {})
                    eva = item.get('evaluation', {})
                    score = eva.get('total_score', 'N/A')
                    
                    st.header(f"🏅 第 {i + 1} 位: {sol.get('name', 'N/A')}")
                    st.metric(label="最終スコア", value=f"{score}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**概要**\n\n{sol.get('summary', 'N/A')}")
                        st.warning(f"**懸念点・改善点**\n\n{eva.get('weaknesses', 'N/A')}")
                    with col2:
                        st.success(f"**優れた点**\n\n{eva.get('strengths', 'N/A')}")
                        st.info(f"**総評**\n\n{eva.get('overall_comment', 'N/A')}")
                    st.markdown("---")
        else:
            status_placeholder.warning("処理が完了しましたが、最終的な解決策は見つかりませんでした。")
        # === 修正箇所 (2) のまま ===