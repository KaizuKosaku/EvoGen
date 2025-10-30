# app_tavily_fixed_v3.py
"""
EvoGen AI with Tavily integration (v3: 2-Phase RAG版)

使い方:
  - 必要ライブラリ:
      pip install streamlit requests google-generativeai
  - 実行:
      streamlit run app_tavily_fixed_v3.py
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
# 3) PromptManager（★修正箇所★）
# ----------------------------
class PromptManager:
    """AIへの指示書（プロンプト）を管理するクラス"""
    
    # === ★修正箇所 1: 2フェーズのクエリを生成するプロンプトに変更 ===
    def get_tavily_multi_phase_query_prompt(self, problem_statement: str) -> str:
        """
        課題解決に必要な情報を「分析」と「解決策」の2フェーズで検索するための
        クエリをLLMに生成させるプロンプト。
        """
        return f"""
        あなたは、提示された「課題」を解決するための調査を2段階で行う専門の調査員です。

        以下の「課題」を分析し、2つのフェーズに対応する**日本語の検索クエリ**をそれぞれ2つずつ生成してください。

        # フェーズ1: 現状・背景分析
        課題文に含まれる固有名詞（組織名、地名、特定のシステム名など）を特定し、
        その対象の「最新情報」「現状のデータ」「関連する背景や制約」を調査するためのクエリ。
        (例: 「九州工業大学 志願者数 最新データ」, 「九州工業大学 現在の広報戦略」)

        # フェーズ2: 解決策の事例・技術調査
        課題そのものを解決するための「最新の対策事例」「関連する新しい技術の動向」「他分野での成功事例」を調査するためのクエリ。
        (例: 「大学 志願者数 増加施策 事例」, 「Z世代向け 大学マーケティング手法」)

        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "analysis_queries": [
            "フェーズ1のクエリ1 (日本語)",
            "フェーズ1のクエリ2 (日本語)"
          ],
          "solution_queries": [
            "フェーズ2のクエリ1 (日本語)",
            "フェーズ2のクエリ2 (日本語)"
          ]
        }}
        """
    # === ★修正箇所 1 終了 ===

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
        # 出力形式: 
        各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。
        
        # !!重要!! 
        - 「specific_method」の内容は、その方法論やメカニズム、その理由などを説明する**簡潔な平易な文章（2〜3文程度）**にしてください。
        - 「specific_method」には**箇条書き、マークダウン、ネストされたJSON、改行コード(\n)を使用しないでください。** 必ず単一の文字列（String）にしてください。

        {{ 
          "solutions": [ 
            {{ 
              "name": "解決策1の名称", 
              "summary": "解決策1の簡潔な概要", 
              "specific_method": "解決策1の具体的な方法や理由を説明する簡潔な文章です。" 
            }},
            {{ 
              "name": "解決策2の名称", 
              "summary": "解決策2の簡潔な概要", 
              "specific_method": "解決策2の具体的な方法や理由を説明する簡潔な文章です。" 
            }}
          ] 
        }}
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
        - 具体的な方法: {solution.get('specific_method', '具体的な方法なし')}
        
        # 評価基準:
        {criteria_prompt_part}
        
        # 出力形式: 評価結果を必ず以下のJSON形式で出力してください。
        {{
          "total_score": 合計点(整数),
          "scores": {scores_json_prompt_part},
          "strengths": "この解決案が優れている点（簡潔に）",
          "weaknesses": "この解決案の懸念点や改善が必要な点（簡潔に）",
          "overall_comment": "評価の総括（簡潔に）"
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
        
        # 出力形式: 
        各解決策に「name」「summary」「specific_method」を必ず含め、JSON形式でリストとして出力してください。
        
        # !!重要!! 
        - 「specific_method」の内容は、その方法論やメカニズム、その理由などを説明する**簡潔な平易な文章（2〜3文程度）**にしてください。
        - 「specific_method」には**箇条書き、マークダウン、ネストされたJSON、改行コード(\n)を使用しないでください。** 必ず単一の文字列（String）にしてください。

        {{ 
          "solutions": [ 
            {{ 
              "name": "新しい解決策1の名称", 
              "summary": "新しい解決策1の簡潔な概要", 
              "specific_method": "新しい解決策1の具体的な方法や理由を説明する簡潔な文章です。" 
            }},
            {{ 
              "name": "新しい解決策2の名称", 
              "summary": "新しい解決策2の簡潔な概要", 
              "specific_method": "新しい解決策2の具体的な方法や理由を説明する簡潔な文章です。" 
            }}
          ] 
        }}
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
        # 形式が崩れた場合（dictだがsolutionsがない、またはリストでない）のエラーハンドリングを強化
        if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list):
            return response["solutions"]
        else:
            st.warning(f"[EvoGenSolver] 初期解決策の生成で不正な形式が返されました。デバッグ情報: {response}")
            return []

    def _evaluate_solutions(self, solutions: List[Dict[str, str]], problem_statement: str, context: Dict) -> Generator[str | List[Dict], None, None]:
        evaluated_solutions = []
        if not solutions:
            yield []
            return

        for i, solution in enumerate(solutions):
            # 解決策オブジェクト自体が不正な形式でないかチェック
            if not isinstance(solution, dict) or "name" not in solution:
                yield f"  - 評価スキップ: 不正な形式の解決策データです。"
                continue
                
            yield f"  - 評価中 {i+1}/{len(solutions)}: {solution.get('name', '名称不明')}"
            prompt = self.prompter.get_evaluation_prompt(solution, problem_statement, context)
            evaluation = self._call_llm(prompt)
            
            # 評価結果が期待通り（dictでtotal_scoreを持つ）かチェック
            if isinstance(evaluation, dict) and "total_score" in evaluation and "error" not in evaluation:
                evaluated_solutions.append({"solution": solution, "evaluation": evaluation})
            else:
                st.warning(f"[EvoGenSolver] 解決策 '{solution.get('name', 'N/A')}' の評価で不正な形式が返されました。デバッグ情報: {evaluation}")


        evaluated_solutions.sort(key=lambda x: x.get("evaluation", {}).get("total_score", 0), reverse=True)
        yield evaluated_solutions

    def _generate_next_generation(self, evaluated_solutions: List[Dict], problem_statement: str, context: Dict) -> List[Dict[str, str]]:
        num_elites = max(1, int(len(evaluated_solutions) * 0.4))
        elite_solutions = evaluated_solutions[:num_elites]
        failed_solutions = evaluated_solutions[num_elites:]
        prompt = self.prompter.get_next_generation_prompt(elite_solutions, failed_solutions, problem_statement, self.num_solutions, context)
        response = self._call_llm(prompt)
        # 形式が崩れた場合（dictだがsolutionsがない、またはリストでない）のエラーハンドリングを強化
        if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list):
            return response["solutions"]
        else:
            st.warning(f"[EvoGenSolver] 次世代の解決策生成で不正な形式が返されました。デバッグ情報: {response}")
            return []

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
        
        if not solutions:
             yield "エラー: 最初の解決策生成に失敗しました。AIが適切な応答を返さなかった可能性があります。処理を終了します。"
             return

        yield "--- 🧐 アイデアを評価中... ---"
        eval_generator = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluator"])
        evaluated_solutions = []
        for item in eval_generator:
            if isinstance(item, str):
                yield item
            else:
                evaluated_solutions = item
        
        if not evaluated_solutions:
             yield "エラー: 解決策の評価に失敗しました。処理を終了します。"
             return

        self.history.append({"generation": 0, "results": evaluated_solutions})
        yield self.history[-1]

        # STEP 3: 世代の進化
        for i in range(1, generations):
            yield f"\n--- 🚀 Generation {i}: 次のアイデアへ進化中... ---"
            previous_generation_results = self.history[-1]["results"]
            
            if not previous_generation_results:
                yield f"エラー: 前世代 ({i-1}) の有効な評価結果がありません。進化を停止します。"
                break

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

            if not evaluated_solutions_next:
                 yield f"エラー: Generation {i} の評価に失敗しました。処理を終了します。"
                 break

            self.history.append({"generation": i, "results": evaluated_solutions_next})
            yield self.history[-1]

        yield "\n--- ✅ 進化プロセス完了 ---"

# === ★修正箇所 2: EvoGenSolver_Tavily の修正 ===
class EvoGenSolver_Tavily(EvoGenSolver):
    """
    Tavily を用いて課題に関連する最新情報を収集し、その情報を
    問題文に組み込んで EvoGen のフローを回す拡張版。
    (v3: 2フェーズRAG)
    """
    def __init__(self, llm_client: LLMClient, tavily_client: TavilyClient, num_solutions_per_generation: int = 5, tavily_results_per_search: int = 5):
        super().__init__(llm_client, num_solutions_per_generation)
        self.tavily = tavily_client
        # tavily_results_per_search は「クエリごと」の取得数とする
        self.tavily_results_per_query = tavily_results_per_search 

    def _get_snippet_text(self, results: List[Dict[str, Any]], max_snippets: int = 5) -> str:
        """Tavilyの結果リストからスニペット文字列を生成するヘルパー"""
        snippet_texts = []
        for r in results[:min(len(results), max_snippets)]:
            title = r.get("title", "")
            snippet = r.get("snippet", "") or r.get("description", "")
            url = r.get("url", "")
            snippet_texts.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n---")
        return "\n".join(snippet_texts)

    def _summarize_multi_phase_results_with_llm(
        self, 
        problem_statement: str, 
        analysis_results: List[Dict[str, Any]], 
        solution_results: List[Dict[str, Any]]
    ) -> str:
        """
        2フェーズのTavily検索結果をLLMに要約させ、問題文に統合する。
        """
        if not analysis_results and not solution_results:
            return problem_statement

        analysis_snippets = self._get_snippet_text(analysis_results, max_snippets=5)
        solution_snippets = self._get_snippet_text(solution_results, max_snippets=5)

        prompt = f"""
        あなたは、2段階のウェブ調査結果を分析し、元の課題文に統合する専門家です。
        
        # 元の課題
        {problem_statement}

        # 調査結果 1: 現状・背景分析 (固有名詞や現状のデータ)
        {analysis_snippets if analysis_snippets else "なし"}

        # 調査結果 2: 解決策の事例・技術 (他事例や技術動向)
        {solution_snippets if solution_snippets else "なし"}

        # タスク
        上記の2つの調査結果を分析し、元の課題を解決する上で特に重要となる情報を抽出・要約してください。
        
        # 出力形式 (JSON)
        {{
          "summary_analysis": "「調査結果1（現状・背景）」の簡潔な要約（1〜2文）",
          "summary_solution": "「調査結果2（解決策・事例）」の簡潔な要約（1〜2文）",
          "key_points": [
            "調査結果全体から得られた重要な事実や制約1",
            "調査結果全体から得られた重要な事実や制約2"
          ],
          "top_sources": [
            {{"title":"最も重要な出典のタイトル1", "url":"..."}},
            {{"title":"最も重要な出典のタイトル2", "url":"..."}}
          ]
        }}
        """
        
        llm_ret = self._call_llm(prompt)
        
        # LLMによる要約が成功した場合
        if isinstance(llm_ret, dict) and any(k in llm_ret for k in ["summary_analysis", "summary_solution", "key_points"]):
            try:
                summary_analysis_text = llm_ret.get("summary_analysis", "現状分析の要約なし")
                summary_solution_text = llm_ret.get("summary_solution", "解決策事例の要約なし")
                kp = llm_ret.get("key_points", [])
                top = llm_ret.get("top_sources", [])
                top_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in top]) if isinstance(top, list) else ""
                
                composed = f"""
## Tavilyリサーチ要約（LLM生成）
### 現状・背景分析
{summary_analysis_text}
### 解決策・事例
{summary_solution_text}

### 抽出された重要点
""" + "\n".join([f"- {p}" for p in kp]) + "\n\n" + \
"### 主な出典\n" + top_text + "\n\n" + \
"--- (以下、元の課題文) ---\n" + problem_statement
                
                return composed
            except Exception:
                pass # フォールバック処理へ

        # LLM要約が失敗した場合（フォールバック）
        fallback_sources = []
        for r in analysis_results[:2]:
            fallback_sources.append(f"- [分析] {r.get('title','No title')} ({r.get('url','')})")
        for r in solution_results[:2]:
            fallback_sources.append(f"- [解決策] {r.get('title','No title')} ({r.get('url','')})")
            
        fallback = "## Tavilyリサーチ要約（フォールバック）\n" + \
                   "最新のウェブ情報を参照しました。上位出典:\n" + "\n".join(fallback_sources) + \
                   "\n\n" + "--- (以下、元の課題文) ---\n" + problem_statement
        return fallback

    # --- solveメソッドの修正 ---
    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        
        # LLMに2フェーズのTavily検索クエリを生成させる
        yield "--- 💡 LLMによる最適な検索クエリ（フェーズ1 & 2）を生成中... ---"
        prompt = self.prompter.get_tavily_multi_phase_query_prompt(problem_statement)
        query_response = self._call_llm(prompt)

        if not isinstance(query_response, dict) or ("analysis_queries" not in query_response and "solution_queries" not in query_response):
            yield f"エラー: Tavilyクエリの生成に失敗しました。AIからの応答が不正です: {query_response}"
            # クエリ生成が失敗しても、Tavily無しで続行
            augmented_problem = problem_statement
        else:
            analysis_queries = query_response.get("analysis_queries", [])
            solution_queries = query_response.get("solution_queries", [])
            
            yield f"--- ✔️ 生成されたクエリ ---"
            yield f"  - 分析クエリ: {', '.join(analysis_queries) if analysis_queries else 'なし'}"
            yield f"  - 解決策クエリ: {', '.join(solution_queries) if solution_queries else 'なし'}"
            
            analysis_results_list = []
            solution_results_list = []

            # --- フェーズ1: 分析クエリの実行 ---
            if analysis_queries:
                yield "--- 🌐 フェーズ1: 課題の現状分析リサーチを開始... ---"
                for q in analysis_queries:
                    if not q.strip(): continue
                    yield f"  - 検索中 (分析): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_per_query)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        analysis_results_list.extend(tavily_resp["results"])
                    elif isinstance(tavily_resp, dict) and "error" in tavily_resp:
                         yield f"  - Tavily エラー (分析クエリ: {q}): {tavily_resp['error']}"
            
            # --- フェーズ2: 解決策クエリの実行 ---
            if solution_queries:
                yield "--- 🌐 フェーズ2: 解決策の事例リサーチを開始... ---"
                for q in solution_queries:
                    if not q.strip(): continue
                    yield f"  - 検索中 (解決策): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_per_query)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        solution_results_list.extend(tavily_resp["results"])
                    elif isinstance(tavily_resp, dict) and "error" in tavily_resp:
                         yield f"  - Tavily エラー (解決策クエリ: {q}): {tavily_resp['error']}"

            # UIに検索結果を渡す
            yield {"tavily_info_analysis": analysis_results_list, "tavily_info_solution": solution_results_list}

            # --- 2つのリサーチ結果を要約・統合 ---
            yield "--- ✍️ 2つのリサーチ結果を要約し、問題文に統合します... ---"
            try:
                augmented_problem = self._summarize_multi_phase_results_with_llm(
                    problem_statement, 
                    analysis_results_list, 
                    solution_results_list
                )
            except Exception as e:
                augmented_problem = problem_statement
                yield f"警告: Tavily 要約中にエラーが発生しました: {e}"

        # 拡張された問題文（または元の問題文）でEvoGenの本体を実行
        yield from super().solve(augmented_problem, generations)
    # === ★修正箇所 2 終了 ===

# ----------------------------
# 5) Streamlit UI (★修正箇所★)
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
    tavily_results_per_search = st.slider("Tavily 検索結果数 (クエリ毎)", 1, 10, 3, help="1つのクエリあたりにTavily から取得する検索結果数。")
    st.markdown("---")
    st.info("Tavily を使って課題に関連する最新情報を取得し、それを参考に解決策を生成します。")

default_problem = """
# 課題
九州工業大学の入学志願者数を増加させるような画期的な解決策を提案せよ。

# 要件・制約条件
- コストがあまりかからないこと。
- 九州工業大学のイメージを損なわないこと。
- 優秀な学生にアプローチできること。
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
        tavily_placeholder = st.container() # プレースホルダーをコンテナに変更
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
            
            # Tavily結果をTavilyプレースホルダーに表示するためのヘルパー関数
            def display_tavily_results(results_list, title):
                with tavily_placeholder.container():
                    st.subheader(title)
                    if results_list:
                        for r in results_list:
                            title = r.get("title", "No title")
                            url = r.get("url", "")
                            snippet = r.get("snippet", "") or r.get("description", "")
                            st.markdown(f"- [{title}]({url})")
                            if snippet:
                                st.caption(snippet)
                    else:
                        st.write("このフェーズでは検索結果がありませんでした。")
                    st.markdown("---")


            # --- Solverを実行し、結果をUIにストリーミング表示 ---
            for result in solver.solve(problem_statement, generations=num_generations):
                if isinstance(result, str):
                    status_placeholder.info(result)

                # === ★修正箇所 3: 2フェーズのTavily結果をハンドリング ===
                elif isinstance(result, dict) and ("tavily_info_analysis" in result or "tavily_info_solution" in result):
                    # 既にTavilyプレースホルダーに書き込んでいる可能性があるため、一度クリアする
                    tavily_placeholder.empty()
                    
                    analysis_data = result.get("tavily_info_analysis", [])
                    solution_data = result.get("tavily_info_solution", [])
                    
                    if analysis_data:
                        display_tavily_results(analysis_data, "🌐 フェーズ1: 課題の現状分析リサーチ結果")
                    
                    if solution_data:
                        display_tavily_results(solution_data, "🌐 フェーズ2: 解決策の事例リサーチ結果")
                # === ★修正箇所 3 終了 ===

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

                # === 世代ごと（途中経過）の表示（UI変更なし） ===
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
                                
                                st.markdown(f"**題名:** {sol.get('name', 'N/A')} (スコア: {score})")
                                content = sol.get('specific_method', 'N/A') 
                                st.markdown(f"**具体的な方法:** {content}")
                                
                                if item != gen_data.get('results', [])[-1]:
                                    st.markdown("---")

        # === 最終結果の表示（UI変更なし） ===
        
        all_solutions = [
            item for gen in solver.history
            for item in gen.get("results", [])
            if item.get("evaluation") and "total_score" in item["evaluation"]
        ]

        if all_solutions:
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
                        st.info(f"**具体的な方法**\n\n{sol.get('specific_method', 'N/A')}")
                        st.warning(f"**懸念点・改善点**\n\n{eva.get('weaknesses', 'N/A')}")
                    with col2:
                        st.success(f"**優れた点**\n\n{eva.get('strengths', 'N/A')}")
                        st.info(f"**総評**\n\n{eva.get('overall_comment', 'N/A')}")
                    st.markdown("---")
        else:
            status_placeholder.warning("処理が完了しましたが、最終的な解決策は見つかりませんでした。")
