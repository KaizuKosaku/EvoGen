# app_tavily_fixed_v16_batch_queries.py
"""
EvoGen AI with Tavily integration (v16.0: バッチクエリ最適化モデル)

v15.0 (gen_ai_04.py) からの変更点:
- (API効率化) ユーザーの要望に基づき、エージェント個別調査の「クエリ生成」を
  バッチ処理（10エージェント分をLLM 1回の呼び出し）で行うように最適化。
- `PromptManager` (修正):
  - (削除) `get_agent_specific_queries_prompt`: v15.0で 10回呼び出されていた
    個別クエリ生成プロンプトを削除。
  - (新設) `get_all_agent_queries_prompt`: 全エージェント(10体)のリストを受け取り、
    全エージェント分のクエリ(20個)を「1回のLLM呼び出し」でまとめて生成する
    バッチ処理用プロンプトを新設。
- `EvoGenSolver_Tavily` (修正):
  - `_run_agent_specific_research` (ロジック修正):
    - v15.0: ループ内で [LLMクエリ生成 -> Tavily検索 -> LLM分析] を実行していた。
    - v16.0: 
      1. ループの「前」に `get_all_agent_queries_prompt` を1回だけ呼び出し、
         全クエリを辞書として取得。
      2. ループ内では、辞書からクエリを引いて [Tavily検索 -> LLM分析] のみ実行。
    - これにより、このステップでのLLM呼び出し回数を 20回 -> 11回 に削減。

v15.0 の特徴 (gen_ai_04.py):
- (エージェント個別調査) 各エージェントが自身の役割専用の調査と分析を行う。
- (深層分析) TavilyでWebページの「全文」を取得し、LLMが「戦略的洞察」を抽出。
- (汎用性) `{"proposal_main": "...", "proposal_details": "..."}` の2分割JSON構造。
- (動的UI) `output_labels` をAIが動的に生成し、UIに反映する。
"""

import streamlit as st
import os
import json
import abc
from typing import List, Dict, Any, Generator, Optional
import time
import random 

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
# 1) LLMクライアント層 (v14.0のまま変更なし)
# ----------------------------
class LLMClient(abc.ABC):
    """LLMクライアントの基本インタフェース"""
    @abc.abstractmethod
    def call(self, prompt: str) -> Dict[str, Any]:
        pass

class GeminiClient(LLMClient):
    """Google Gemini 用のクライアント（v9.0 JSON修復機能付き）"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"): 
        if genai is None:
            raise ImportError("`google-generativeai`ライブラリが未インストールです。pip install google-generativeai を実行してください。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """
        マークダウンや他のテキストでラップされている可能性のある
        文字列から、最初と最後の波括弧/角括弧に基づいてJSONブロックを抽出する。
        """
        if not text:
            return None
        
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return None
            
        if start_brace == -1:
            start = start_bracket
        elif start_bracket == -1:
            start = start_brace
        else:
            start = min(start_brace, start_bracket)
            
        end_brace = text.rfind('}')
        end_bracket = text.rfind(']')
        
        if end_brace == -1 and end_bracket == -1:
            return None
            
        end = max(end_brace, end_bracket)
        
        if end <= start:
            return None
            
        potential_json = text[start:end+1]
        return potential_json

    def _get_json_repair_prompt(self, malformed_text: str) -> str:
        """
        LLMが生成した不正な形式のテキストを修復させるためのプロンプトを生成する。
        """
        return f"""
        # 指示
        あなたは以前、JSON形式での出力を求められましたが、以下のテキストを生成しました。
        しかし、このテキストはJSONとして正しくパース（解析）できませんでした。

        # 不正な形式のテキスト
        ```
        {malformed_text}
        ```

        # タスク
        上記のテキスト内容を**完全に**反映しつつ、**マークダウン (```json ... ```) や説明文を一切含まない、
        厳密に正しいJSONオブジェクト（`{{ ... }}` または `[ ... ]` で始まる）**
        として修正し、そのJSONだけを出力してください。
        """

    def call(self, prompt: str, is_retry: bool = False) -> Dict[str, Any]:
        """
        prompt -> LLM 呼び出し -> JSON クリーニング -> JSON パースを試みる
        パースに失敗した場合、LLMに修復を依頼するリトライを1回行う。
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            text = getattr(response, "text", None) or getattr(response, "response", None) or str(response)
            
            cleaned_text = self._extract_json(text)
            
            if cleaned_text:
                try:
                    return json.loads(cleaned_text) 
                except Exception as e_clean:
                    st.warning(f"[GeminiClient Warning] JSONのパースに失敗しました (クリーニング後)。 Error: {e_clean}")
                    
                    if is_retry:
                        st.error(f"[GeminiClient Error] JSON修復リトライにも失敗しました。")
                        return {"raw_text": text, "parse_error": f"Retry failed: {e_clean}"}
                    else:
                        st.info(f"[GeminiClient Info] JSON修復のため、LLMにリトライします...")
                        repair_prompt = self._get_json_repair_prompt(text)
                        return self.call(repair_prompt, is_retry=True)
            else:
                st.warning(f"[GeminiClient Warning] 応答からJSONブロックが見つかりませんでした。")
                
                if is_retry:
                    st.error(f"[GeminiClient Error] JSON修復リトライ後も、JSONブロックが見つかりませんでした。")
                    return {"raw_text": text, "parse_error": "Retry failed: No JSON block found"}
                else:
                    st.info(f"[GeminiClient Info] JSON修復のため、LLMにリトライします...")
                    repair_prompt = self._get_json_repair_prompt(text)
                    return self.call(repair_prompt, is_retry=True)
                
        except Exception as e:
            st.error(f"[GeminiClient Error] API 呼び出し中にエラーが発生しました: {e}")
            if is_retry:
                return {"error": f"API call failed during retry: {e}"}
            else:
                return {"error": str(e)}


# ----------------------------
# 2) Tavily クライアント (v14.0のまま変更なし)
# ----------------------------
class TavilyClient:
    """
    Tavily Search API とのやり取りを行うシンプルなクライアント。
    (v14.0: 全文取得対応)
    """
    DEFAULT_ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT, timeout: int = 15):
        if requests is None:
            raise ImportError("`requests`ライブラリが未インストールです。pip install requests を実行してください。")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, num_results: int = 5, domain: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        """
        (v14.0) 全文取得 (`include_raw_content: True`) を常に行う
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "query": query, 
            "max_results": num_results,
            "include_raw_content": True, # Webページの全文(生テキスト)を要求
            "search_depth": "advanced"    # より詳細な検索を実行
        }
        
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
# 3) PromptManager (★v16.0: 修正箇所★)
# ----------------------------
class PromptManager:
    """AIへの指示書（プロンプト）を管理するクラス"""
    
    def get_tavily_multi_phase_query_prompt(self, problem_statement: str) -> str:
        """
        (v14.0のまま) 課題文の「事前補強」用
        """
        return f"""
        あなたは、提示された「課題」を解決するための調査を2段階で行う専門の調査員です。

        以下の「課題」を分析し、2つのフェーズに対応する**日本語の検索クエリ**をそれぞれ4つずつ生成してください。

        # フェーズ1: 現状・背景分析
        課題文に含まれる固有名詞（組織名、地名、特定のシステム名など）を特定し、
        その対象の「最新情報」「現状のデータ」「関連する背景や制約」を調査するためのクエリ。

        # フェーズ2: 解決策の事例・技術調査
        課題そのものを解決するための「最新の対策事例」「関連する新しい技術の動向」「他分野での成功事例」を調査するためのクエリ。

        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "analysis_queries": [
            "フェーズ1のクエリ1 (日本語)",
            "フェーズ1のクエリ2 (日本語)",
            "フェーズ1のクエリ3 (日本語)",
            "フェーズ1のクエリ4 (日本語)"
          ],
          "solution_queries": [
            "フェーズ2のクエリ1 (日本語)",
            "フェーズ2のクエリ2 (日本語)",
            "フェーズ2のクエリ3 (日本語)",
            "フェーズ2のクエリ4 (日本語)"
          ]
        }}
        """

    def get_agent_personas_prompt(self, problem_statement: str) -> str:
        """
        (v13.0のまま)
        """
        return f"""
        # 役割
        あなたは、非常に複雑な課題を解決するために、AIエージェントからなる「スウォーム（群れ）」を編成する「マスタープランナー」です。

        # タスク
        以下の「課題」を解決するために、最も効果的なAIエージェント群と、成果物の表示ラベルを定義してください。
        編成は以下のステップで厳密に行ってください。

        ## ステップ1: 課題の徹底分析 (Your Internal Monologue)
        1.  **核心的目標(Goal)は何か？**
        2.  **タスクの性質**: この課題は「解決策(Solution)」か「創作物(Creative)」か？
        3.  **主要な制約(Constraints)は何か？**
        4.  **主要な利害関係者(Stakeholders)は誰か？**

        ## ステップ2: 解決・進化担当エージェント (10体) の定義
        - ステップ1の分析に基づき、課題解決に最適化された「互いに異なる10の視点」を持つ専門家（solver_agents）を定義してください。
        - **重要**: 「マーケター」のような一般的な役割ではなく、「**[利害関係者]の[特定の課題]を解決する専門家**」や「**[主要な制約]をクリアする[特定技術]の専門家**」のように、**この課題専用に特化させた役割（role）**を定義してください。
        - `instructions`には、その専門性を活かして「初期解の生成」と「既存解の進化」の両方でどう振る舞うべきか具体的に指示してください。

        ## ステップ3: 課題特化型 評価エージェント (3体) の定義
        - ステップ1の分析に基づき、生成された提案を評価するために**最も重要となる3つの異なる評価観点**を特定してください。
        - その3つの観点に基づき、それぞれ専門の評価エージェント（evaluators）を3体定義してください。
        - `role`: あなたが考案した、課題に特化した評価者の役割名。
        - `evaluation_guideline`: (v11.0のまま) その役割が提案を厳密に評価するために使用する、**具体的かつ詳細な評価指針（ガイドライン）**。

        ## ステップ4: 動的UIラベルの定義 (v13.0)
        - ステップ1の「タスクの性質」分析に基づき、最終的な成果物をUIに表示するための2つのラベル (`output_labels`) を定義してください。
        - **`main_label`**: 成果物の「核」となる部分のラベル。(例: "提案の名称", "創作した俳句")
        - **`details_label`**: 成果物の「詳細」となる部分のラベル。(例: "概要と具体的な方法", "俳句の意図と背景")
        
        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "output_labels": {{
             "main_label": "（ステップ4で定義したメインラベル）",
             "details_label": "（ステップ4で定義した詳細ラベル）"
          }},
          "solver_agents": [
            {{ "role": "（ステップ2で定義した専門的役割1）", "instructions": "..." }},
            // ... (10体分)
            {{ "role": "（ステップ2で定義した専門的役割10）", "instructions": "..." }}
          ],
          "evaluators": [
            {{ "role": "（ステップ3で考案した評価役割1）", "evaluation_guideline": "..." }},
            {{ "role": "（ステップ3で考案した評価役割2）", "evaluation_guideline": "..." }},
            {{ "role": "（ステップ3で考案した評価役割3）", "evaluation_guideline": "..." }}
          ]
        }}
        """

    # === ★v16.0: 新設 (全エージェントのクエリをバッチ生成) ===
    def get_all_agent_queries_prompt(self, problem_statement: str, solver_agents: List[Dict]) -> str:
        """
        (v16.0) 10体すべてのエージェントの役割に基づき、
        必要な検索クエリ(合計20個)を「1回のLLM呼び出し」で
        まとめて生成させる。
        """
        
        # エージェントリストをプロンプト用に整形
        agent_list_text = []
        for i, agent in enumerate(solver_agents):
            agent_list_text.append(f"### エージェント {i+1}")
            agent_list_text.append(f"role: \"{agent.get('role', 'N/A')}\"")
            agent_list_text.append(f"instructions: {agent.get('instructions', 'N/A')}")
        
        agents_definition_block = "\n".join(agent_list_text)

        return f"""
        # 全体の課題
        {problem_statement}

        # 編成された専門家チーム (10体)
        {agents_definition_block}

        # タスク
        あなたは、上記の専門家チーム（10体）の調査を補佐する「調査チーフ」です。
        各専門家が、その独自の「役割(role)」と「指示(instructions)」に基づき、
        「全体の課題」に対する優れた提案を行うために必要となる
        **日本語の検索クエリ**を、各エージェントごとに**厳密に2つ**ずつ生成してください。

        # !!最重要!! 出力形式 (JSON)
        - 10体のエージェント全員分のクエリを生成してください。
        - キーは、上記で提示された**「role」の文字列と完全に一致**させてください。
        - JSONオブジェクト `{{ ... }}` のみを出力してください。
        
        {{
          "agent_queries": {{
            "（エージェント1の role 文字列）": [
              "（エージェント1の視点での検索クエリ1）",
              "（エージェント1の視点での検索クエリ2）"
            ],
            "（エージェント2の role 文字列）": [
              "（エージェント2の視点での検索クエリ1）",
              "（エージェント2の視点での検索クエリ2）"
            ],
            // ... 10体全員分 ...
            "（エージェント10の role 文字列）": [
              "（エージェント10の視点での検索クエリ1）",
              "（エージェント10の視点での検索クエリ2）"
            ]
          }}
        }}
        """

    # === ★v15.0: (v16.0でも変更なし) エージェント個別分析 ===
    def get_agent_specific_analysis_prompt(self, problem_statement: str, agent_role: str, agent_instructions: str, raw_content_text: str) -> str:
        """
        (v15.0) 特定のエージェントが、自身の役割視点で全文コンテンツを分析し、
        「10個の箇条書きの洞察」を抽出する。
        """
        return f"""
        # 全体の課題
        {problem_statement}

        # あなたの専門家としての役割
        あなたは「{agent_role}」です。
        
        # あなたへの指示
        {agent_instructions}

        # あなた専用の調査資料 (Webページ全文)
        {raw_content_text}
        
        # タスク
        あなたは今、あなたの役割専用の「調査資料」（Webページの全文）を読み終えました。
        あなたの「役割」と「指示」に厳密に従い、上記の「全体の課題」に対する
        独自の提案を生成するために、この調査資料から得られる
        **最も重要で具体的な洞察（キーインサイト）**を、
        **簡潔な箇条書きで10個程度**、抽出してください。

        # 出力形式 (JSON)
        {{
          "key_insights": [
            "（{agent_role}の視点で抽出した重要な洞察1）",
            "（{agent_role}の視点で抽出した重要な洞察2）",
            "（{agent_role}の視点で抽出した重要な洞察3）",
            "（{agent_role}の視点で抽出した重要な洞察4）",
            "（{agent_role}の視点で抽出した重要な洞察5）",
            "（{agent_role}の視点で抽出した重要な洞察6）",
            "（{agent_role}の視点で抽出した重要な洞察7）",
            "（{agent_role}の視点で抽出した重要な洞察8）",
            "（{agent_role}の視点で抽出した重要な洞察9）",
            "（{agent_role}の視点で抽出した重要な洞察10）"
          ]
        }}
        """

    # === ★v15.0: (v16.0でも変更なし) 個別調査情報を参照 ===
    def get_initial_generation_prompt(self, problem_statement: str, num_solutions: int, context: Dict[str, Any]) -> str:
        """
        (v15.0) `proposal_main` と `proposal_details` を生成させる。
        エージェント専用の `agent_research_insights` を参照する。
        """
        
        # v15.0: エージェント個別の調査情報を取得
        insights = context.get('agent_research_insights', [])
        insights_text = "\n".join([f"- {item}" for item in insights]) if insights else "（追加の調査情報なし）"

        return f"""
        # 役割: {context.get('role', 'あなたは一流のイノベーターです。')}
        # 指示: {context.get('instructions', f'以下の課題に対し、互いに全く異なるアプローチからの提案を{num_solutions}個生成してください。')}
        # 課題文: {problem_statement}

        # ★あなた専用の調査情報 (v15.0)★
        # 以下の個別の調査結果を**必ず**参考にして、独自の提案を生成してください。
        {insights_text}

        # !!最重要!! (出力形式)
        各提案に「proposal_main」「proposal_details」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義 (v13.0)
        * **proposal_main**: 提案の「核」となる部分。(例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。(例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_main": "提案1の核 (名称 または 創作物そのもの)", 
              "proposal_details": "提案1の詳細 (具体的な内容、意図、背景、理由、効果など) を説明する2〜4行の文章です。\nこのように改行を含めても構いません。"
            }}
          ] 
        }}
        """

    def get_evaluation_prompt(self, solution: Dict[str, str], problem_statement: str, context: Dict[str, Any]) -> str:
        """
        (v13.0のまま)
        """
        
        evaluator_role = context.get('role', 'あなたは客観的で厳しい批評家です。')
        evaluation_guideline = context.get('evaluation_guideline', '提示された提案を、課題の要件に基づき厳密に評価してください。')

        return f"""
        # あなたの厳格な役割
        あなたは「{evaluator_role}」です。

        # あなたの最重要評価ガイドライン
        {evaluation_guideline}

        # 評価対象の課題
        {problem_statement}
        
        # 評価対象の提案 (v13.0)
        - 提案の核 (名称/創作物): {solution.get('proposal_main', '内容なし')}
        - 提案の詳細 (方法/理由): {solution.get('proposal_details', '詳細なし')}
        
        # タスク
        あなたの「役割」と「最重要評価ガイドライン」に厳密に従い、上記の「提案」を評価してください。
        ガイドラインに照らして、この提案が課題をどれだけ効果的に解決/達成できるか、または劣っているかを具体的に分析してください。

        # 出力形式 (JSON)
        {{
          "total_score": (0-100の整数),
          "strengths": "（{evaluator_role}の観点で優れている点）",
          "weaknesses": "（{evaluator_role}の観点で懸念・改善が必要な点）",
          "overall_comment": "（{evaluator_role}の観点での総括）"
        }}
        """

    # === ★v15.0: (v16.0でも変更なし) 個別調査情報を参照 ===
    def get_next_generation_prompt(self, elite_solutions: List[Dict], failed_solutions: List[Dict], problem_statement: str, num_solutions: int, context: Dict[str, Any]) -> str:
        """
        (v15.0) 既存の解を「進化」させ、新しい2分割JSONフォーマットで出力する。
        エージェント専用の `agent_research_insights` を参照する。
        """
        elite_text = "\n".join([f"- {s['solution'].get('proposal_main', 'N/A')} (スコア: {s['evaluation'].get('total_score', 0)})" for s in elite_solutions])
        failed_text = "\n".join([f"- {s['solution'].get('proposal_main', 'N/A')} (弱点: {s['evaluation'].get('weaknesses', 'N/A')})" for s in failed_solutions])

        # v15.0: エージェント個別の調査情報を取得
        insights = context.get('agent_research_insights', [])
        insights_text = "\n".join([f"- {item}" for item in insights]) if insights else "（追加の調査情報なし）"

        return f"""
        # 役割: {context.get('role', 'あなたは優れた戦略家であり編集者です。')}
        # 指示: {context.get('instructions', '高評価案の良い点を組み合わせ、低評価案の失敗から学び、新しい提案を生成してください。')}
        # タスク: 前世代の分析に基づき、次世代の新しい提案を{num_solutions}個生成してください。
        
        # 分析対象1：高評価だった提案（優れた遺伝子）: 
        {elite_text}
        # 分析対象2：低評価だった提案（学ぶべき教訓）: 
        {failed_text}

        # ★あなた専用の調査情報 (v15.0)★
        # 以下の個別の調査結果も**必ず**参考にして、提案を進化させてください。
        {insights_text}
        
        # 新しい提案の生成指示: {context.get('instructions')}
        
        # !!最重要!! (出力形式)
        各提案に「proposal_main」「proposal_details」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義 (v13.0)
        * **proposal_main**: 提案の「核」となる部分。 (例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。 (例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_main": "新しい提案1の核 (名称 または 創作物そのもの)", 
              "proposal_details": "新しい提案1の詳細 (内容、意図、背景、理由、効果など) を説明する2〜4行の文章です。"
            }}
          ] 
        }}
        """

    def get_revolutionary_generation_prompt(self, problem_statement: str, num_solutions: int, existing_roles: List[str]) -> str:
        """
        (v13.0のまま)
        突然変異用。個別の調査情報は参照しない。
        """
        
        existing_roles_list = "\n".join([f"- {role}" for role in existing_roles]) if existing_roles else "なし"

        return f"""
        # 役割: 
        あなたは「常識外れのイノベーター」を任命するマスタープランナーです。
        あなたは「突然変異」を引き起こすため、既存の提案や過去の評価（エリート解、失敗解）、
        および**既存のエージェント調査情報もすべて無視**します。

        # タスク:
        以下の「課題」に対し、既存のエージェントとは**全く異なる新しい観点**を持つ
        「革新的な専門家」を{num_solutions}人（または{num_solutions}個）定義し、
        その専門家の視点から、革新的な提案を{num_solutions}個生成してください。

        # 課題文: 
        {problem_statement}

        # 既存の専門家ロール (これらとは異なる視点にすること):
        {existing_roles_list}

        # !!重要!! 
        - ステップ1（内部思考）: 既存ロールがカバーしていない、全く新しい「役割（ロール）」を考案する。
        - ステップ2（内部思考）: その役割に基づき、革新的な提案（proposal_main, proposal_details）を考案する。
        - ステップ3（出力）: 考案した提案を、指定されたJSON形式で出力する。

        # !!最重要!! (出力形式)
        各提案に「proposal_main」「proposal_details」を必ず含め、JSON形式でリストとして出力してください。
        「proposal_main」には、考案した新しい専門家の役割や、その革新性が伝わるような名称/創作物を設定してください。

        # 出力項目の定義 (v13.0)
        * **proposal_main**: 提案の「核」となる部分。 (例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。 (例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_main": "（考案した新専門家の役割を反映した革新的な名称 または 創作物）", 
              "proposal_details": "（その提案の詳細 (内容、意図、背景、理由、効果など) を説明する2〜4行の文章です。）" 
            }}
          ] 
        }}
        """


# ----------------------------
# 4) EvoGenSolver (v15.0のまま変更なし)
# ----------------------------
class EvoGenSolver:
    """元の EvoGenSolver（主要ロジック）"""
    def __init__(self, llm_client: LLMClient, num_solutions_per_generation: int = 10):
        self.client = llm_client
        self.num_solutions = num_solutions_per_generation 
        self.prompter = PromptManager()
        self.history = []

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        return self.client.call(prompt) 

    def _generate_agent_personas(self, problem_statement: str) -> Dict:
        # (v13.0)
        prompt = self.prompter.get_agent_personas_prompt(problem_statement)
        return self._call_llm(prompt)

    def _generate_initial_solutions(self, problem_statement: str, context: List[Dict]) -> List[Dict[str, str]]:
        """
        (v15.0) `context` は `solver_agents` のリスト
        """
        initial_agent_list = context 
        if not isinstance(initial_agent_list, list) or len(initial_agent_list) == 0:
            st.warning(f"[EvoGenSolver] 解決・進化エージェントのリストが不正です。")
            return []
        
        num_initial_agents = len(initial_agent_list)
        st.info(f"💡 {num_initial_agents}体の専門エージェントが初期提案（10個）を分担して生成中...")
        
        all_solutions = []
        for i, agent_context in enumerate(initial_agent_list):
            st.caption(f"  - エージェント {i+1}/{num_initial_agents} ({agent_context.get('role', 'N/A')}) が生成中...")
            
            # (v15.0) `get_initial_generation_prompt` に `agent_context` (調査情報を含む) を渡す
            prompt = self.prompter.get_initial_generation_prompt(
                problem_statement, 
                1, 
                agent_context # 'role', 'instructions', 'agent_research_insights' が含まれる
            )
            response = self._call_llm(prompt) 
            
            if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list) and len(response["solutions"]) > 0:
                all_solutions.append(response["solutions"][0])
            else:
                st.warning(f"[EvoGenSolver] エージェント {i+1} が不正な形式を返しました。デバッグ情報: {response}")
                
        return all_solutions

    def _evaluate_solutions(self, solutions: List[Dict[str, str]], problem_statement: str, context: Dict) -> Generator[str | List[Dict], None, None]:
        # (v13.0のまま)
        evaluator_agent_list = context
        if not isinstance(evaluator_agent_list, list) or len(evaluator_agent_list) == 0:
            st.error("[EvoGenSolver] 評価エージェントのリストが不正です。処理を中断します。")
            yield []
            return

        evaluated_solutions = []
        if not solutions:
            yield []
            return

        num_evaluators = len(evaluator_agent_list)

        for i, solution in enumerate(solutions):
            if not isinstance(solution, dict) or "proposal_main" not in solution:
                yield f"  - 評価スキップ: 不正な形式の提案データです。"
                continue
            
            yield f"  - 評価中 {i+1}/{len(solutions)}: {solution.get('proposal_main', '名称不明')} ( {num_evaluators}体のエージェントによる評価)"

            individual_evaluations = []
            
            for j, eval_context in enumerate(evaluator_agent_list):
                yield f"    - 評価者 {j+1}/{num_evaluators} ({eval_context.get('role', 'N/A')}) が評価..."
                
                prompt = self.prompter.get_evaluation_prompt(solution, problem_statement, eval_context)
                evaluation = self._call_llm(prompt)
                
                if isinstance(evaluation, dict) and "total_score" in evaluation and "error" not in evaluation:
                    individual_evaluations.append(evaluation)
                else:
                    st.warning(f"[EvoGenSolver] 提案 '{solution.get('proposal_main', 'N/A')}' の評価者 {j+1} が不正な形式を返しました。デバッグ情報: {evaluation}")

            if not individual_evaluations:
                st.warning(f"[EvoGenSolver] 提案 '{solution.get('proposal_main', 'N/A')}' の有効な評価がありませんでした。")
                continue

            total_score_sum = sum(e.get('total_score', 0) for e in individual_evaluations)
            aggregated_score = round(total_score_sum / len(individual_evaluations))
            
            agg_strengths = "\n---\n".join([f"評価者{k+1} ({e.get('role', 'N/A')}):\n{e.get('strengths', 'N/A')}" for k, e in enumerate(individual_evaluations)])
            agg_weaknesses = "\n---\n".join([f"評価者{k+1} ({e.get('role', 'N/A')}):\n{e.get('weaknesses', 'N/A')}" for k, e in enumerate(individual_evaluations)])
            agg_comment = "\n---\n".join([f"評価者{k+1} ({e.get('role', 'N/A')}):\n{e.get('overall_comment', 'N/A')}" for k, e in enumerate(individual_evaluations)])

            aggregated_evaluation = {
                "total_score": aggregated_score,
                "strengths": agg_strengths,
                "weaknesses": agg_weaknesses,
                "overall_comment": agg_comment,
                "individual_evals": individual_evaluations 
            }
            
            evaluated_solutions.append({"solution": solution, "evaluation": aggregated_evaluation})
            yield f"    - 総合評価スコア: {aggregated_score}"


        evaluated_solutions.sort(key=lambda x: x.get("evaluation", {}).get("total_score", 0), reverse=True)
        yield evaluated_solutions

    def _generate_next_generation(self, evaluated_solutions: List[Dict], problem_statement: str, context: List[Dict]) -> List[Dict[str, str]]:
        """
        (v15.0) `context` は `solver_agents` のリスト
        """
        solver_agent_list = context 
        if not isinstance(solver_agent_list, list) or len(solver_agent_list) == 0:
            st.warning(f"[EvoGenSolver] 解決・進化エージェントのリストが不正です。")
            return []

        num_elites = max(1, int(len(evaluated_solutions) * 0.4))
        elite_solutions = evaluated_solutions[:num_elites]
        failed_solutions = evaluated_solutions[num_elites:]

        st.info(f"🚀 {self.num_solutions} 体の解決・進化エージェントを選出して次世代を生成...")

        new_solutions = []
        for i in range(self.num_solutions):
            
            if random.random() < 0.20:
                # 20%の確率: 革新 (突然変異)
                st.caption(f"  - ⚡ (突然変異) エージェント {i+1}/{self.num_solutions} が「新規エージェントの定義」と「革新的な提案」を実行...")
                
                existing_roles = [a.get('role', 'N/A') for a in solver_agent_list]
                
                # (v13.0) 調査情報は参照しない
                prompt = self.prompter.get_revolutionary_generation_prompt(
                    problem_statement, 
                    1, 
                    existing_roles 
                )
            else:
                # 80%の確率: 進化 (既存エージェントを再利用)
                selected_agent_context = random.choice(solver_agent_list) 
                st.caption(f"  - 🧬 (進化) エージェント {i+1}/{self.num_solutions} ({selected_agent_context.get('role', 'N/A')}) が「既存の提案」を進化...")
                
                # (v15.0) `get_next_generation_prompt` に `selected_agent_context` を渡す
                prompt = self.prompter.get_next_generation_prompt(
                    elite_solutions, 
                    failed_solutions, 
                    problem_statement, 
                    1, 
                    selected_agent_context # 'role', 'instructions', 'agent_research_insights' が含まれる
                )
            
            response = self._call_llm(prompt) 
            
            if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list) and len(response["solutions"]) > 0:
                new_solutions.append(response["solutions"][0])
            else:
                st.warning(f"[EvoGenSolver] エージェント {i+1} が不正な形式を返しました。デバッグ情報: {response}")

        return new_solutions

    # === (v15.0) リファクタリング済み ===

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        """
        (v15.0) ステップ1: スウォーム編成のみを行う。
        """
        self.history = []

        yield "--- 🧠 課題を分析し、最適なAIエージェント・スウォームを編成中... ---"
        agent_personas = self._generate_agent_personas(problem_statement) 

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["solver_agents", "evaluators", "output_labels"]):
            yield "エラー: チーム編成に失敗しました。処理を中断します。"
            yield f"**デバッグ情報:** AIからの応答が不正です。APIキーが正しいか確認してください。\n```\n{agent_personas}\n```"
            return

        yield f"--- ✔️ チーム編成完了 ---"
        yield {"agent_team": agent_personas} # `output_labels` もここに含まれる
        
        # (v15.0) 実行ロジックを `solve_internal` に移譲
        yield from self.solve_internal(problem_statement, agent_personas, generations)

    def solve_internal(self, problem_statement: str, agent_personas: Dict, generations: int) -> Generator[str | Dict, None, None]:
        """
        (v15.0) ステップ2: 提案の生成・評価・進化の「実行」サイクル。
        """
        if self.history: 
             pass
        else:
             self.history = []
             
        yield "\n--- 💡 Generation 0: 最初の提案 (10個) を生成中... ---"
        solutions = self._generate_initial_solutions(problem_statement, agent_personas["solver_agents"])
        
        if not solutions:
             yield "エラー: 最初の提案生成に失敗しました。AIが適切な応答を返さなかった可能性があります。処理を終了します。"
             return

        yield "--- 🧐 提案を評価中 (3エージェント x 10提案)... ---"
        eval_generator = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluators"])
        evaluated_solutions = []
        for item in eval_generator:
            if isinstance(item, str):
                yield item
            else:
                evaluated_solutions = item
        
        if not evaluated_solutions:
             yield "エラー: 提案の評価に失敗しました。処理を終了します。"
             return

        self.history.append({"generation": 0, "results": evaluated_solutions})
        yield self.history[-1]

        # G1以降の進化サイクル
        for i in range(1, generations):
            yield f"\n--- 🚀 Generation {i}: 次の提案へ進化中... ---"
            previous_generation_results = self.history[-1]["results"]
            
            if not previous_generation_results:
                yield f"エラー: 前世代 ({i-1}) の有効な評価結果がありません。進化を停止します。"
                break
            
            solutions = self._generate_next_generation(previous_generation_results, problem_statement, agent_personas["solver_agents"]) 

            if not solutions:
                yield f"エラー: Generation {i} の提案生成に失敗しました。AIが適切な応答を返さなかった可能性があります。処理を終了します。"
                break

            yield f"--- 🧐 Generation {i} の提案を評価中... ---"
            eval_generator_next = self._evaluate_solutions(solutions, problem_statement, agent_personas["evaluators"])
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


# ----------------------------
# 5) EvoGenSolver_Tavily (★v16.0: 修正箇所★)
# ----------------------------
class EvoGenSolver_Tavily(EvoGenSolver):
    """
    (v16.0) 
    1. 課題文の事前補強 (v14)
    2. スウォーム編成 (v13)
    3. ★エージェント個別調査 (v16: バッチクエリ)★
    4. 実行サイクル (v15)
    """
    def __init__(self, llm_client: LLMClient, tavily_client: TavilyClient, num_solutions_per_generation: int = 10, tavily_results_per_search: int = 5):
        super().__init__(llm_client, num_solutions_per_generation)
        self.tavily = tavily_client
        # (v15.0)
        self.tavily_results_per_agent_query = max(1, tavily_results_per_search // 2) 
        # (v15.0)
        self.tavily_results_for_augmentation = tavily_results_per_search 

    def _format_raw_content_for_llm(self, results: List[Dict[str, Any]], context_tag: str, max_items: int = 3, truncate_chars: int = 4000) -> str:
        """
        (v14.0のまま)
        """
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

    def _summarize_multi_phase_results_with_llm(
        self, 
        problem_statement: str, 
        analysis_results: List[Dict[str, Any]], 
        solution_results: List[Dict[str, Any]]
    ) -> str:
        """
        (v14.0のまま) 課題文の「事前補強」用
        """
        
        if not analysis_results and not solution_results:
            return problem_statement

        analysis_content_text = self._format_raw_content_for_llm(
            analysis_results, 
            "ANALYSIS CONTENT", 
            max_items=3, 
            truncate_chars=4000
        )
        solution_content_text = self._format_raw_content_for_llm(
            solution_results, 
            "SOLUTION CONTENT", 
            max_items=3, 
            truncate_chars=4000
        )

        # v14.0の深層分析プロンプト
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
        **スニペット（抜粋）ではなく、提供された全文コンテンツを深く分析してください。**

        # 出力形式 (JSON)
        分析結果を以下のJSON形式で出力してください。
        {{
          "summary_analysis": "「調査資料1（現状・背景）」を深く分析した*戦略的洞察*。単なる要約ではなく、課題の背景にある重要な文脈や制約を指摘する。(1〜3文)",
          "summary_solution": "「調査資料2（解決策・事例）」から抽出した*重要な傾向*。他社の事例や新技術から学べる、課題解決のヒントを指摘する。(1〜3文)",
          "key_points": [
            "「調査資料1（現状・背景）」においてその他、考慮するべきと思われる複数の観点に関する簡潔な1文を10個程度作成する",
            "「調査資料2（解決策・事例）」においてその他、考慮するべきと思われる数の観点に関する簡潔な1文を10個程度作成する",
          ]
        }}
        """
        
        llm_ret = self._call_llm(prompt) 
        
        if isinstance(llm_ret, dict) and any(k in llm_ret for k in ["summary_analysis", "summary_solution", "key_points"]):
            try:
                summary_analysis_text = llm_ret.get("summary_analysis", "現状分析の要約なし")
                summary_solution_text = llm_ret.get("summary_solution", "解決策事例の要約なし")
                kp = llm_ret.get("key_points", [])
                
                # v14.0 (gen_ai_03.py) の `top_sources` をパースする部分が欠落していたため
                # v14.0 のコードを復元・修正 (v15.0 で欠落していた)
                top = llm_ret.get("top_sources", [])
                top_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in top]) if isinstance(top, list) else ""
                
                composed = f"""
## Tavilyリサーチ要約（LLMによる詳細分析）
### 現状・背景分析 (戦略的洞察)
{summary_analysis_text}
### 解決策・事例 (重要な傾向)
{summary_solution_text}

### 抽出された重要点
""" + "\n".join([f"- {p}" for p in kp]) + "\n\n" + \
"### 主な出典\n" + top_text + "\n\n" + \
"--- (以下、元の課題文) ---\n" + problem_statement
                
                return composed
            except Exception:
                pass 

        # (v13.0互換) フォールバックロジック
        fallback_sources = []
        for r in analysis_results[:2]:
            fallback_sources.append(f"- [分析] {r.get('title','No title')} ({r.get('url','')})")
        for r in solution_results[:2]:
            fallback_sources.append(f"- [解決策] {r.get('title','No title')} ({r.get('url','')})")
            
        fallback = "## Tavilyリサーチ要約（フォールバック）\n" + \
                   "最新のウェブ情報を参照しました。上位出典:\n" + "\n".join(fallback_sources) + \
                   "\n\n" + "--- (以下、元の課題文) ---\n" + problem_statement
        return fallback

    # === ★v16.0: 修正 (バッチクエリ生成ロジック) ===
    def _run_agent_specific_research(self, problem_statement: str, solver_agents: List[Dict]) -> Generator[str, None, List[Dict]]:
        """
        (v16.0) エージェント個別調査を実行。
        1. (LLM x1) 全エージェントのクエリをバッチ生成
        2. (Loop x10) [Tavily検索 -> LLM分析] を実行
        """
        if not solver_agents:
            yield "警告: 解決エージェントが定義されていないため、個別調査をスキップします。"
            return []
            
        yield f"--- 🤖 10体の解決エージェントの専用調査クエリをバッチ生成中... ---"
        
        # 1. (v16.0) 全エージェントのクエリを1回のLLM呼び出しで生成
        all_queries_prompt = self.prompter.get_all_agent_queries_prompt(problem_statement, solver_agents)
        all_queries_response = self._call_llm(all_queries_prompt)
        
        all_queries_dict = {}
        if isinstance(all_queries_response, dict) and "agent_queries" in all_queries_response:
            all_queries_dict = all_queries_response["agent_queries"]
        else:
            yield f"  - 警告: 全エージェントのクエリ一括生成に失敗。個別調査をスキップします。 (Debug: {all_queries_response})"
            return solver_agents # 調査情報なしで元のリストを返す
            
        yield f"--- ✔️ クエリバッチ生成完了。10体のエージェントが個別の深層リサーチを開始... ---"
        
        updated_agents = []
        num_agents = len(solver_agents)

        # 2. (v16.0) 各エージェントが「検索」と「分析」を実行
        for i, agent_context in enumerate(solver_agents):
            role = agent_context.get("role", "不明な役割")
            instructions = agent_context.get("instructions", "")
            
            # (v16.0) LLMを呼び出す代わりに、辞書からクエリを取得
            queries = all_queries_dict.get(role, [])
            
            if not queries:
                yield f"  - {i+1}/{num_agents}: 「{role}」 はクエリを取得できませんでした。調査をスキップ。"
                updated_agents.append(agent_context) # 調査情報なしで追加
                continue

            # (v15.0のまま) クエリでTavily検索（全文取得）を実行
            yield f"  - {i+1}/{num_agents}: 「{role}」 が調査を実行中 (クエリ: {', '.join(queries)})..."
            agent_search_results = []
            for q in queries:
                if not q.strip(): continue
                tavily_resp = self.tavily.search(q, num_results=self.tavily_results_per_agent_query)
                if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                    agent_search_results.extend(tavily_resp["results"])
                elif isinstance(tavily_resp, dict) and "error" in tavily_resp:
                     yield f"  - Tavily エラー (エージェントクエリ: {q}): {tavily_resp['error']}"

            if not agent_search_results:
                yield f"  - 警告: 「{role}」 は調査結果を得られませんでした。調査をスキップ。"
                updated_agents.append(agent_context) # 調査情報なしで追加
                continue

            # (v15.0のまま) 全文コンテンツを整形
            raw_content_text = self._format_raw_content_for_llm(
                agent_search_results,
                f"AGENT {i+1} RESEARCH",
                max_items=self.tavily_results_per_agent_query * 2, # クエリ2個分
                truncate_chars=3000 
            )

            # (v15.0のまま) 全文をLLMで分析し、箇条書きの洞察を抽出
            yield f"  - {i+1}/{num_agents}: 「{role}」 が調査結果（全文）を分析し、洞察を抽出中..."
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
            else:
                yield f"  - 警告: 「{role}」 の分析に失敗。 (Debug: {analysis_response})"
            
            # (v15.0のまま) エージェントの辞書に調査結果 (`agent_research_insights`) を注入
            agent_context["agent_research_insights"] = insights
            updated_agents.append(agent_context)
            yield f"  - {i+1}/{num_agents}: 「{role}」 が {len(insights)} 個の個別洞察を獲得。"

        yield f"--- ✔️ 全エージェントの個別調査が完了 ---"
        return updated_agents # 調査情報が注入されたエージェントリストを返す


    # === ★v16.0: 修正 (v15.0 のバグ修正) ===
    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        """
        (v16.0) Tavily版のフルプロセスを実行
        1. 課題文の事前補強 (v14)
        2. スウォーム編成 (v13)
        3. ★エージェント個別調査 (v16)★
        4. 実行サイクル (v15)
        """
        self.history = []

        # --- ステップ1: 課題文の事前補強 (v14.0ロジック) ---
        yield "--- 💡 課題文補強のため、LLMが最適な検索クエリ（フェーズ1 & 2）を生成中... ---"
        prompt = self.prompter.get_tavily_multi_phase_query_prompt(problem_statement)
        query_response = self._call_llm(prompt)

        augmented_problem = problem_statement 

        if not isinstance(query_response, dict) or ("analysis_queries" not in query_response and "solution_queries" not in query_response):
            yield f"エラー: Tavilyクエリの生成に失敗しました。AIからの応答が不正です: {query_response}"
        else:
            analysis_queries = query_response.get("analysis_queries", [])
            solution_queries = query_response.get("solution_queries", [])
            
            yield f"--- ✔️ 課題補強用クエリ生成完了 ---"
            
            analysis_results_list = []
            solution_results_list = []
            
            if analysis_queries:
                yield "--- 🌐 (課題補強) フェーズ1: 現状分析リサーチ (全文取得) を開始... ---"
                for q in analysis_queries:
                    if not q.strip(): continue
                    yield f"  - 検索中 (分析): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_for_augmentation)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        analysis_results_list.extend(tavily_resp["results"])
            
            if solution_queries:
                yield "--- 🌐 (課題補強) フェーズ2: 解決策事例リサーチ (全文取得) を開始... ---"
                for q in solution_queries:
                    if not q.strip(): continue
                    yield f"  - 検索中 (解決策): {q}"
                    tavily_resp = self.tavily.search(q, num_results=self.tavily_results_for_augmentation)
                    if isinstance(tavily_resp, dict) and "results" in tavily_resp:
                        solution_results_list.extend(tavily_resp["results"])

            yield {"tavily_info_analysis": analysis_results_list, "tavily_info_solution": solution_results_list}

            yield "--- ✍️ (課題補強) Webページ全文をLLMが深層分析し、問題文に統合します... ---"
            try:
                augmented_problem = self._summarize_multi_phase_results_with_llm(
                    problem_statement, 
                    analysis_results_list, 
                    solution_results_list
                )
            except Exception as e:
                yield f"警告: Tavily 深層分析中にエラーが発生しました: {e}"
        
        yield {"augmented_problem": augmented_problem}

        # --- ステップ2: スウォーム編成 (v13.0ロジック) ---
        yield "--- 🧠 補強された課題を分析し、最適なAIエージェント・スウォームを編成中... ---"
        agent_personas = self._generate_agent_personas(augmented_problem) 

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["solver_agents", "evaluators", "output_labels"]):
            yield "エラー: チーム編成に失敗しました。処理を中断します。"
            yield f"**デバッグ情報:** AIからの応答が不正です。APIキーが正しいか確認してください。\n```\n{agent_personas}\n```"
            return

        yield f"--- ✔️ チーム編成完了 ---"
        yield {"agent_team": agent_personas} 

        # --- ステップ3: ★エージェント個別調査 (v16.0 新ロジック)★ ---
        
        # ★v16.0 修正: v15.0 のジェネレータ処理のバグを修正。
        # `yield from` を使い、ジェネレータの `return` 値を正しく受け取る。
        updated_agents_list = yield from self._run_agent_specific_research(
            augmented_problem, 
            agent_personas["solver_agents"]
        )

        if not updated_agents_list:
             yield "警告: エージェントの個別調査で問題が発生しました。調査情報なしで続行します。"
             updated_agents_list = agent_personas["solver_agents"] # 元のリストで続行

        # 調査情報が注入されたエージェントリストで `agent_personas` を上書き
        agent_personas["solver_agents"] = updated_agents_list
        
        # (v15.0)「調査情報が追加された」完全なチーム情報をUIに再送信
        yield {"agent_team_updated": agent_personas}


        # --- ステップ4: 実行サイクル (v15.0) ---
        yield from super().solve_internal(augmented_problem, agent_personas, generations)


# ----------------------------
# 6) Streamlit UI (v15.0のまま変更なし)
# ----------------------------
st.set_page_config(page_title="EvoGen AI + Tavily (Agent Research)", layout="wide")
st.title("EvoGen AI 🧬")
st.markdown("進化型生成AI解探索フレームワーク (v16.0: バッチクエリ最適化モデル)")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_key = st.text_input("Google Gemini API Key", type="password", help="Gemini の API キーを入力してください（保存されません）。")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Tavily の API キーを入力してください（保存されません）。")
    st.subheader("パラメータ")
    num_generations = st.slider("世代数", 1, 20, 2, help="提案を進化させる回数です。")
    num_solutions = st.slider("世代ごとの(最大)提案の数", 3, 10, 10, help="第1世代以降に生成・評価する提案の数です。(第0世代は常に10個)")
    tavily_results_per_search = st.slider(
        "Tavily 検索結果数 (クエリ毎)", 1, 10, 4, 
        help="""
        課題補強フェーズ: この数だけ検索します (例: 4件)。\n
        エージェント個別調査フェーズ: この数を2（クエリ数）で割った数を、クエリごとに検索します (例: 4なら2件ずつ)。
        """
    ) 
    st.markdown("---")
    st.info("Tavily を使って課題に関連する**Webページの全文**を取得し、LLMが**詳細分析**した上で課題文を補強します。")

# (v13.0互換)
default_problem = """
# 課題
中小企業の経理部門における、請求書処理の業務効率を劇的に改善する
新しいAIソリューションを提案せよ。

# 要件・制約条件
- 導入コストが低いこと。（月額5万円以下）
- 専門的なIT知識がなくても利用できること。
- 既存の会計ソフト（例: freee, MFクラウド）と連携できることが望ましい。
"""
problem_statement = st.text_area("解決したい課題（または創作したいお題）を入力してください", value=default_problem, height=260, help="例: 「中小企業の請求書処理を改善するAIソリューションを提案せよ」 や 「『春』をテーマにした斬新な俳句を5つ考えて」")

# (v13.0互換)
if st.button("提案の生成を開始", type="primary"):
    if not gemini_key:
        st.error("サイドバーでGoogle Gemini APIキーを入力してください。")
    elif not tavily_key:
        st.error("サイドバーでTavily APIキーを入力してください。")
    elif not problem_statement.strip():
        st.warning("課題を入力してください。")
    else:
        # (v13.0)
        default_labels = {"main_label": "提案 (名称/創作物)", "details_label": "詳細 (内容/理由)"}
        st.session_state.output_labels = default_labels
        
        status_placeholder = st.empty()
        team_placeholder = st.empty()
        augmented_problem_placeholder = st.container() 
        tavily_placeholder = st.container() 
        results_area = st.container()
        final_result_placeholder = st.container()

        with st.spinner("🌀 AIが思考中です... (Webページの全文分析を含むため時間がかかる場合があります)"):
            try:
                gemini_client = GeminiClient(api_key=gemini_key)
                tavily_client = TavilyClient(api_key=tavily_key)
            except Exception as e:
                st.error(f"クライアントの初期化に失敗しました: {e}")
                st.stop()

            # ★v16.0: ここで `EvoGenSolver_Tavily` がインスタンス化される
            solver = EvoGenSolver_Tavily(
                llm_client=gemini_client,
                tavily_client=tavily_client,
                num_solutions_per_generation=num_solutions,
                tavily_results_per_search=tavily_results_per_search
            )
            
            # (v14.0互換)
            def display_tavily_results(results_list, title):
                with tavily_placeholder.container():
                    st.subheader(title)
                    if results_list:
                        for r in results_list:
                            title = r.get("title", "No title")
                            url = r.get("url", "")
                            st.markdown(f"- [{title}]({url})")
                    else:
                        st.write("このフェーズでは検索結果がありませんでした。")
                    st.markdown("---")


            # --- Solverを実行し、結果をUIにストリーミング表示 ---
            for result in solver.solve(problem_statement, generations=num_generations):
                if isinstance(result, str):
                    status_placeholder.info(result) 

                # (v14.0互換)
                elif isinstance(result, dict) and ("tavily_info_analysis" in result or "tavily_info_solution" in result):
                    tavily_placeholder.empty()
                    analysis_data = result.get("tavily_info_analysis", [])
                    solution_data = result.get("tavily_info_solution", [])
                    if analysis_data:
                        display_tavily_results(analysis_data, "🌐 (課題補強) フェーズ1: 現状分析リサーチ結果")
                    if solution_data:
                        display_tavily_results(solution_data, "🌐 (課題補強) フェーズ2: 解決策事例リサーチ結果")
                
                # (v14.0互換)
                elif isinstance(result, dict) and "augmented_problem" in result:
                    with augmented_problem_placeholder.container():
                        st.subheader("🔍 リサーチ結果で補強された課題文 (LLM詳細分析)")
                        with st.expander("補強された課題文の詳細を表示", expanded=False): 
                            st.markdown(result["augmented_problem"])
                        st.markdown("---")
                
                # (v15.0互換)
                elif isinstance(result, dict) and ("agent_team" in result or "agent_team_updated" in result):
                    
                    team_data_key = "agent_team_updated" if "agent_team_updated" in result else "agent_team"
                    team = result[team_data_key]

                    if "output_labels" in team:
                        st.session_state.output_labels = team["output_labels"]
                    
                    with team_placeholder.container():
                        st.subheader("🤖 編成されたAIエージェント・スウォーム")
                        
                        labels_to_show = st.session_state.output_labels
                        st.markdown(f"**成果物ラベル:** `{labels_to_show.get('main_label')}` / `{labels_to_show.get('details_label')}`")

                        is_updated = (team_data_key == "agent_team_updated")
                        with st.expander("チームの詳細を表示", expanded=is_updated):
                            st.markdown("##### 💡🧬 解決・進化担当 (10体)")
                            gen_list = team.get("solver_agents", [])
                            if gen_list:
                                for i, gen in enumerate(gen_list):
                                    st.markdown(f"**{i+1}. {gen.get('role', '未定義')}**")
                                    st.caption(f"指示: {gen.get('instructions', '未定義')}")
                                    
                                    # (v15.0) 個別調査情報を表示
                                    insights = gen.get("agent_research_insights")
                                    if insights:
                                        with st.container(border=True):
                                            st.markdown(f"**個別の調査情報 (洞察):**")
                                            insights_md = "\n".join([f"  - {item}" for item in insights])
                                            st.markdown(insights_md)
                                    elif is_updated:
                                        st.caption("（このエージェントは個別調査に失敗、または結果ゼロ）")
                            
                            st.markdown("---")
                            st.markdown("##### 🧐 評価担当 (3体)") 
                            eva_list = team.get("evaluators", [])
                            if eva_list:
                                for i, eva in enumerate(eva_list):
                                    st.markdown(f"**{i+1}. {eva.get('role', 'N/A')}**")
                                    guideline = eva.get('evaluation_guideline', '評価ガイドライン未定義')
                                    st.caption(f"ガイドライン: {guideline}")

                # (v13.0互換)
                elif isinstance(result, dict) and "generation" in result:
                    labels = st.session_state.output_labels
                    
                    gen_data = result
                    with results_area.container():
                        st.subheader(f"第 {gen_data['generation']} 世代の結果")
                        with st.container(border=True):
                            if not gen_data.get('results'):
                                st.write("この世代では有効な提案が生成されませんでした。")
                                continue
                            
                            for item in gen_data.get('results', []):
                                sol = item.get('solution', {})
                                eva = item.get('evaluation', {})
                                score = eva.get('total_score', 0)
                                
                                st.markdown(f"**{labels.get('main_label', '提案')}:** {sol.get('proposal_main', 'N/A')} (スコア: {score})")
                                st.markdown(f"**{labels.get('details_label', '詳細')}:**\n {sol.get('proposal_details', 'N/A')}")
                                
                                if item != gen_data.get('results', [])[-1]:
                                    st.markdown("---")

        # (v13.0互換)
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
            
            labels = st.session_state.output_labels

            status_placeholder.empty()
            st.balloons()

            with final_result_placeholder:
                st.success("🏆 処理完了！スコアトップ5の提案はこちらです。")
                
                for i, item in enumerate(top_5_solutions):
                    sol = item.get('solution', {})
                    eva = item.get('evaluation', {})
                    score = eva.get('total_score', 'N/A')
                    
                    st.header(f"🏅 第 {i + 1} 位")
                    st.metric(label="最終スコア (3エージェント平均)", value=f"{score}")

                    st.info(f"**{labels.get('main_label', '提案')}**\n\n{sol.get('proposal_main', 'N/A')}")
                    st.info(f"**{labels.get('details_label', '詳細')}**\n\n{sol.get('proposal_details', 'N/A')}")

                    # 評価を表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**優れた点 (3名の評価者より)**")
                        st.text_area(
                            f"優れた点 {i+1}", 
                            value=eva.get('strengths', 'N/A'), 
                            height=250, 
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    with col2:
                        st.warning(f"**懸念点・改善点 (3名の評価者より)**")
                        st.text_area(
                            f"懸念点・改善点 {i+1}", 
                            value=eva.get('weaknesses', 'N/A'), 
                            height=250, 
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    
                    st.info(f"**総評 (3名の評価者より)**")
                    st.text_area(
                        f"総評 {i+1}",
                        value=eva.get('overall_comment', 'N/A'),
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    st.markdown("---")
        else:
            status_placeholder.warning("処理が完了しましたが、最終的な提案は見つかりませんでした。")