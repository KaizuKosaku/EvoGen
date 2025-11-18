# app_tavily_fixed_v12_universal_proposal.py
"""
EvoGen AI with Tavily integration (v12.0: 汎用提案モデル版)

v11.0 (app_tavily9.py) からの変更点:
- (汎用性向上) ユーザーの要望に基づき、あらゆる問題に対応できるようデータ構造を
  「解決策(solution)」から「提案(proposal)」に変更。
- (データ構造)
  - 従来の `{"name": "...", "summary": "...", "specific_method": "..."}` を廃止。
  - 新たに `{"proposal_title": "...", "proposal_content": "...", "proposal_rationale": "..."}` を採用。
    - `proposal_title`: 提案の名称 (例: AIソリューション名、俳句の題名)
    - `proposal_content`: 提案の核 (例: ソリューション概要、俳句そのもの)
    - `proposal_rationale`: 提案の理由 (例: 具体的な方法論、俳句の狙いや背景)
- (PromptManager 修正)
  - `get_initial_generation_prompt`, `get_next_generation_prompt`, 
    `get_revolutionary_generation_prompt` が新しい汎用JSON構造を
    生成するようにプロンプトを全面改修。
  - `get_evaluation_prompt` が新しい汎用JSON構造を評価対象として
    受け取るように修正。
- (EvoGenSolver 修正)
  - `_evaluate_solutions` 内のログ参照を `name` から `proposal_title` に変更。
- (Streamlit UI 修正)
  - 世代別および最終結果の表示を、新しい汎用データ構造
    (proposal_title, proposal_content, proposal_rationale) を
    表示するようにレイアウト変更。

v11.0 の特徴 (v10.1から変更):
- (評価精度向上) 評価エージェントが「評価ガイドライン(evaluation_guideline)」を
  動的に生成し、それに基づき高精度な評価を行う。
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
# 1) LLMクライアント層 (変更なし)
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
# 2) Tavily クライアント (変更なし)
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
# 3) PromptManager (★修正箇所★)
# ----------------------------
class PromptManager:
    """AIへの指示書（プロンプト）を管理するクラス"""
    
    def get_tavily_multi_phase_query_prompt(self, problem_statement: str) -> str:
        """
        (v10.0のまま)
        課題解決に必要な情報を「分析」と「解決策」の2フェーズで検索するための
        クエリをLLMに生成させるプロンプト。(クエリ数4)
        """
        return f"""
        あなたは、提示された「課題」を解決するための調査を2段階で行う専門の調査員です。

        以下の「課題」を分析し、2つのフェーズに対応する**日本語の検索クエリ**をそれぞれ4つずつ生成してください。

        # フェーズ1: 現状・背景分析
        課題文に含まれる固有名詞（組織名、地名、特定のシステム名など）を特定し、
        その対象の「最新情報」「現状のデータ」「関連する背景や制約」を調査するためのクエリ。
        (例: 「システムX 最新バージョン情報」, 「組織Yの現状の戦略」)

        # フェーズ2: 解決策の事例・技術調査
        課題そのものを解決するための「最新の対策事例」「関連する新しい技術の動向」「他分野での成功事例」を調査するためのクエリ。
        (例: 「データベース パフォーマンス改善 事例」, 「BtoBマーケティング 最新手法」)

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
        (v11.0のまま)
        あらゆる課題を分析し、専門特化した「解決エージェント」と、
        課題に応じて最適化された「評価エージェント」（役割＋評価ガイドライン）をゼロから生成する。
        """
        return f"""
        # 役割
        あなたは、非常に複雑な課題を解決するために、AIエージェントからなる「スウォーム（群れ）」を編成する「マスタープランナー」です。

        # タスク
        以下の「課題」を解決するために、最も効果的なAIエージェント群を編成してください。
        編成は以下のステップで厳密に行ってください。

        ## ステップ1: 課題の徹底分析 (Your Internal Monologue)
        (このステップは出力に含めず、エージェント定義のために内部で実行してください)
        1.  **核心的目標(Goal)は何か？**: 課題文が最終的に達成したい状態は何か？ (例: 「収益を20%向上させる」「システムの応答速度を50ms短縮する」「ブランド認知度を高める」)
        2.  **主要な制約(Constraints)は何か？**: 課題文に明記されている、あるいは暗黙的に含まれる制約は？ (例: 「予算100万円以内」「3ヶ月以内に実装」「既存のAシステムと連携必須」「法的規制の遵守」)
        3.  **主要な利害関係者(Stakeholders)は誰か？**: この課題の影響を受けるのは誰か？ (例: 「中小企業の経理担当者」「アプリの新規ユーザー」「研究室のメンバー」「社会全体」)
        4.  **課題のボトルネックは何か？**: なぜ今、この目標が達成できていないのか？

        ## ステップ2: 解決・進化担当エージェント (10体) の定義
        - ステップ1の分析に基づき、課題解決に最適化された「互いに異なる10の視点」を持つ専門家（solver_agents）を定義してください。
        - **重要**: 「マーケター」のような一般的な役割ではなく、「**[利害関係者]の[特定の課題]を解決する専門家**」や「**[主要な制約]をクリアする[特定技術]の専門家**」のように、**この課題専用に特化させた役割（role）**を定義してください。
        - `instructions`には、その専門性を活かして「初期解の生成」と「既存解の進化」の両方でどう振る舞うべきか具体的に指示してください。

        ## ステップ3: 課題特化型 評価エージェント (3体) の定義
        - ステップ1の分析（核心的目標、主要な制約、利害関係者）に基づき、生成された解決策を評価するために**最も重要となる3つの異なる評価観点**を特定してください。
        - その3つの観点に基づき、それぞれ専門の評価エージェント（evaluators）を3体定義してください。
        - **!!最重要!!**: 「インパクト」「実現性」「リスク」のような固定的な役割に**縛られないでください**。**この課題を評価するためだけに最適化された役割（role）**をゼロから考案してください。
        - (例: 「俳句を作成する」課題の場合、「芸術性・情景描写 評価者」「革新性・季語解釈 評価者」「読者の心理的効果 評価者」など。)
        - (例: 「ビジネスAIを提案する」課題の場合、「ROI・収益性 評価者」「技術的実現性・運用 評価者」「UX・顧客受容性 評価者」など。)
        - `role`: あなたが考案した、課題に特化した評価者の役割名。
        - `evaluation_guideline`: (★v11.0のまま★) その役割が解決策を厳密に評価するために使用する、**具体的かつ詳細な評価指針（ガイドライン）**。このガイドラインには、何を最重要視し、どのような観点で優劣を判断すべきかを明確に記述してください。

        # 課題
        {problem_statement}

        # 出力形式 (JSON)
        {{
          "solver_agents": [
            {{ "role": "（ステップ2で定義した専門的役割1）", "instructions": "..." }},
            {{ "role": "（ステップ2で定義した専門的役割2）", "instructions": "..." }},
            // ... (10体分)
            {{ "role": "（ステップ2で定義した専門的役割10）", "instructions": "..." }}
          ],
          "evaluators": [
            // 評価者1: あなたが考案した課題特化の役割
            {{ 
              "role": "（ステップ3で考案した評価役割1）", 
              "evaluation_guideline": "（その役割のための具体的かつ詳細な評価ガイドライン。何を最重要視し、どう判断すべきか。）"
            }},
            // 評価者2: あなたが考案した課題特化の役割
            {{ 
              "role": "（ステップ3で考案した評価役割2）", 
              "evaluation_guideline": "（その役割のための具体的かつ詳細な評価ガイドライン。何を最重要視し、どう判断すべきか。）"
            }},
            // 評価者3: あなたが考案した課題特化の役割
            {{ 
              "role": "（ステップ3で考案した評価役割3）", 
              "evaluation_guideline": "（その役割のための具体的かつ詳細な評価ガイドライン。何を最重要視し、どう判断すべきか。）"
            }}
          ]
        }}
        """

    # === ★v12.0: 修正点 1 (汎用提案フォーマットの指示) ===
    def get_initial_generation_prompt(self, problem_statement: str, num_solutions: int, context: Dict[str, str]) -> str:
        """
        (★v12.0: 汎用提案モデル版★)
        「解決策」にも「創作物」にも対応できる汎用的な
        `proposal_title`, `proposal_content`, `proposal_rationale`
        を生成させる。
        """
        return f"""
        # 役割: {context.get('role', 'あなたは一流のイノベーターです。')}
        # 指示: {context.get('instructions', f'以下の課題に対し、互いに全く異なるアプローチからの提案を{num_solutions}個生成してください。')}
        # 課題文: {problem_statement}

        # !!最重要!! (出力形式)
        各提案に「proposal_title」「proposal_content」「proposal_rationale」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義
        * **proposal_title**: 提案の簡潔な名称 (例: 「AI請求書ソリューション」, 「春風の俳句」, 「新商品のキャッチコピーA」)
        * **proposal_content**: 提案の「核」となる内容。
            * (解決策の場合): 提案の概要を記述してください。
            * (創作物の場合): 俳句、キャッチコピー、名前などの「創作物そのもの」を記述してください。
        * **proposal_rationale**: 提案の「理由」や「方法」。
            * (解決策の場合): 「具体的な方法」やメカニズム、その理由を2〜4行で説明してください。
            * (創作物の場合): 「その創作物の狙いや効果、背景、理由」を2〜4行で説明してください。
        * **重要**: 「proposal_rationale」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_title": "提案1の名称", 
              "proposal_content": "提案1の核となる内容 (概要や創作物そのもの)", 
              "proposal_rationale": "提案1の具体的な方法、または狙いや理由を説明する2〜4行の文章です。\nこのように改行を含めても構いません。"
            }}
          ] 
        }}
        """

    # === ★v12.0: 修正点 2 (汎用提案フォーマットの評価) ===
    def get_evaluation_prompt(self, solution: Dict[str, str], problem_statement: str, context: Dict[str, Any]) -> str:
        """
        (★v12.0: 汎用提案モデル版★)
        AIが生成した「役割」と「評価ガイドライン」に基づき、
        新しい汎用提案フォーマットを評価するプロンプトを構築する。
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
        
        # 評価対象の提案 (★v12.0 修正箇所)
        - 名称/タイトル: {solution.get('proposal_title', '名称不明')}
        - 提案内容 (概要/創作物): {solution.get('proposal_content', '内容なし')}
        - 具体的な方法/理由: {solution.get('proposal_rationale', '具体的な方法/理由なし')}
        
        # タスク
        あなたの「役割」と「最重要評価ガイドライン」に厳密に従い、上記の「提案」を評価してください。
        ガイドラインに照らして、この提案が課題をどれだけ効果的に解決/達成できるか、または劣っているかを具体的に分析してください。

        # 出力形式 (JSON)
        以下の形式で、評価結果をJSONで厳密に出力してください。
        - **total_score**: あなたのガイドラインに基づいた総合評価点 (0〜100点の整数)。
        - **strengths**: あなたのガイドラインの観点で、特に優れている点。（簡潔に）
        - **weaknesses**: あなたのガイドラインの観点で、懸念・改善が必要な点。（簡潔に）
        - **overall_comment**: 評価の総括。（簡潔に）

        {{
          "total_score": (0-100の整数),
          "strengths": "（{evaluator_role}の観点で優れている点）",
          "weaknesses": "（{evaluator_role}の観点で懸念・改善が必要な点）",
          "overall_comment": "（{evaluator_role}の観点での総括）"
        }}
        """

    # === ★v12.0: 修正点 3 (汎用提案フォーマットの進化) ===
    def get_next_generation_prompt(self, elite_solutions: List[Dict], failed_solutions: List[Dict], problem_statement: str, num_solutions: int, context: Dict[str, str]) -> str:
        """
        (★v12.0: 汎用提案モデル版★)
        既存の解を「進化」させ、新しい汎用提案フォーマットで出力する。
        """
        # ★v12.0 修正: 'name' -> 'proposal_title'
        elite_text = "\n".join([f"- {s['solution'].get('proposal_title', 'N/A')} (スコア: {s['evaluation'].get('total_score', 0)})" for s in elite_solutions])
        failed_text = "\n".join([f"- {s['solution'].get('proposal_title', 'N/A')} (弱点: {s['evaluation'].get('weaknesses', 'N/A')})" for s in failed_solutions])

        return f"""
        # 役割: {context.get('role', 'あなたは優れた戦略家であり編集者です。')}
        # 指示: {context.get('instructions', '高評価案の良い点を組み合わせ、低評価案の失敗から学び、新しい提案を生成してください。')}
        # タスク: 前世代の分析に基づき、次世代の新しい提案を{num_solutions}個生成してください。
        # 分析対象1：高評価だった提案（優れた遺伝子）: 
        {elite_text}
        # 分析対象2：低評価だった提案（学ぶべき教訓）: 
        {failed_text}
        # 新しい提案の生成指示: {context.get('instructions')}
        
        # !!最重要!! (出力形式)
        各提案に「proposal_title」「proposal_content」「proposal_rationale」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義
        * **proposal_title**: 提案の簡潔な名称 (例: 「AI請求書ソリューション」, 「春風の俳句」, 「新商品のキャッチコピーA」)
        * **proposal_content**: 提案の「核」となる内容。
            * (解決策の場合): 提案の概要を記述してください。
            * (創作物の場合): 俳句、キャッチコピー、名前などの「創作物そのもの」を記述してください。
        * **proposal_rationale**: 提案の「理由」や「方法」。
            * (解決策の場合): 「具体的な方法」やメカニズム、その理由を2〜4行で説明してください。
            * (創作物の場合): 「その創作物の狙いや効果、背景、理由」を2〜4行で説明してください。
        * **重要**: 「proposal_rationale」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_title": "新しい提案1の名称", 
              "proposal_content": "新しい提案1の核となる内容 (概要や創作物そのもの)", 
              "proposal_rationale": "新しい提案1の具体的な方法、または狙いや理由を説明する2〜4行の文章です。\nこのように改行を含めても構いません。"
            }}
          ] 
        }}
        """

    # === ★v12.0: 修正点 4 (汎用提案フォーマットの革新) ===
    def get_revolutionary_generation_prompt(self, problem_statement: str, num_solutions: int, existing_roles: List[str]) -> str:
        """
        (★v12.0: 汎用提案モデル版★)
        全く新しい「革新的なエージェント」を定義させ、
        新しい汎用提案フォーマットで出力させる。
        """
        
        existing_roles_list = "\n".join([f"- {role}" for role in existing_roles]) if existing_roles else "なし"

        return f"""
        # 役割: 
        あなたは「常識外れのイノベーター」を任命するマスタープランナーです。
        あなたは「突然変異」を引き起こすため、既存の提案や過去の評価（エリート解、失敗解）は**完全に無視**します。

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
        - ステップ2（内部思考）: その役割に基づき、革新的な提案（proposal_title, proposal_content, proposal_rationale）を考案する。
        - ステップ3（出力）: 考案した提案を、指定されたJSON形式で出力する。

        # !!最重要!! (出力形式)
        各提案に「proposal_title」「proposal_content」「proposal_rationale」を必ず含め、JSON形式でリストとして出力してください。
        「proposal_title」には、考案した新しい専門家の役割や、その革新性が伝わるような名称を付けてください。

        # 出力項目の定義
        * **proposal_title**: 提案の簡潔な名称 (例: 「AI請求書ソリューション」, 「春風の俳句」, 「新商品のキャッチコピーA」)
        * **proposal_content**: 提案の「核」となる内容。
            * (解決策の場合): 提案の概要を記述してください。
            * (創作物の場合): 俳句、キャッチコピー、名前などの「創作物そのもの」を記述してください。
        * **proposal_rationale**: 提案の「理由」や「方法」。
            * (解決策の場合): 「具体的な方法」やメカニズム、その理由を2〜4行で説明してください。
            * (創作物の場合): 「その創作物の狙いや効果、背景、理由」を2〜4行で説明してください。
        * **重要**: 「proposal_rationale」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_title": "（考案した新専門家の役割を反映した革新的な名称）", 
              "proposal_content": "（その専門家が生成した革新的な提案の核となる内容）", 
              "proposal_rationale": "（その提案の具体的な方法、または狙いや理由を説明する2〜4行の文章です。）" 
            }}
          ] 
        }}
        """


# ----------------------------
# 4) EvoGenSolver (★修正箇所★)
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
        # (v11.0のまま)
        prompt = self.prompter.get_agent_personas_prompt(problem_statement)
        return self._call_llm(prompt)

    def _generate_initial_solutions(self, problem_statement: str, context: Dict) -> List[Dict[str, str]]:
        # (v9.0のまま)
        initial_agent_list = context 
        if not isinstance(initial_agent_list, list) or len(initial_agent_list) == 0:
            st.warning(f"[EvoGenSolver] 解決・進化エージェントのリストが不正です。")
            return []
        
        num_initial_agents = len(initial_agent_list)
        st.info(f"💡 {num_initial_agents}体の専門エージェントが初期解（10個）を分担して生成中...")
        
        all_solutions = []
        for i, agent_context in enumerate(initial_agent_list):
            st.caption(f"  - エージェント {i+1}/{num_initial_agents} ({agent_context.get('role', 'N/A')}) が生成中...")
            
            # (v12.0 の汎用プロンプトが呼ばれる)
            prompt = self.prompter.get_initial_generation_prompt(problem_statement, 1, agent_context)
            response = self._call_llm(prompt) 
            
            if isinstance(response, dict) and "solutions" in response and isinstance(response["solutions"], list) and len(response["solutions"]) > 0:
                all_solutions.append(response["solutions"][0])
            else:
                st.warning(f"[EvoGenSolver] エージェント {i+1} が不正な形式を返しました。デバッグ情報: {response}")
                
        return all_solutions

    # === ★v12.0: 修正点 5 (ログ出力等を 'proposal_title' に変更) ===
    def _evaluate_solutions(self, solutions: List[Dict[str, str]], problem_statement: str, context: Dict) -> Generator[str | List[Dict], None, None]:
        
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
            # ★v12.0 修正: 'name' -> 'proposal_title'
            if not isinstance(solution, dict) or "proposal_title" not in solution:
                yield f"  - 評価スキップ: 不正な形式の提案データです。"
                continue
            
            # ★v12.0 修正: 'name' -> 'proposal_title'
            yield f"  - 評価中 {i+1}/{len(solutions)}: {solution.get('proposal_title', '名称不明')} ( {num_evaluators}体のエージェントによる評価)"

            individual_evaluations = []
            
            for j, eval_context in enumerate(evaluator_agent_list):
                yield f"    - 評価者 {j+1}/{num_evaluators} ({eval_context.get('role', 'N/A')}) が評価..."
                
                # (v12.0 の汎用評価プロンプトが呼ばれる)
                prompt = self.prompter.get_evaluation_prompt(solution, problem_statement, eval_context)
                evaluation = self._call_llm(prompt)
                
                if isinstance(evaluation, dict) and "total_score" in evaluation and "error" not in evaluation:
                    individual_evaluations.append(evaluation)
                else:
                    # ★v12.0 修正: 'name' -> 'proposal_title'
                    st.warning(f"[EvoGenSolver] 提案 '{solution.get('proposal_title', 'N/A')}' の評価者 {j+1} が不正な形式を返しました。デバッグ情報: {evaluation}")

            if not individual_evaluations:
                # ★v12.0 修正: 'name' -> 'proposal_title'
                st.warning(f"[EvoGenSolver] 提案 '{solution.get('proposal_title', 'N/A')}' の有効な評価がありませんでした。")
                continue

            # (v11.0のままの集計ロジックでOK)
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

    def _generate_next_generation(self, evaluated_solutions: List[Dict], problem_statement: str, context: Dict) -> List[Dict[str, str]]:
        # (v9.0のまま)
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
                # 20%の確率: 革新 (v12.0 の汎用プロンプトが呼ばれる)
                st.caption(f"  - ⚡ (突然変異) エージェント {i+1}/{self.num_solutions} が「新規エージェントの定義」と「革新的な提案」を実行...")
                
                existing_roles = [a.get('role', 'N/A') for a in solver_agent_list]
                
                prompt = self.prompter.get_revolutionary_generation_prompt(
                    problem_statement, 
                    1, 
                    existing_roles 
                )
            else:
                # 80%の確率: 進化 (v12.0 の汎用プロンプトが呼ばれる)
                selected_agent_context = random.choice(solver_agent_list) 
                st.caption(f"  - 🧬 (進化) エージェント {i+1}/{self.num_solutions} ({selected_agent_context.get('role', 'N/A')}) が「既存の提案」を進化...")
                
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
                st.warning(f"[EvoGenSolver] エージェント {i+1} が不正な形式を返しました。デバッグ情報: {response}")

        return new_solutions

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        # (v9.0のまま)
        self.history = []

        yield "--- 🧠 課題を分析し、最適なAIエージェント・スウォームを編成中... ---"
        agent_personas = self._generate_agent_personas(problem_statement) 

        if not agent_personas or "error" in agent_personas or not all(k in agent_personas for k in ["solver_agents", "evaluators"]):
            yield "エラー: チーム編成に失敗しました。処理を中断します。"
            yield f"**デバッグ情報:** AIからの応答が不正です。APIキーが正しいか確認してください。\n```\n{agent_personas}\n```"
            return

        yield f"--- ✔️ チーム編成完了 ---"
        yield {"agent_team": agent_personas}

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
# 5) EvoGenSolver_Tavily (変更なし)
# ----------------------------
class EvoGenSolver_Tavily(EvoGenSolver):
    """
    Tavily を用いて課題に関連する最新情報を収集し、その情報を
    問題文に組み込んで EvoGen のフローを回す拡張版。
    """
    def __init__(self, llm_client: LLMClient, tavily_client: TavilyClient, num_solutions_per_generation: int = 10, tavily_results_per_search: int = 5):
        super().__init__(llm_client, num_solutions_per_generation)
        self.tavily = tavily_client
        self.tavily_results_per_query = tavily_results_per_search 

    def _get_snippet_text(self, results: List[Dict[str, Any]], max_snippets: int = 5) -> str:
        # (v9.0のまま)
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
        # (v9.0のまま)
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
                pass 

        fallback_sources = []
        for r in analysis_results[:2]:
            fallback_sources.append(f"- [分析] {r.get('title','No title')} ({r.get('url','')})")
        for r in solution_results[:2]:
            fallback_sources.append(f"- [解決策] {r.get('title','No title')} ({r.get('url','')})")
            
        fallback = "## Tavilyリサーチ要約（フォールバック）\n" + \
                   "最新のウェブ情報を参照しました。上位出典:\n" + "\n".join(fallback_sources) + \
                   "\n\n" + "--- (以下、元の課題文) ---\n" + problem_statement
        return fallback

    def solve(self, problem_statement: str, generations: int = 3) -> Generator[str | Dict, None, None]:
        # (v10.1のまま)
        
        yield "--- 💡 LLMによる最適な検索クエリ（フェーズ1 & 2）を生成中... ---"
        prompt = self.prompter.get_tavily_multi_phase_query_prompt(problem_statement)
        query_response = self._call_llm(prompt)

        if not isinstance(query_response, dict) or ("analysis_queries" not in query_response and "solution_queries" not in query_response):
            yield f"エラー: Tavilyクエリの生成に失敗しました。AIからの応答が不正です: {query_response}"
            augmented_problem = problem_statement
        else:
            analysis_queries = query_response.get("analysis_queries", [])
            solution_queries = query_response.get("solution_queries", [])
            
            yield f"--- ✔️ 生成されたクエリ ---"
            yield f"  - 分析クエリ: {', '.join(analysis_queries) if analysis_queries else 'なし'}"
            yield f"  - 解決策クエリ: {', '.join(solution_queries) if solution_queries else 'なし'}"
            
            analysis_results_list = []
            solution_results_list = []
            
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

            yield {"tavily_info_analysis": analysis_results_list, "tavily_info_solution": solution_results_list}

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
        
        # (v10.1のまま) 拡張された問題文（または元の問題文）をUIに渡す
        yield {"augmented_problem": augmented_problem}

        # (v12.0) 拡張された問題文で EvoGen の汎用提案スウォームロジックを実行
        yield from super().solve(augmented_problem, generations)


# ----------------------------
# 6) Streamlit UI (★v12.0: 表示内容を汎用フォーマットに対応★)
# ----------------------------
st.set_page_config(page_title="EvoGen AI + Tavily (Generalist Swarm)", layout="wide")
st.title("EvoGen AI 🧬")
st.markdown("進化型生成AI解探索フレームワーク (v12.0: 汎用提案モデル)")

# --- サイドバー設定 (v10.1のまま) ---
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_key = st.text_input("Google Gemini API Key", type="password", help="Gemini の API キーを入力してください（保存されません）。")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Tavily の API キーを入力してください（保存されません）。")
    st.subheader("パラメータ")
    num_generations = st.slider("世代数", 1, 20, 2, help="提案を進化させる回数です。")
    num_solutions = st.slider("世代ごとの(最大)提案の数", 3, 10, 10, help="第1世代以降に生成・評価する提案の数です。(第0世代は常に10個)")
    tavily_results_per_search = st.slider("Tavily 検索結果数 (クエリ毎)", 1, 10, 4, help="1つのクエリあたりにTavily から取得する検索結果数。") 
    st.markdown("---")
    st.info("Tavily を使って課題に関連する最新情報を取得し、それを参考に提案を生成します。")

# (v10.1のまま)
default_problem = """
# 課題
中小企業の経理部門における、請求書処理の業務効率を劇的に改善する
新しいAIソリューションを提案せよ。

# 要件・制約条件
- 導入コストが低いこと。（月額5万円以下）
- 専門的なIT知識がなくても利用できること。
- 既存の会計ソフト（例: freee, MFクラウド）と連携できることが望ましい。
"""
problem_statement = st.text_area("解決したい課題（または創作したいお題）を入力してください", value=default_problem, height=260)

# (v10.1のまま)
if st.button("提案の生成を開始", type="primary"):
    if not gemini_key:
        st.error("サイドバーでGoogle Gemini APIキーを入力してください。")
    elif not tavily_key:
        st.error("サイドバーでTavily APIキーを入力してください。")
    elif not problem_statement.strip():
        st.warning("課題を入力してください。")
    else:
        status_placeholder = st.empty()
        team_placeholder = st.empty()
        augmented_problem_placeholder = st.container() 
        tavily_placeholder = st.container() 
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
            
            # (v10.1のまま)
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

                # (v10.1のまま)
                elif isinstance(result, dict) and ("tavily_info_analysis" in result or "tavily_info_solution" in result):
                    tavily_placeholder.empty()
                    analysis_data = result.get("tavily_info_analysis", [])
                    solution_data = result.get("tavily_info_solution", [])
                    if analysis_data:
                        display_tavily_results(analysis_data, "🌐 フェーズ1: 課題の現状分析リサーチ結果")
                    if solution_data:
                        display_tavily_results(solution_data, "🌐 フェーズ2: 解決策の事例リサーチ結果")
                
                # (v10.1のまま)
                elif isinstance(result, dict) and "augmented_problem" in result:
                    with augmented_problem_placeholder.container():
                        st.subheader("🔍 リサーチ結果で補強された課題文")
                        with st.expander("補強された課題文の詳細を表示", expanded=False): 
                            st.markdown(result["augmented_problem"])
                        st.markdown("---")
                
                # (v11.0のまま)
                elif isinstance(result, dict) and "agent_team" in result:
                    with team_placeholder.container():
                        st.subheader("🤖 編成されたAIエージェント・スウォーム")
                        team = result["agent_team"]
                        with st.expander("チームの詳細を表示"):
                            
                            st.markdown("##### 💡🧬 解決・進化担当 (10体)")
                            gen_list = team.get("solver_agents", [])
                            if gen_list:
                                for i, gen in enumerate(gen_list):
                                    st.markdown(f"**{i+1}. {gen.get('role', '未定義')}:** {gen.get('instructions', '未定義')}")
                            else:
                                st.markdown("（定義されませんでした）")
                            
                            st.markdown("---")
                            st.markdown("##### 🧐 評価担当 (3体)") 
                            eva_list = team.get("evaluators", [])
                            if eva_list:
                                for i, eva in enumerate(eva_list):
                                    st.markdown(f"**{i+1}. {eva.get('role', 'N/A')}**")
                                    guideline = eva.get('evaluation_guideline', '評価ガイドライン未定義')
                                    st.caption(f"ガイドライン: {guideline}")
                            else:
                                st.markdown("（定義されませんでした）")

                # (★v12.0: 世代ごと結果表示を汎用フォーマットに対応)
                elif isinstance(result, dict) and "generation" in result:
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
                                
                                # ★v12.0 修正
                                st.markdown(f"**題名:** {sol.get('proposal_title', 'N/A')} (スコア: {score})")
                                st.markdown(f"**提案内容 (概要/創作物):**\n {sol.get('proposal_content', 'N/A')}")
                                st.markdown(f"**方法/理由:**\n {sol.get('proposal_rationale', 'N/A')}")
                                
                                if item != gen_data.get('results', [])[-1]:
                                    st.markdown("---")

        # === ★v12.0: 最終結果の表示を汎用フォーマットに対応 ===
        
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
                st.success("🏆 処理完了！スコアトップ5の提案はこちらです。")
                
                for i, item in enumerate(top_5_solutions):
                    sol = item.get('solution', {})
                    eva = item.get('evaluation', {})
                    score = eva.get('total_score', 'N/A')
                    
                    # ★v12.0 修正
                    st.header(f"🏅 第 {i + 1} 位: {sol.get('proposal_title', 'N/A')}")
                    st.metric(label="最終スコア (3エージェント平均)", value=f"{score}")

                    # ★v12.0 修正: 提案内容と理由を先に表示
                    st.info(f"**提案内容 (概要/創作物)**\n\n{sol.get('proposal_content', 'N/A')}")
                    st.info(f"**具体的な方法 / 狙い・理由**\n\n{sol.get('proposal_rationale', 'N/A')}")

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