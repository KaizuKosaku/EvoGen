
from typing import List, Dict, Any

class PromptManager:
    """Manages prompt templates for the AI agent."""
    
    def get_tavily_multi_phase_query_prompt(self, problem_statement: str) -> str:
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
        - ステップ1の分析に基づき、生成された提案をスコアとして厳密に評価するために**最も重要となる3つの異なる評価観点**を特定してください。
        - その3つの観点に基づき、それぞれ専門の評価エージェント（evaluators）を3体定義してください。
        - `role`: あなたが考案した、課題に特化した評価者の役割名。
        - `evaluation_guideline`: その役割が提案を厳密に評価するために使用する、**具体的かつ詳細な評価指針（ガイドライン）**。

        ## ステップ4: 動的UIラベルの定義
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

    def get_all_agent_queries_prompt(self, problem_statement: str, solver_agents: List[Dict]) -> str:
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
            // ... 10体全員分 ...
            "（エージェント10の role 文字列）": [
              "（エージェント10の視点での検索クエリ1）",
              "（エージェント10の視点での検索クエリ2）"
            ]
          }}
        }}
        """

    def get_agent_specific_analysis_prompt(self, problem_statement: str, agent_role: str, agent_instructions: str, raw_content_text: str) -> str:
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
            // ...
            "（{agent_role}の視点で抽出した重要な洞察10）"
          ]
        }}
        """

    def get_initial_generation_prompt(self, problem_statement: str, num_solutions: int, context: Dict[str, Any]) -> str:
        insights = context.get('agent_research_insights', [])
        insights_text = "\n".join([f"- {item}" for item in insights]) if insights else "（追加の調査情報なし）"

        return f"""
        # 役割: {context.get('role', 'あなたは一流のイノベーターです。')}
        # 指示: {context.get('instructions', f'以下の課題に対し、互いに全く異なるアプローチからの提案を{num_solutions}個生成してください。')}
        # 課題文: {problem_statement}

        # ★あなた専用の調査情報★
        # 以下の個別の調査結果を**必ず**参考にして、独自の提案を生成してください。
        {insights_text}

        # !!最重要!! (出力形式)
        各提案に「proposal_main」「proposal_details」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義
        * **proposal_main**: 提案の「核」となる部分。(例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。(例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\\n)は使用して構いません。**

        # 出力JSONの例
        {{ 
          "solutions": [ 
            {{ 
              "proposal_main": "提案1の核 (名称 または 創作物そのもの)", 
              "proposal_details": "提案1の詳細 (具体的な内容、意図、背景、理由、効果など) を説明する2〜4行の文章です。\\nこのように改行を含めても構いません。"
            }}
          ] 
        }}
        """

    def get_evaluation_prompt(self, solution: Dict[str, str], problem_statement: str, context: Dict[str, Any]) -> str:
        evaluator_role = context.get('role', 'あなたは客観的で厳しい批評家です。')
        evaluation_guideline = context.get('evaluation_guideline', '提示された提案を、課題の要件に基づき厳密に評価してください。')

        return f"""
        # あなたの厳格な役割
        あなたは「{evaluator_role}」です。

        # あなたの最重要評価ガイドライン
        {evaluation_guideline}

        # 評価対象の課題
        {problem_statement}
        
        # 評価対象の提案
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

    def get_next_generation_prompt(self, elite_solutions: List[Dict], failed_solutions: List[Dict], problem_statement: str, num_solutions: int, context: Dict[str, Any]) -> str:
        elite_text = "\n".join([f"- {s['solution'].get('proposal_main', 'N/A')} (スコア: {s['evaluation'].get('total_score', 0)})" for s in elite_solutions])
        failed_text = "\n".join([f"- {s['solution'].get('proposal_main', 'N/A')} (弱点: {s['evaluation'].get('weaknesses', 'N/A')})" for s in failed_solutions])

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

        # ★あなた専用の調査情報★
        # 以下の個別の調査結果も**必ず**参考にして、提案を進化させてください。
        {insights_text}
        
        # 新しい提案の生成指示: {context.get('instructions')}
        
        # !!最重要!! (出力形式)
        各提案に「proposal_main」「proposal_details」を必ず含め、JSON形式でリストとして出力してください。

        # 出力項目の定義
        * **proposal_main**: 提案の「核」となる部分。 (例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。 (例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\\n)は使用して構いません。**

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

        # 出力項目の定義
        * **proposal_main**: 提案の「核」となる部分。 (例: 「提案の名称」 または 「創作物そのもの」)
        * **proposal_details**: 提案の「詳細」となる部分。 (例: 「具体的な内容や方法、得られる効果」 または 「意図、背景、理由、狙い」) を2〜4行で説明してください。
        * **重要**: 「proposal_details」には箇条書き、マークダウン、ネストされたJSONを使用しないでください。ただし、**文章内での改行コード(\\n)は使用して構いません。**

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
