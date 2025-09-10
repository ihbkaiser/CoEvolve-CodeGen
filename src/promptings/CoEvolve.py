from typing import List, Optional, Dict, Tuple
import tiktoken
import os
import json
import re
import sys
import time
import random
from datetime import datetime
from copy import deepcopy
from .Base import BaseStrategy # Assuming this is in your project structure
from models.Base import BaseModel # Adjust paths as needed
from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.LCBDataset import *
from results.Results import Results
from evaluations.func_evaluate import evaluate_io
import numpy as np
from forms import * # Assuming this imports your form models like PlanOutput, etc.
from multi_thread import multi_thread_task_dict
class AnalysisReflection:
    def __init__(self):
        self.historical_data = {} # Dictionary to store iteration data
    def update_historical_data(self, iteration: int, data: Dict):
        """Store data for the given iteration."""
        self.historical_data[iteration] = data
    def generate_prompt_for_plan_reflection(self, iteration: int, error_analysis: Dict, problem: str, problem_understanding: str, plan: str, historical_logs: Dict) -> str:
        """Generate a conversational prompt for plan debugging, evolving from R(t-1)."""
        previous_reflection = historical_logs.get('analysis_reflection', 'No previous analysis reflection available')
        insights = error_analysis.get('insights', '')
        success_rate = error_analysis.get('success_rate', 0.0)
        test_log = error_analysis.get("test_results", "")
        success_rate_str = f"{success_rate:.2f}%"
        prompt = f"""
You are a **debugging assistant** for a competitive programming problem. The plan is having problems, your task is to generate reflection on improving it.
# Context Provided:
## Problem:
{problem}
## Current Plan:
{plan}
## Current Test Log:
{test_log}
## Previous Reflection Prompt on Improving Plan From Previous Iteration:
{previous_reflection}
## New Insights:
{insights}
Write a new reflection prompt to correct the plan, use latest insights and previous reflection as context, but do not repeat or copy the old reflection.
⚠️ **IMPORTANT:**
- **Do not generate code.**
- **The test cases are always correct — never question their validity.**
"""
        return prompt
class CoEvolve(BaseStrategy):
    def __init__(
        self,
        k: int = 1,
        t: int = 5,
        max_attempts: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.top_plan = 1
        self.t = t
        self.number_of_code_per_plan = 1
        self.trust_weights = {
            'plan': 0.3,
            'code': 0.4,
            'content': 0.3
        }
        self.analysis_meaning = {
            "plan": "Identifies errors or problems in the planning approach.",
            "code": "Identifies errors or problems in the code implementation.",
            "content": "Identifies mismatches between problem, plan, and code."
        }
        self.history = []
        self.max_attempts = max_attempts
        self.verbose = True
        self.rt = AnalysisReflection() # Initialize AnalysisReflection for debugging guidance
    def _extract_json_string(self, text: str) -> Optional[str]:
        m = re.search(r'```json\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'({[\s\S]*})', text, re.DOTALL)
        return m.group(1) if m else None
    def _fix_invalid_escapes(self, json_str: str) -> str:
        json_str = json_str.replace('\b', '\\b').replace('\f', '\\f').replace('\r', '\\r').replace('\t', '\\t')
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        return json_str
    def parse_key_from_md(self, text: str, key: str) -> str:
        """
        Extracts the content under a markdown heading matching the key (e.g., ## key or ### key).
        Supports varying heading levels (#, ##, ###). If no match, returns the whole text as fallback.
        """
        # Flexible pattern for any number of # followed by the key, then content until next heading or end
        pattern = re.compile(r'#+\s*' + re.escape(key) + r'\s*(.*?)(?=#+\s*|\Z)', re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()
    def parse_code(self, response: str) -> str:
        if self.verbose:
            print("Step: Parsing code")
            print(f"Input response: {response}...")
        if "```" not in response:
            return response
        code_pattern = r'```((.|\n)*?)```'
        languages = ['Python', 'python', 'Python3', 'python3', 'C', 'c', 'C++', 'c++', 'Java', 'java', 'Node', 'node', 'Rust', 'rust', 'PHP', 'php', 'Go', 'go', 'Ruby', 'ruby', 'C#', 'c#', 'csharp']
        for lang in languages:
            if f"```{lang}" in response:
                code_pattern = r'```' + lang + r'((.|\n)*?)```'
                break
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            code_str = code_blocks[-1][0] if isinstance(code_blocks[-1], tuple) else code_blocks[-1]
        else:
            code_str = response
        parsed_code = code_str.strip()
        if self.verbose:
            print("Step: Code parsing successful")
            print(f"Parsed code: {parsed_code}...")
        return parsed_code
    def get_sample_io_str(self, item) -> str:
        if self.verbose:
            print("Step: Getting sample I/O string")
        if isinstance(self.data, XCodeDataset):
            sample_io = f"Input:\n{item['sample_inputs']}\nExpected output:\n{item['sample_outputs']}"
        elif isinstance(self.data, LCBDataset):
            return self.data.get_sample_io(item)
        else:
            sample_io_list = item.get('sample_io', [])
            if sample_io_list:
                if isinstance(sample_io_list[0], str):
                    sample_io = "\n".join(io for io in sample_io_list)
                elif isinstance(sample_io_list[0], dict):
                    sample_io = "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io_list])
            else:
                sample_io = ''
        if self.verbose:
            print("Step: Sample I/O retrieved")
            print(f"Sample I/O: {sample_io}...")
        return sample_io
    def get_problem_understanding(self, item) -> Tuple[str, int, int]:
        if self.verbose:
            print("Step: Generating problem understanding")
        problem_text = self.data.get_prompt(item)
        input_for_understanding = [
        {
        "role": "user",
        "content": f"""
        **You are a code generation assistant tasked with analyzing a programming problem.**
        ## Problem Description
        {problem_text}
        ## Sample Input/Output
        {self.get_sample_io_str(item)}
        ## Guidelines (IMPORTANT)
        - Clarify the **requirements** and **objectives** of the problem, give a concise understanding of the problem
        - Outline edge cases and important things to consider
        - Do **not** provide code, algorithms, or full solutions.
        - Clearly highlight what the problem is asking the solver to achieve.
        """
        }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for understanding")
            understanding, pr_tok, com_tok = self.gpt_chat(processed_input=input_for_understanding)
            item['api_calls'] += 1
            if self.verbose:
                print("Step: Understanding parsed")
                print(f"Understanding: {understanding}...")
            return understanding, pr_tok, com_tok
        except Exception as e:
            print(f"Error in get_problem_understanding: {e}")
            return "", 0, 0
    def generate_code_from_plan(self, item, planning: str, problem_text: str, sample_io_prompt: str, previous_codes: str = "", understanding: str = "") -> Tuple[List[Tuple[str, float, str]], int, int]:
        if self.verbose:
            print("Step: Generating code from plan")
            print(f"Plan: {planning}...")
        codes_with_scores = []
        pr_tok = 0
        com_tok = 0
        std_input_prompt = """
    - Strictly follow the sample input and output format.
    - The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function.
    - For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
    - Do not add extra print statement otherwise it will failed the test cases.
         """ if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset, LCBDataset)) else ""
        context = f"# Problem:\n {problem_text}\n"
        # context += understanding if understanding else ""
        for c_idx in range(1, self.number_of_code_per_plan + 1):
            if self.verbose:
                print(f"Step: Generating code variant {c_idx}")
            diversity_prompt = "" if c_idx == 1 else f"""
**Generate a distinct implementation** from previous ones: {previous_codes}. Use a unique approach, such as alternative data structures (e.g., list vs. dictionary, array vs. set in {self.language}), varied coding patterns (e.g., functional vs. imperative style).
Ensure the implementation strictly follows the provided plan and solves the problem correctly.
"""
            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"""# Task
**You are a programmer** tasked with solving a given problem using the **{self.language}** programming language. See the plan to solve the plan and implement code to solve it.
{context}
# Planning
{planning}
# Sample Test Cases
{sample_io_prompt}
{diversity_prompt}
# Instructions
- The generated **{self.language}** code must be inside a triple backtick (```) code block.
- Do not add extra explanation or words.
- Do not add assert statements in your code.
{std_input_prompt}
"""
                }
            ]
            try:
                if self.verbose:
                    print("Step: Making API call for code generation")
                code_response, pr_tok_1, com_tok_1 = self.gpt_chat(processed_input=input_for_code_generation)
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                item['api_calls'] += 1
                code = self.parse_code(code_response)
                if self.verbose:
                    print(f"Generated code variant {c_idx}: {code}")
                # Evaluate the code
                try:
                    passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
                    score = 1.0 if passed else 0.0
                except Exception as e:
                    print(f"Error evaluating code: {e}")
                    score = 0.0
                    test_log = f"Evaluation failed: {e}"
                codes_with_scores.append((code, score, test_log))
                previous_codes += f"\n- {code}"
            except Exception as e:
                print(f"Error generating code {c_idx}: {e}")
        if self.verbose:
            print(f"Step: {len(codes_with_scores)} code variants generated and evaluated")
        return codes_with_scores, pr_tok, com_tok
    def generate_plans(self, item, problem_understanding=None) -> Tuple[List[Tuple[str, float]], int, int]:
        if self.verbose:
            print("Step: Starting plan generation")
        plans_with_scores = []
        pr_tok = 0
        com_tok = 0
        previous_approaches = ""
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        if problem_understanding is None:
            problem_understanding, pr_u, com_u = self.get_problem_understanding(item)
            pr_tok += pr_u
            com_tok += com_u
        max_plans = self.k
        for t in range(1, max_plans + 1):
            if self.verbose:
                print(f"Step: Generating plan variant {t}")
            diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": (
                        f"You are a programmer tasked with generating appropriate plan to solve a given problem using the **{self.language}** programming language."
                        f"**# Target Problem:**\n{problem_text}\n\n"
                        f"**# Target Problem Understanding:**\n{problem_understanding}\n\n"
                        f"**## Sample Test Cases:**\n{sample_io_prompt}"
                        "**Expected Output Structure:**"
                        "### Recall Example Problem"
                        "**Recall a relevant and distinct problems** (different from problem mentioned above) and"
                        "- **Describe it**"
                        f"- **Generate {self.language} code** step by step to solve that problem"
                        "- **Discuss the algorithm** to solve this problem"
                        "- **Finally generate a planning** to solve that problem"
                        "### Algorithm to solve the original problem"
                        "- **Write down the algorithm** that is well suited for the original problem"
                        "- **Give some tutorials** about the algorithm for example:"
                        " - How to approach this type of algorithm"
                        " - Important things to consider"
                        "### Plan"
                        "- **Write down a detailed, step-by-step plan** to solve the **original problem**."
                        "--------"
                        "**IMPORTANT:**"
                        "- **Strictly follow** the instructions."
                        "- **DO NOT generate** code in your response."
                    ),
                },
            ]
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning generation attempt {attempt + 1} for variant {t}")
                try:
                    planning_resp, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_problem_planning)
                    planning = self.parse_key_from_md(planning_resp, "Plan")
                    pr_tok += pr_tok_temp
                    com_tok += com_tok_temp
                    item['api_calls'] += 1
                    break
                except Exception as e:
                    print(f"Error in planning attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        continue
            llm_score = 1 # Placeholder, as verification is commented out
            plans_with_scores.append((planning, llm_score))
            previous_approaches += f"\n- {planning}"
            if self.verbose:
                print(f"Step: Plan variant {t} completed")
                print(f"LLM score: {llm_score}")
        if len(plans_with_scores) < self.k:
            print(f"Warning: Only {len(plans_with_scores)}/{self.k} valid plans generated")
        if self.verbose:
            print(f"Step: {len(plans_with_scores)} plans generated")
        return plans_with_scores, pr_tok, com_tok
    def merged_analyses(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> Dict:
        if self.verbose:
            print("Step: Performing merged analyses (plan + code + content in one API call)")
        code_prompt_section = f"### Code\n{code}\n"
        input_prompt = [
            {
                "role": "user",
                "content": f"""
**You are a code generation assistant tasked with analyzing a programming problem in debugging, logical reasoning, and assessing solution alignments.**
---
## Context:
### Problem Description
{problem}
### Proposed Plan
{plan}
{code_prompt_section}
### Test Log (failing input/output)
{test_log}
---
## Response Structure
Your response must be structured with the following sections only:
### Plan Analysis
#### Simulation
Provide a detailed **step-by-step simulation** of the plan on the failing test cases, highlighting where divergences occur.
#### Insight
- Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.
- Finally, discuss how to correct this plan.

### Code Analysis
#### Simulation
Provide a detailed **line-by-line simulation** of the code on the failing test cases, highlighting divergences and errors.
#### Insight
Based on the simulation, provide concise insights on how to correct this code.

### Content Analysis
Provide a **single concise insight** (4-5 sentences) that includes:
* A **detailed evaluation** of the alignment between the plan and the code
* A **conclusion** on which component(s) should be updated
  (e.g., update the plan, update the code, update both, or no updates needed)
* Brief suggestions on how to improve alignment if necessary

---
## IMPORTANT
- **Strictly follow** the instructions and structure.
- The **test log is always true**. Do not modify or doubt it.
- Do not be overconfident. The **current plan or code has issues**.
- Do **not** generate new code.
- For content analysis, **focus only** on alignment issues with a concise insight.
- Do **NOT** introduce new solutions or rewrite the code/plan.
"""
            },
        ]
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Merged analyses attempt {attempt + 1}")
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_prompt)
                print(f"Response from merged analyses: {response}")
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                plan_simulation = self.parse_key_from_md(response, "Plan Analysis").split("#### Insight")[0].split("#### Simulation")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Plan Analysis") else ""
                plan_insight = self.parse_key_from_md(response, "Plan Analysis").split("#### Insight")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Plan Analysis") else self.parse_key_from_md(response, "Plan Analysis")
                code_simulation = self.parse_key_from_md(response, "Code Analysis").split("#### Insight")[0].split("#### Simulation")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Code Analysis") else ""
                code_insight = self.parse_key_from_md(response, "Code Analysis").split("#### Insight")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Code Analysis") else self.parse_key_from_md(response, "Code Analysis")
                content_insight = self.parse_key_from_md(response, "Content Analysis")
                analysis_result = {
                    'plan_analysis': {'simulation': plan_simulation, 'insights': plan_insight},
                    'code_analysis': {'simulation': code_simulation, 'insights': code_insight},
                    'content_analysis': {'insights': content_insight},
                    'pr_tok': pr_tok,
                    'com_tok': com_tok
                }
                if self.verbose:
                    print("Step: Merged analyses successful")
                    print(f"Plan insights: {analysis_result['plan_analysis']['insights']}...")
                    print(f"Code insights: {analysis_result['code_analysis']['insights']}...")
                    print(f"Content insights: {analysis_result['content_analysis']['insights']}...")
                return analysis_result
            except Exception as e:
                print(f"Error in merged_analyses attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return {
                        'plan_analysis': {'insights': ''},
                        'code_analysis': {'insights': ''},
                        'content_analysis': {'insights': ''},
                        'pr_tok': pr_tok,
                        'com_tok': com_tok
                    }
    def get_all_scores(self, decisions: List[str], analyses: Dict[str, Dict]) -> Tuple[Dict[str, Dict[str, ConfidenceOutput]], Dict[str, Dict[str, ConsistencyOutput]]]:
        """
        Computes all confidence and consistency scores in a single API call.
        Returns: (confidence_scores, consistency_scores)
        """
        if self.verbose:
            print("Step: Computing all confidence and consistency scores in a single API call")
        ANALYSES_ORDER = ["plan", "code", "content"]
        analysis_names = [n for n in ANALYSES_ORDER if n in analyses]
        if not analysis_names:
            return (
                {d: {n: ConfidenceOutput() for n in ANALYSES_ORDER} for d in decisions},
                {d: {} for d in decisions}
            )
        # Generate pairs for consistency
        pairs = []
        for i, n1 in enumerate(analysis_names):
            for n2 in analysis_names[i+1:]:
                pairs.append((n1, n2))
        agent_descriptions = {
            "plan": "Plan Analyst: finds logical flaws, missing steps, or edge cases in the plan.",
            "code": "Code Analyst: finds implementation bugs, logic mistakes, or I/O handling issues.",
            "content": "Content Evaluator: checks misalignment among problem, plan, and code."
        }
        analysis_meanings = {
            "plan": "Evaluates planning approach quality.",
            "code": "Evaluates code implementation quality.",
            "content": "Evaluates alignment between problem, plan, and code."
        }
        packed_analyses = [
            {
                "name": name,
                "role": agent_descriptions.get(name, ""),
                "purpose": analysis_meanings.get(name, ""),
                "insights": analyses.get(name, {}).get("insights", "")
            }
            for name in analysis_names
        ]
        user_content = (
            "**You are a senior competitive programming reviewer.** "
            "**Evaluate** confidence (how strongly each analysis supports/refutes each decision) and consistency (how much analysis pairs agree/disagree on each decision).\n\n"
            f"**Decisions:**\n{json.dumps(decisions)}\n\n"
            "**Decision meanings:**\n"
            "- **update code only:** The plan that generates the code is correct, but the code is wrong (e.g., implementation errors, bugs in code).\n"
            "- **update plan:** Both plan and code are wrong, but should fix the plan because the error is more serious (e.g., wrong approach, misunderstanding of the problem).\n\n"
            f"**Analysis Types and Insights:**\n{json.dumps(packed_analyses, ensure_ascii=False, indent=2)}\n\n"
            f"**Analysis Pairs for Consistency:**\n{json.dumps(pairs)}\n\n"
            "**Confidence Scoring rules** (in [0.0, 1.0]):\n"
            "- **1.0** = strongly supports with clear, direct evidence.\n"
            "- **0.7-0.9** = supports with mostly relevant reasoning, minor gaps.\n"
            "- **0.4-0.6** = weak/partial support; relevant but missing key links.\n"
            "- **0.1-0.3** = minimal/unclear relevance.\n"
            "- **0.0** = no relevance or contradicts the decision.\n\n"
            "**Consistency Scoring rules** (in [0.0, 1.0]):\n"
            "- **1.0** = both clearly support or both clearly refute the decision with aligned reasoning.\n"
            "- **0.7-0.9** = generally agree with minor differences in focus.\n"
            "- **0.4-0.6** = mixed/partial agreement; some overlap but notable differences.\n"
            "- **0.1-0.3** = mostly disagree with conflicting reasoning.\n"
            "- **0.0** = fully contradictory or unrelated conclusions.\n\n"
            "**Instructions:**\n"
            "1) For each **<decision> — <analysis_type>**, judge confidence with brief reasoning (1-3 sentences).\n"
            "2) For each **<decision> — <analysis1-analysis2>**, judge consistency with brief reasoning (1-3 sentences).\n"
            "3) If insights contradict, set low scores.\n\n"
            "**Output JSON ONLY** (no extra text, no markdown):\n"
            "{\n"
            ' "confidence_scores": {\n'
            ' "<decision>": {\n'
            ' "<analysis_type>": {\n'
            ' "confidence": float,\n'
            ' "reasoning": str\n'
            " }\n"
            " }\n"
            " },\n"
            ' "consistency_scores": {\n'
            ' "<decision>": {\n'
            ' "<analysis1>-<analysis2>": {\n'
            ' "consistency": float,\n'
            ' "reasoning": str\n'
            " }\n"
            " }\n"
            " }\n"
            "}"
        )
        messages = [{"role": "user", "content": user_content}]
        confidence_result: Dict[str, Dict[str, ConfidenceOutput]] = {}
        consistency_result: Dict[str, Dict[str, ConsistencyOutput]] = {}
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Scores API call attempt {attempt + 1}")
            try:
                response, _, _ = self.gpt_chat(messages)
                item['api_calls'] += 1
                json_str = self._extract_json_string(response)
                if not json_str:
                    if self.verbose:
                        print(f"Invalid output: No JSON found\nResponse head: {response}...")
                    continue
                json_str = self._fix_invalid_escapes(json_str)
                data = json.loads(json_str)
                confidence_scores = data.get("confidence_scores", {})
                consistency_scores = data.get("consistency_scores", {})
                for d in decisions:
                    confidence_result[d] = {}
                    for name in analysis_names:
                        item = confidence_scores.get(d, {}).get(name, {})
                        confidence_result[d][name] = ConfidenceOutput(
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                            reasoning=str(item.get("reasoning", "") or "")
                        )
                    consistency_result[d] = {}
                    for n1, n2 in pairs:
                        key = f"{n1}-{n2}"
                        item = consistency_scores.get(d, {}).get(key, {})
                        consistency_result[d][key] = ConsistencyOutput(
                            consistency=float(item.get("consistency", 0.0) or 0.0),
                            reasoning=str(item.get("reasoning", "") or "")
                        )
                if self.verbose:
                    print("Step: All scores calculated")
                    for d in decisions:
                        for name in analysis_names:
                            print(f"[CONF] {d}/{name}: {confidence_result[d][name].confidence:.3f}")
                        for key, obj in consistency_result[d].items():
                            print(f"[CONS] {d}/{key}: {obj.consistency:.3f}")
                return confidence_result, consistency_result
            except Exception as e:
                print(f"Error in get_all_scores attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Step: Max attempts reached, returning default scores")
                    return (
                        {d: {n: ConfidenceOutput() for n in analysis_names} for d in decisions},
                        {d: {f"{n1}-{n2}": ConsistencyOutput() for n1, n2 in pairs} for d in decisions}
                    )
    def fast_collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        """
        Updated collaborative_decision to use merged analyses and combined scores.
        """
        if self.verbose:
            print("Step: Starting collaborative decision with merged analysis")
        merged_result = {
            'plan_analysis': {'insights': ''},
            'code_analysis': {'insights': ''},
            'content_analysis': {'insights': ''},
            'pr_tok': 0,
            'com_tok': 0
        }
        try:
            problem_understanding, _, _ = self.get_problem_understanding(item)
            problem_text = self.data.get_prompt(item)
        
            # Perform merged analyses in one API call
            merged_result = self.merged_analyses(plan, code, outcomes, problem_text, problem_understanding)
            item['api_calls'] += 1
            # Extract analysis results
            analyses = {
                'plan': merged_result['plan_analysis'],
                'code': merged_result['code_analysis'],
                'content': merged_result['content_analysis']
            }
        
            decisions = ['update plan', 'update code only']
        
            # Compute all confidence and consistency scores in one API call
            confidence_scores, consistency_scores = self.get_all_scores(decisions, analyses)
        
            scores = {}
            for decision in decisions:
                if self.verbose:
                    print(f"Step: Scoring decision '{decision}'")
                total = 0.0
                for name in analyses.keys():
                    w = self.trust_weights[name]
                    conf = confidence_scores[decision][name].confidence
                    cons_prod = 1.0
                    for name2 in analyses.keys():
                        if name2 != name:
                            pair_key_1 = f"{name}-{name2}"
                            pair_key_2 = f"{name2}-{name}"
                            try:
                                cons = consistency_scores[decision][pair_key_1].consistency
                            except KeyError:
                                cons = consistency_scores[decision][pair_key_2].consistency
                            cons_prod *= cons
                    total += w * conf * cons_prod
                scores[decision] = total
                if self.verbose:
                    print(f"Step: Score for '{decision}': {total}")
        
            max_score = max(scores.values())
            candidates = [k for k, v in scores.items() if v == max_score]
            if len(candidates) > 1:
                return "update code only", merged_result
            decision = candidates[0]
            if self.verbose:
                print("Step: Decision made")
                print(f"Decision: {decision}")
            return decision, merged_result
     
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only", merged_result
    def debug_plan(self, iteration: int, plan: str, error_analysis: Dict, problem: str, problem_understanding: str, decision: str):
        if self.verbose:
            print(f"Step: Debugging plan at iteration {iteration}")
        prev_logs = self.rt.historical_data.get(iteration - 1, {})
        rt_prompt = self.rt.generate_prompt_for_plan_reflection(
            iteration, error_analysis, problem, problem_understanding, plan, historical_logs=prev_logs
        )
        try:
            if self.verbose:
                print("Step: Generating analysis reflection for plan")
                print(f"Prompt for analysis reflection: {rt_prompt}")
            analysis_reflection, _, _ = self.gpt_chat([{
                'role': 'user',
                'content': rt_prompt
            }])
            if self.verbose:
                print("Step: Analysis reflection generated")
                print(f"Reflection: {analysis_reflection}...")
        except Exception as e:
            print(f"Error generating analysis reflection for plan: {e}")
            analysis_reflection = "Error generating analysis reflection"
        update_prompt = [
            {
                'role': 'user',
                'content': f"""You are a programmer tasked with generating appropriate plan to solve a given problem using the **{self.language}** programming language. You already have a wrong plan. Correct it so that it can generate correct plan.
## Problem
{problem}
## Plan Critique
{analysis_reflection}
Your response must be structured as follows:
## New Plan
- Write down a detailed, step-by-step modified plan to solve the **original problem**.
- Ensure each step logically follows from the previous one.
**IMPORTANT Instruction:**
- Your response must contain only the plan.
- Do not add any explanation.
- Do not generate code.
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for plan update")
            updated_response, _, _ = self.gpt_chat(update_prompt)
            revised_plan = updated_response.strip()
            if self.verbose:
                print("Step: Plan updated")
                print(f"Revised plan: {revised_plan}...")
        except Exception as e:
            print(f"Error debugging plan: {e}")
            revised_plan = plan
        self.rt.update_historical_data(iteration, {
            'previous_plan': plan,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1,
            'analysis_reflection': analysis_reflection
        })
        if self.verbose:
            print("Step: Historical data updated for plan")
        return revised_plan, analysis_reflection
    def debug_code(self, iteration: int, plan: str, code: str, error_analysis: Dict, problem: str, problem_understanding: str, decision: str):
        """
        Debug the code using only code analysis insights, without reflection analysis.
        """
        if self.verbose:
            print(f"Step: Debugging code at iteration {iteration} using code analysis only")
        std_input_prompt = """
    - Strictly follow the sample input and output format.
    - The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function.
    - For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
    - Do not add extra print statement otherwise it will failed the test cases.
         """ if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset, LCBDataset)) else ""
        # Extract insights and test results from error_analysis
        insights = error_analysis.get('insights', 'No insights provided')
        test_log = error_analysis.get('test_results', 'No test results provided')
        # Prompt for code refinement using code analysis insights directly
        code_prompt = [
            {
                "role": "user",
                "content": f"""You are a programmer who has received a solution of a problem written in **{self.language}** that fails to pass certain test cases. Your task is to modify the code in such a way so that it can pass all the test cases. Do not generate same code.
## Problem:
{problem}
## Current Plan
{plan}
## Buggy Code
```{self.language}
{code}
```
## Test Log
{test_log}
## Code Critique
{insights}
**Task:** Using the provided code critique and test log, **refine the code** to correct the identified issues.
**IMPORTANT:** Your response must contain **only the {self.language} code** to solve this problem:
```{self.language}
# Your corrected code, with comments explaining each correction.
```
**Important Instructions:**
- Strictly follow the instructions.
- Do not add testing code for example assert statement in your code.
- Do not be overconfident that the generated code is correct. It is wrong.
- The modified **{self.language}** code must be enclosed within triple backticks
{std_input_prompt}
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for code update")
            updated_response, _, _ = self.gpt_chat(code_prompt)
            revised_code = self.parse_code(updated_response)
            if self.verbose:
                print("Step: Code updated")
                print(f"Revised code: {revised_code}...")
        except Exception as e:
            print(f"Error debugging code: {e}")
            revised_code = code
        if self.verbose:
            print("Step: Historical data updated for code")
        return revised_code, insights
    def _inner_run(self, item):
        self.rt.historical_data = {}
        if self.verbose:
            print("Step: Starting inner run")
        pr_tok = 0
        com_tok = 0
        all_codes_with_scores = [] # List to collect (code, score)
        try:
            problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
            pr_tok += pr_tok_u
            com_tok += com_tok_u
        except Exception as e:
            print(f"Error getting problem understanding: {e}")
            problem_understanding = ""
        try:
            plans_with_scores, pr_tok_p, com_tok_p = self.generate_plans(item, problem_understanding)
            pr_tok += pr_tok_p
            com_tok += com_tok_p
            if self.verbose:
                print("Step: Plans generated")
                print(f"Number of plans: {len(plans_with_scores)}")
        except Exception as e:
            print(f"Error generating plans: {e}")
            plans_with_scores = []
        if not plans_with_scores:
            print("Warning: No valid plans generated. Returning default code.")
            return "# No valid solution generated", pr_tok, com_tok
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        for plan_idx, (planning, plan_score) in enumerate(plans_with_scores, 1):
            if self.verbose:
                print(f"Step: Processing plan {plan_idx}")
            try:
                codes_with_scores, pr_tok_code, com_tok_code = self.generate_code_from_plan(
                    item, planning, problem_text, sample_io_prompt, "", problem_understanding
                )
                pr_tok += pr_tok_code
                com_tok += com_tok_code
            except Exception as e:
                print(f"Error generating codes for plan {plan_idx}: {e}")
                continue
            for code_idx, (code, code_score, test_log) in enumerate(codes_with_scores, 1):
                all_codes_with_scores.append((code, code_score))
                if self.verbose:
                    print(f"Step: Added initial code for plan {plan_idx}, code {code_idx} - Score: {code_score}")
                passed = code_score == 1.0
                if passed:
                    if self.verbose:
                        print(f"Step: Code passed samples for plan {plan_idx}, code {code_idx}")
                    return code, pr_tok, com_tok
                current_planning = planning
                current_code = code
                current_test_log = test_log
                current_code_score = code_score
                for i in range(1, self.t + 1):
                    if self.verbose:
                        print(f"Step: Iteration {i} for plan {plan_idx}")
                    try:
                        decision, merged_result = self.fast_collaborative_decision(current_planning, current_code, current_test_log, item)
                        if self.verbose:
                            print(f"Step: Decision made: {decision}")
                    except Exception as e:
                        print(f"Error in decision: {e}")
                        decision = "update code only"
                    if decision == 'update plan':
                        try:
                            A_plan = merged_result['plan_analysis']
                            revised_plan, _ = self.debug_plan(i, current_planning, {
                                'insights': A_plan['insights'],
                                'test_results': current_test_log,
                                'success_rate': current_code_score * 100,
                            }, problem_text, problem_understanding, decision)
                            codes_with_scores, pr_tok_code, com_tok_code = self.generate_code_from_plan(item, revised_plan, problem_text, sample_io_prompt, "", problem_understanding)
                            pr_tok += pr_tok_code
                            com_tok += com_tok_code
                            if codes_with_scores:
                                for new_code, new_score, new_test_log in codes_with_scores:
                                    all_codes_with_scores.append((new_code, new_score))
                                    if self.verbose:
                                        print(f"Step: Added updated code after plan debug - Score: {new_score}")
                                    if new_score == 1.0:
                                        if self.verbose:
                                            print(f"Step: Updated code passed samples after plan debug")
                                        return new_code, pr_tok, com_tok
                                # Update current for next iteration (pick the best from new codes)
                                best_new = max(codes_with_scores, key=lambda x: x[1])
                                current_code, current_code_score, current_test_log = best_new
                            current_planning = revised_plan
                        except Exception as e:
                            print(f"Error updating plan: {e}")
                            continue
                    else:
                        try:
                            A_code = merged_result['code_analysis']
                            revised_code, _ = self.debug_code(i, current_planning, current_code, {
                                'insights': A_code['insights'],
                                'test_results': current_test_log,
                                'success_rate': current_code_score * 100,
                            }, problem_text, problem_understanding, decision)
                            try:
                                passed, new_test_log = self.data.evaluate_sample_io(item, revised_code, self.language)
                                new_score = 1.0 if passed else 0.0
                            except Exception as e:
                                print(f"Error evaluating updated code: {e}")
                                new_test_log = f"Evaluation failed: {e}"
                                new_score = 0.0
                            all_codes_with_scores.append((revised_code, new_score))
                            if self.verbose:
                                print(f"Step: Added updated code after code debug - Score: {new_score}")
                            if new_score == 1.0:
                                if self.verbose:
                                    print(f"Step: Updated code passed samples after code debug")
                                return revised_code, pr_tok, com_tok
                            # Update current for next iteration
                            current_code = revised_code
                            current_code_score = new_score
                            current_test_log = new_test_log
                        except Exception as e:
                            print(f"Error updating code: {e}")
                            continue
        # At the end, select the code with the highest score
        if all_codes_with_scores:
            best_code, best_score = max(all_codes_with_scores, key=lambda x: x[1])
            if self.verbose:
                print(f"Step: Selected best code with score: {best_score}")
            return best_code, pr_tok, com_tok
        else:
            print("Warning: No codes generated. Returning default.")
            return "# No valid solution generated", pr_tok, com_tok
    def run_single_pass(self, item: dict):
        if self.verbose:
            print("Step: Starting single pass run")
        max_retries = 1
        for attempt in range(1, max_retries + 1):
            if self.verbose:
                print(f"Step: Run attempt {attempt}")
            try:
                item['api_calls'] = item.get('api_calls', 0)
                result = self._inner_run(item)
                if self.verbose:
                    print("Step: Run successful")
                return result
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    return "No_solution_found", 0, 0
