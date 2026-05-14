from prompt_generator import *
from data_utils import *
import json



def build_memory_context(cases, rules):
    chunks = []
    for idx, case in enumerate(cases):
        strategy = case.get('syn_report', '') or case.get('reasoning_trace', '')
        chunks.append(
            f"Case Memory {idx + 1}:\n"
            f"- Past Question: {case.get('question', '')}\n"
            f"- Reusable Strategy: {strategy}\n"
            f"- Outcome: {'success' if case.get('success') else 'failure'}\n"
        )
    for idx, rule in enumerate(rules):
        chunks.append(
            f"Rule Memory {idx + 1}:\n"
            f"- Rule: {rule.get('rule_text', '')}\n"
            f"- Confidence: {rule.get('confidence', 0.0)}\n"
        )
    return "\n".join(chunks)

def parse_confidence_from_syn_report(syn_report):
    text = (syn_report or "").lower()
    marker = "confidence:"
    if marker not in text:
        return 0.5
    frag = text.split(marker, 1)[1].strip().splitlines()[0].strip()
    try:
        val = float(frag)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, val))

def count_memory_citations(text):
    t = (text or "").lower()
    marker = "memory references:"
    if marker not in t:
        return 0
    refs = t.split(marker, 1)[1].splitlines()[0].strip()
    if refs in ["", "none", "n/a", "na"]:
        return 0
    parts = [p.strip() for p in refs.split(",") if p.strip()]
    return len(parts)

def parse_analysis_json(text):
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        analysis = str(obj.get("analysis", "")).strip()
        refs = obj.get("memory_references", [])
        if not isinstance(refs, list):
            refs = []
        refs = [str(x).strip() for x in refs if str(x).strip()]
        return analysis if analysis else raw, refs
    except Exception:
        return raw, []

def memory_candidates_from_brief(memory_context):
    refs = []
    for line in (memory_context or "").splitlines():
        s = line.strip()
        if s.startswith("Case Memory ") and s.endswith(":"):
            refs.append(s[:-1])
        if s.startswith("Rule Memory ") and s.endswith(":"):
            refs.append(s[:-1])
    return refs

def count_valid_refs(refs, candidates):
    if not candidates:
        return 0
    cands = {c.lower() for c in candidates}
    cnt = 0
    for r in refs:
        if str(r).strip().lower() in cands:
            cnt += 1
    return cnt

def detect_conflict(question_analyses, option_analyses):
    q_text = " ".join(question_analyses.values()).lower() if isinstance(question_analyses, dict) else ""
    o_text = " ".join(option_analyses.values()).lower() if isinstance(option_analyses, dict) else ""
    neg_markers = ["however", "but", "in contrast", "conflict", "disagree", "inconsistent"]
    score = sum(1 for m in neg_markers if (m in q_text or m in o_text))
    return score >= 2

def parse_arbiter_json(text):
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        ans = str(obj.get("arbiter_answer", "")).strip().upper()
        reasoning = str(obj.get("arbiter_reasoning", "")).strip()
        used = obj.get("used_memories", [])
        conf = float(obj.get("arbiter_confidence", 0.0))
        if not isinstance(used, list):
            used = []
        used = [str(x).strip() for x in used if str(x).strip()]
        return ans, reasoning, used, max(0.0, min(1.0, conf))
    except Exception:
        return "", "", [], 0.0


def fully_decode(qid, realqid, question, options, gold_answer, handler, args, dataobj, memory_context=""):

    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []

    if args.method == "base_direct":
        direct_prompt = get_direct_prompt(question, options)
        output = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, max_tokens=50, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot":
        cot_prompt = get_cot_prompt(question, options)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output)
    else:
        # get question domains
        question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
        raw_question_domain = handler.get_output_multiagent(user_input=prompt_get_question_domain, temperature=0, max_tokens=50, system_role=question_classifier)
        if raw_question_domain == "ERROR.":
            raw_question_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
        question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")

        # get option domains
        options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
        raw_option_domain = handler.get_output_multiagent(user_input=prompt_get_options_domain, temperature=0, max_tokens=50, system_role=options_classifier)
        if raw_option_domain == "ERROR.":
            raw_option_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
        options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")

        # get question analysis
        tmp_question_analysis = []
        question_memory_citations = 0
        question_valid_memory_citations = 0
        memory_candidates = memory_candidates_from_brief(memory_context) if args.enable_inner_enhancement else []
        for _domain in question_domains:
            question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(
                question, _domain, memory_brief=memory_context if args.enable_inner_enhancement else ""
            )
            raw_question_analysis = handler.get_output_multiagent(user_input=prompt_get_question_analysis, temperature=0, max_tokens=300, system_role=question_analyzer)
            if args.enable_inner_enhancement:
                parsed_analysis, refs = parse_analysis_json(raw_question_analysis)
                if memory_candidates and len(refs) == 0:
                    repair_prompt = (
                        "Reformat the following analysis into strict JSON and add at least one relevant "
                        "memory reference from the provided candidates when possible.\n\n"
                        f"Candidates: {memory_candidates}\n"
                        f"Analysis text:\n{raw_question_analysis}\n\n"
                        "Output JSON only:\n"
                        "{\"analysis\":\"...\", \"memory_references\":[\"Case Memory 1\"]}"
                    )
                    repaired = handler.get_output_multiagent(
                        user_input=repair_prompt, temperature=0, max_tokens=220, system_role=""
                    )
                    repaired_analysis, repaired_refs = parse_analysis_json(repaired)
                    if repaired_refs:
                        parsed_analysis, refs = repaired_analysis, repaired_refs
                question_memory_citations += len(refs)
                question_valid_memory_citations += count_valid_refs(refs, memory_candidates)
                tmp_question_analysis.append(parsed_analysis)
            else:
                question_memory_citations += count_memory_citations(raw_question_analysis)
                tmp_question_analysis.append(raw_question_analysis)
        question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

        # get option analysis
        tmp_option_analysis = []
        option_memory_citations = 0
        option_valid_memory_citations = 0
        memory_candidates = memory_candidates_from_brief(memory_context) if args.enable_inner_enhancement else []
        for _domain in options_domains:
            option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(
                question, options, _domain, question_analyses,
                memory_brief=memory_context if args.enable_inner_enhancement else ""
            )
            raw_option_analysis = handler.get_output_multiagent(user_input=prompt_get_options_analyses, temperature=0, max_tokens=300, system_role=option_analyzer)
            if args.enable_inner_enhancement:
                parsed_analysis, refs = parse_analysis_json(raw_option_analysis)
                if memory_candidates and len(refs) == 0:
                    repair_prompt = (
                        "Reformat the following analysis into strict JSON and add at least one relevant "
                        "memory reference from the provided candidates when possible.\n\n"
                        f"Candidates: {memory_candidates}\n"
                        f"Analysis text:\n{raw_option_analysis}\n\n"
                        "Output JSON only:\n"
                        "{\"analysis\":\"...\", \"memory_references\":[\"Case Memory 1\"]}"
                    )
                    repaired = handler.get_output_multiagent(
                        user_input=repair_prompt, temperature=0, max_tokens=220, system_role=""
                    )
                    repaired_analysis, repaired_refs = parse_analysis_json(repaired)
                    if repaired_refs:
                        parsed_analysis, refs = repaired_analysis, repaired_refs
                option_memory_citations += len(refs)
                option_valid_memory_citations += count_valid_refs(refs, memory_candidates)
                tmp_option_analysis.append(parsed_analysis)
            else:
                option_memory_citations += count_memory_citations(raw_option_analysis)
                tmp_option_analysis.append(raw_option_analysis)
        option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')

        if args.method == "anal_only":
            answer_prompt = get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses)
            output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
            ans, output = cleansing_final_output(output)
        else:
            # get synthesized report
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            o_analyses_text = transform_dict2text(option_analyses, "options", options)
            synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(
                q_analyses_text, o_analyses_text,
                memory_brief=memory_context if args.enable_inner_enhancement else ""
            )
            raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report, temperature=0, max_tokens=2500, system_role=synthesizer)
            if "Total Analysis:" not in raw_synthesized_report and raw_synthesized_report != "ERROR.":
                reformat_prompt = (
                    "Reformat the following text into exactly this format:\n"
                    "Key Knowledge: [extracted key knowledge]\n"
                    "Total Analysis: [synthesized analysis]\n\n"
                    f"Text:\n{raw_synthesized_report}"
                )
                reformatted = handler.get_output_multiagent(
                    user_input=reformat_prompt, temperature=0, max_tokens=800, system_role=""
                )
                if reformatted != "ERROR.":
                    raw_synthesized_report = reformatted
            syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
            conflict_detected = detect_conflict(question_analyses, option_analyses)
            syn_confidence = parse_confidence_from_syn_report(raw_synthesized_report)

            if args.method == "syn_only":
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report, memory_context=memory_context)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                ans, output = cleansing_final_output(output)
            elif args.method == "syn_verif":
                all_domains = question_domains + options_domains


                syn_repo_history = [syn_report]

            
                hasno_flag = True   # default value: in order to get into the while loop
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}    # 'domain' : 'yes' / 'no'
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False
                    # hold a meeting for all domain experts to vote and gather advice if they do not agree
                    for domain in all_domains:
                        voter, cons_prompt = get_consensus_prompt(domain, syn_report)
                        raw_domain_opi = handler.get_output_multiagent(user_input=cons_prompt, temperature=0, max_tokens=30, system_role=voter)
                        domain_opinion = cleansing_voting(raw_domain_opi)   # "yes" / "no"
                        domain_opinions[domain] = domain_opinion
                        if domain_opinion == "no":
                            advice_prompt = get_consensus_opinion_prompt(domain, syn_report)
                            advice_output = handler.get_output_multiagent(user_input=advice_prompt, temperature=0, max_tokens=500, system_role=voter)
                            revision_advice[domain] = advice_output
                            hasno_flag = True
                    if hasno_flag:
                        revision_prompt = get_revision_prompt(syn_report, revision_advice)
                        if args.enable_inner_enhancement and (conflict_detected or syn_confidence < args.inner_low_confidence_threshold):
                            revision_prompt += f"\n\nUse this memory brief to resolve conflict or low confidence:\n{memory_context}\n"
                        revised_analysis = handler.get_output_multiagent(user_input=revision_prompt, temperature=0, max_tokens=2500, system_role="")
                        syn_report = cleansing_syn_report(question, options, revised_analysis)
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)
                    vote_history.append(domain_opinions)
                arbiter_used = False
                arbiter_memories = []
                arbiter_confidence = 0.0
                # arbitration for low-confidence or conflict cases
                if args.enable_arbitration and (conflict_detected or syn_confidence < args.arbitration_confidence_threshold):
                    arbiter_prompt = (
                        "You are an arbitration expert in a multi-expert clinical consultation.\n"
                        f"Question:\n{question}\n\n"
                        f"Options:\n{options}\n\n"
                        f"Synthesized report:\n{syn_report}\n\n"
                        f"Memory brief:\n{memory_context}\n\n"
                        "Output strict JSON only:\n"
                        "{\"arbiter_answer\":\"A|B|C|D|E\", "
                        "\"arbiter_reasoning\":\"...\", "
                        "\"used_memories\":[\"Case Memory 1\", \"Rule Memory 1\"], "
                        "\"arbiter_confidence\":0.0}"
                    )
                    arbiter_raw = handler.get_output_multiagent(
                        user_input=arbiter_prompt, temperature=0, max_tokens=500, system_role=""
                    )
                    a_ans, a_reasoning, a_memories, a_conf = parse_arbiter_json(arbiter_raw)
                    if a_ans in ["A", "B", "C", "D", "E"]:
                        ans = a_ans
                        output = f"ArbiterReasoning: {a_reasoning}"
                        arbiter_used = True
                        arbiter_memories = a_memories
                        arbiter_confidence = a_conf

                if not arbiter_used:
                    # final answer derivation
                    answer_prompt = get_final_answer_prompt_wsyn(syn_report, memory_context=memory_context)
                    output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                    ans, output = cleansing_final_output(output)
                        


    data_info = {
        'question': question,
        'options': options,
        'pred_answer': ans,
        'gold_answer': gold_answer,
        'question_domains': question_domains,
        'option_domains': options_domains,
        'question_analyses': question_analyses,
        'option_analyses': option_analyses,
        'syn_report': syn_report,
        'vote_history': vote_history,
        'revision_history': revision_history,
        'syn_repo_history': syn_repo_history,
        'raw_output': output,
        'question_memory_citations': question_memory_citations if 'question_memory_citations' in locals() else 0,
        'question_valid_memory_citations': question_valid_memory_citations if 'question_valid_memory_citations' in locals() else 0,
        'option_memory_citations': option_memory_citations if 'option_memory_citations' in locals() else 0,
        'option_valid_memory_citations': option_valid_memory_citations if 'option_valid_memory_citations' in locals() else 0,
        'conflict_detected': conflict_detected if 'conflict_detected' in locals() else False,
        'syn_confidence': syn_confidence if 'syn_confidence' in locals() else 0.5,
        'arbiter_used': arbiter_used if 'arbiter_used' in locals() else False,
        'arbiter_memories': arbiter_memories if 'arbiter_memories' in locals() else [],
        'arbiter_confidence': arbiter_confidence if 'arbiter_confidence' in locals() else 0.0,
    }
    
    return data_info
