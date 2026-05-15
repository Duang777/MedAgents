from data_utils import MyDataset
from api_utils import api_handler
from string import punctuation
import argparse
import tqdm
import json
import os
import time
from utils import *
from memory_bank import CaseBank
from rule_bank import RuleBank
from reflection_agent import EvolutionReflector
from dual_stage_retriever import DualStageRetriever


if __name__ == '__main__':
    def _count_and_fix_jsonl(path):
        if not os.path.exists(path):
            return 0
        valid_lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    valid_lines.append(s)
                except json.JSONDecodeError:
                    break
        # Truncate invalid tail for safe resume.
        with open(path, 'w', encoding='utf-8') as f:
            for s in valid_lines:
                f.write(s + '\n')
        return len(valid_lines)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='chatgpt')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=21)
    parser.add_argument('--end_pos', type=int, default=50)
    parser.add_argument('--output_files_folder', default='./outputs/MedQA')

    parser.add_argument('--method', type=str, default='syn_verif', choices=['syn_verif', 'syn_only', 'anal_only', 'base_direct', 'base_cot'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    parser.add_argument('--enable_memory', action='store_true')
    parser.add_argument('--memory_path', default='./memory/case_bank.json')
    parser.add_argument('--rule_path', default='./memory/rule_bank.json')
    parser.add_argument('--memory_top_k', type=int, default=3)
    parser.add_argument('--rule_top_k', type=int, default=2)
    parser.add_argument('--memory_max_size', type=int, default=500)
    parser.add_argument('--rule_max_size', type=int, default=500)
    parser.add_argument('--enable_reflection', action='store_true')
    parser.add_argument('--familiarity_top_m', type=int, default=15)
    parser.add_argument('--familiarity_threshold', type=float, default=0.6)
    parser.add_argument('--rerank_w1', type=float, default=0.5)
    parser.add_argument('--rerank_w2', type=float, default=0.3)
    parser.add_argument('--rerank_w3', type=float, default=0.2)
    parser.add_argument('--reflection_confidence_threshold', type=float, default=0.9)
    parser.add_argument('--enable_inner_enhancement', action='store_true')
    parser.add_argument('--inner_low_confidence_threshold', type=float, default=0.6)
    parser.add_argument('--enable_arbitration', action='store_true')
    parser.add_argument('--arbitration_confidence_threshold', type=float, default=0.7)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite_output', action='store_true')
    args = parser.parse_args()

    print(args)

    ### get handler
    handler = api_handler(args.model_name)

    ### get dataobj
    dataobj = MyDataset('test', args, traindata_obj=None)

    case_bank = None
    rule_bank = None
    reflector = None
    retriever = None
    if args.enable_memory:
        case_bank = CaseBank(storage_path=args.memory_path, max_size=args.memory_max_size)
        rule_bank = RuleBank(storage_path=args.rule_path, max_size=args.rule_max_size)
        retriever = DualStageRetriever(
            case_bank=case_bank,
            rule_bank=rule_bank,
            api_handler=handler,
            familiarity_top_m=args.familiarity_top_m,
            familiarity_threshold=args.familiarity_threshold,
            w1=args.rerank_w1,
            w2=args.rerank_w2,
            w3=args.rerank_w3,
        )
    if args.enable_memory and args.enable_reflection:
        reflector = EvolutionReflector(
            handler=handler,
            rule_bank=rule_bank,
            confidence_threshold=args.reflection_confidence_threshold,
        )

    ### set output_file_name
    exact_output_file = f"{args.output_files_folder}/{args.model_name}-{args.method}"
    os.makedirs(args.output_files_folder, exist_ok=True)

    if args.overwrite_output and os.path.exists(exact_output_file):
        os.remove(exact_output_file)

    resumed_count = 0
    if args.resume:
        resumed_count = _count_and_fix_jsonl(exact_output_file)

    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    effective_start = args.start_pos + resumed_count
    if effective_start > end_pos:
        effective_start = end_pos
    test_range = range(effective_start, end_pos)
    print(
        f"Run range: {args.start_pos}~{end_pos}, resume={args.resume}, "
        f"resumed_count={resumed_count}, effective_start={effective_start}"
    )


    input_prompt = {}
    running_total = 0
    running_correct = 0
    wall_start = time.time()
    pbar = tqdm.tqdm(test_range, desc=f"{effective_start} ~ {end_pos}")
    for idx in pbar:
        raw_sample = dataobj.get_by_idx(idx)
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        memory_context = ""
        retrieved_cases = []
        retrieved_rules = []
        if retriever is not None:
            retrieved_cases = retriever.retrieve_cases(question, top_k=args.memory_top_k)
            retrieved_rules = retriever.retrieve_rules(question, top_kr=args.rule_top_k)
            memory_context = build_memory_context(retrieved_cases, retrieved_rules)
        
        realqid = idx
        if args.dataset_name in ['MedQA', 'MedMCQA'] or 'MMLU' in args.dataset_name:
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj, memory_context=memory_context)
        elif args.dataset_name == 'PubMedQA':
            question = raw_sample['context'] + ' ' + question
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj, memory_context=memory_context)
        elif args.dataset_name in ['MedicationQA']:
            options = ''
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj, memory_context=memory_context)

        if case_bank is not None:
            success = data_info['pred_answer'] == gold_answer
            confidence = 1.0 if success else 0.0

            question_embedding = handler.get_embedding(question)
            if question_embedding:
                case_bank.add_case(
                    question=question,
                    options=options,
                    pred_answer=data_info['pred_answer'],
                    gold_answer=gold_answer,
                    syn_report=data_info.get('syn_report', ''),
                    reasoning_trace=data_info.get('raw_output', ''),
                    success=success,
                    confidence=confidence,
                    embedding=question_embedding,
                )

            new_rules = []
            if reflector is not None:
                new_rules = reflector.reflect_and_update(
                    question=question,
                    reasoning_trace=data_info.get('syn_report', ''),
                    gold_answer=gold_answer,
                    pred_answer=data_info['pred_answer'],
                    confidence=confidence,
                )
            data_info['new_rules_count'] = len(new_rules)

        data_info['memory_retrieved_count'] = len(retrieved_cases)
        data_info['rules_retrieved_count'] = len(retrieved_rules)
        q_domains_n = len(data_info.get('question_domains', [])) if isinstance(data_info.get('question_domains', []), list) else 0
        o_domains_n = len(data_info.get('option_domains', [])) if isinstance(data_info.get('option_domains', []), list) else 0
        q_cites = int(data_info.get('question_memory_citations', 0))
        q_valid_cites = int(data_info.get('question_valid_memory_citations', 0))
        o_cites = int(data_info.get('option_memory_citations', 0))
        o_valid_cites = int(data_info.get('option_valid_memory_citations', 0))
        data_info['memory_citation_rate_question'] = (q_cites / q_domains_n) if q_domains_n else 0.0
        data_info['memory_citation_rate_question_valid'] = (q_valid_cites / q_domains_n) if q_domains_n else 0.0
        data_info['memory_citation_rate_option'] = (o_cites / o_domains_n) if o_domains_n else 0.0
        data_info['memory_citation_rate_option_valid'] = (o_valid_cites / o_domains_n) if o_domains_n else 0.0
        # memory attribution summary
        supporting_memories = []
        for i, c in enumerate(retrieved_cases):
            supporting_memories.append({
                "type": "case",
                "id": c.get("id", ""),
                "label": f"Case Memory {i+1}",
            })
        for i, r in enumerate(retrieved_rules):
            supporting_memories.append({
                "type": "rule",
                "id": r.get("id", ""),
                "label": f"Rule Memory {i+1}",
            })
        data_info['supporting_memories'] = supporting_memories
        data_info['final_confidence'] = float(data_info.get('arbiter_confidence', 0.0)) if data_info.get('arbiter_used') else float(data_info.get('syn_confidence', 0.5))
        data_info['memory_enabled'] = case_bank is not None

        record = json.dumps(data_info)
        with open(exact_output_file, 'a') as f:
            f.write(record + '\n')

        running_total += 1
        if data_info['pred_answer'] == gold_answer:
            running_correct += 1
        running_acc = running_correct / running_total if running_total else 0.0
        elapsed = time.time() - wall_start
        avg_sec = elapsed / running_total if running_total else 0.0
        remain = len(test_range) - running_total
        eta_sec = int(avg_sec * max(0, remain))
        pbar.set_postfix({
            "acc": f"{running_acc:.4f}",
            "eta_sec": eta_sec,
            "done": running_total
        })
