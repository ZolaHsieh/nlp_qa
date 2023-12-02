from tqdm import tqdm
from collections import defaultdict

def compute_metrics(metric, 
                    start_logits, end_logits, 
                    processed_dataset, 
                    orig_dataset, 
                    n_largest=20,
                    max_answer_length=30):
    
    ## 建立 sample to index 對照表
    sample_id2idxs = defaultdict(list)
    for i, sample_id in enumerate(processed_dataset['sample_id']):
        sample_id2idxs[sample_id].append(i)

    predicted_answers = []
    for sample in orig_dataset: ## 原資料集
        sample_id = sample['id']
        context = sample['context']

        ## 最佳分數 ＆ans
        best_score = float('-inf')
        best_answer = None
        for idx in sample_id2idxs[sample_id]: ## 用上一步的mapping去找expand後的資料
            start_logit = start_logits[idx] # (384,) vector
            end_logit = end_logits[idx] # (384,) vector
            offsets = processed_dataset[idx]['offset_mapping']

            ## 取P(start)*P(end)最大的 -> log(start) + log(end) 最大
            start_indices = (-start_logit).argsort() #descending order
            end_indices = (-end_logit).argsort()

            for start_idx in start_indices[:n_largest]:
                for end_idx in end_indices[:n_largest]:
                    
                    ## 先確認是否有不合法的組合
                    if offsets[start_idx] is None or\
                        offsets[end_idx] is None or\
                        end_idx < start_idx or\
                        end_idx - start_idx + 1 > max_answer_length:
                        continue


                    # 計算分數並更新
                    score = start_logit[start_idx] + end_logit[end_idx]
                    if score > best_score:
                        best_score = score

                        ## 取得ans 字串
                        first_ch = offsets[start_idx][0]
                        last_ch = offsets[end_idx][1]
                        best_answer = context[first_ch:last_ch]

        # save best answer
        predicted_answers.append({'id': sample_id, 'prediction_text': best_answer})

    ## true answer 格式
    true_answers = [{'id': x['id'], 'answers': x['answers']} for x in orig_dataset]

    ## 計算
    return metric.compute(predictions=predicted_answers, references=true_answers)