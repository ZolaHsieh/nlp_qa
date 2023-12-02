from transformers import AutoTokenizer

class QATokenizer:
    def __init__(self, checkpoint, max_length, stride):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length
        self.stride = stride


    def find_answer_token_idx(self, ctx_start, ctx_end,
                            ans_start_char, ans_end_char,
                            offset):
        start_idx = 0
        end_idx = 0

        if (ans_start_char >= offset[ctx_start][0]) and (ans_end_char <= offset[ctx_end][1]): # ans 位置在切割後的context內
            i = ctx_start
            for start_end_char in offset[ctx_start:]:
                start, end = start_end_char
                if start == ans_start_char:
                    start_idx = i

                if end == ans_end_char:
                    end_idx = i
                    break

                i += 1
        return start_idx, end_idx


    def tokenize_fn_train(self, batch):

        questions = [q.strip() for q in batch['question']]
        inputs = self.tokenizer(text = questions,
                        text_pair = batch['context'],
                        max_length=self.max_length, # the maximum length of context
                        truncation='only_second', # 不能truncate question
                        stride=self.stride, # move forward 50 (overlap 50 chars)
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding='max_length')

        offset_mapping = inputs.pop("offset_mapping")
        orig_sample_idxs = inputs.pop("overflow_to_sample_mapping")
        answers = batch['answers']
        start_idxs, end_idxs = [], []

        for i,  offset in enumerate(offset_mapping):

            ## ans 在原文中的位置
            sample_idx = orig_sample_idxs[i] #用overflow_to_sample_mapping 對照回原資料的id
            answer = answers[sample_idx]

            ans_start_char = answer['answer_start'][0]
            ans_end_char = ans_start_char + len(answer['text'][0])


            #找到切割後的句子arr中context 位置
            sequence_ids = inputs.sequence_ids(i)
            # print(len(offset), len(sequence_ids))

            ctx_start = sequence_ids.index(1) # first occurrence
            ctx_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1 # last occurrence

            # 找到切割後contex內ans的位置，若不包含ans, 回傳 0,0
            new_s, new_e = self.find_answer_token_idx(ctx_start, ctx_end,
                                                ans_start_char, ans_end_char,
                                                offset)
            start_idxs.append(new_s)
            end_idxs.append(new_e)

        inputs["start_positions"] = start_idxs
        inputs["end_positions"] = end_idxs
        return inputs


    def tokenize_fn_val(self, batch):

        questions = [q.strip() for q in batch['question']]
        inputs = self.tokenizer(text = questions,
                                text_pair = batch['context'],
                                max_length=self.max_length, # the maximum length of context
                                truncation='only_second', # 不能truncate question
                                stride=self.stride, # move forward 50 (overlap 50 chars)
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                padding='max_length')

        orig_sample_idxs = inputs.pop("overflow_to_sample_mapping")
        sample_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = orig_sample_idxs[i]
            sample_ids.append(batch['id'][sample_idx]) #存原本資料的 str id

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)] # mask掉question的部分為None

        inputs['sample_id'] = sample_ids
        return inputs