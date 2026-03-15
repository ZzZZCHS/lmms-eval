import torch
from tqdm import tqdm
import os

# compress_result1 = torch.load("compress_results1.pt")
# compress_result2 = torch.load("compress_results2.pt")

# compress_input1 = torch.load("compress_input1.pt")
# compress_input2 = torch.load("compress_input2.pt")

# compress_before_merge1 = torch.load("compress_before_merge1.pt")
# compress_before_merge2 = torch.load("compress_before_merge2.pt")

# compress_after_merge1 = torch.load("compress_after_merge1.pt")
# compress_after_merge2 = torch.load("compress_after_merge2.pt")

# compress_output1 = torch.load("compress_output1.pt")
# compress_output2 = torch.load("compress_output2.pt")

# hidden1 = compress_output1['all_features']
# hidden2 = compress_output2['all_features']

# true_num = 0
# false_num = 0
# for i in range(hidden1.shape[0]):
#     true_sum = (hidden1[i] == hidden2[i]).sum()
#     if true_sum == hidden1.shape[1]:
#         true_num += 1
#     elif true_sum == 0:
#         false_num += 1
#     else:
#         print(i, true_sum)

def check_same_dict(dict1, dict2, prefix=""):
    for k in dict1.keys():
        if k not in dict2:
            print(prefix + k, "not in dict2")
            continue
        if isinstance(dict1[k], torch.Tensor) and isinstance(dict2[k], torch.Tensor):
            if not torch.equal(dict1[k], dict2[k]):
                print(prefix + k, "not equal")
                if k == "all_features":
                    tot = dict1[k].numel()
                    eq = (dict1[k] == dict2[k]).sum().item()
                    print(f"{tot-eq}/{tot} elements not equal")
                    if torch.isnan(dict1[k]).any() or torch.isnan(dict2[k]).any():
                        print("NaN values found")
                    diff = (dict1[k] - dict2[k]) / (dict1[k].abs() + 1e-7)
                    print(dict1[k].dtype)
                    print("max abs diff:", diff.abs().max().item())
                else:
                    print(dict1[k])
                    print(dict2[k])
        else:
            if dict1[k] != dict2[k]:
                print(prefix + k, "not equal")
                print(dict1[k])
                print(dict2[k])

# check_same_dict(compress_result1, compress_result2, prefix="compress_result ")
# check_same_dict(compress_input1, compress_input2, prefix="compress_input ")
# check_same_dict(compress_before_merge1, compress_before_merge2, prefix="compress_before_merge ")
# check_same_dict(compress_after_merge1, compress_after_merge2, prefix="compress_after_merge ")
# check_same_dict(compress_output1, compress_output2, prefix="compress_output ")
            

# print(true_num, false_num)
# breakpoint()

# all_results1 = torch.load("all_results1.pt")
# all_results2 = torch.load("all_results2.pt")

# for doc_id in all_results1.keys():
#     if doc_id not in all_results2:
#         print(doc_id, "not in all_results2")
#         continue
#     check_same_dict(all_results1[doc_id], all_results2[doc_id], prefix=f"doc_id {doc_id} ")

# llm_inputs1 = torch.load("llm_input1.pt")
# llm_inputs2 = torch.load("llm_input2.pt")

# for doc_id in llm_inputs1.keys():
#     if doc_id not in llm_inputs2:
#         print(doc_id, "not in llm_inputs2")
#         continue
    
#     hidden_states1 = llm_inputs1[doc_id]["hidden_states"]
#     hidden_states2 = llm_inputs2[doc_id]["hidden_states"]
    
#     position_ids1 = llm_inputs1[doc_id]["position_ids"]
#     position_ids2 = llm_inputs2[doc_id]["position_ids"]
    
#     if not torch.equal(position_ids1, position_ids2):
#         print(doc_id, "position_ids not equal")
#         print(position_ids1)
#         print(position_ids2)
    
#     video_start_idx1 = llm_inputs1[doc_id]["video_start_idx"]
#     video_start_idx2 = llm_inputs2[doc_id]["video_start_idx"]
    
#     video_token_len1 = llm_inputs1[doc_id]["video_token_len"]
#     video_token_len2 = llm_inputs2[doc_id]["video_token_len"]
#     before_video1 = hidden_states1[:, :video_start_idx1, :]
#     before_video2 = hidden_states2[:, :video_start_idx2, :]
    
#     video1 = hidden_states1[:, video_start_idx1:video_start_idx1+video_token_len1, :]
#     video2 = hidden_states2[:, video_start_idx2:video_start_idx2+video_token_len2, :]
    
#     after_video1 = hidden_states1[:, video_start_idx1+video_token_len1:, :]
#     after_video2 = hidden_states2[:, video_start_idx2+video_token_len2:, :]
    
    # if not torch.equal(before_video1, before_video2):
    #     print(doc_id, "before video not equal")
    # if not torch.equal(video1, video2):
    #     print(doc_id, "video not equal")
    #     eq_num = (video1 == video2).sum().item()
    #     tot_num = video1.numel()
    #     print(f"{tot_num-eq_num}/{tot_num} elements not equal")
    #     diff = (video1 - video2) / (video1.abs() + 1e-7)
    #     print("max abs diff:", diff.abs().max().item())
    # if not torch.equal(after_video1, after_video2):
    #     print(doc_id, "after video not equal")
    
# llm_inputs1_dir = "llm_input_newnew1"
# llm_inputs2_dir = "llm_input_newnew2"  
# diffs = []
# for doc_id in tqdm(range(2174)):
#     if not os.path.exists(f"{llm_inputs1_dir}/{doc_id}.pt"):
#         continue
#     llm_inputs1 = torch.load(f"{llm_inputs1_dir}/{doc_id}.pt")
#     llm_inputs2 = torch.load(f"{llm_inputs2_dir}/{doc_id}.pt")
#     position_ids1 = llm_inputs1["position_ids"]
#     position_ids2 = llm_inputs2["position_ids"]
#     cache_position1 = llm_inputs1["cache_position"]
#     cache_position2 = llm_inputs2["cache_position"]
#     position_embeddings1 = llm_inputs1["position_embeddings"]
#     position_embeddings2 = llm_inputs2["position_embeddings"]
#     hidden_states1 = llm_inputs1["hidden_states"]
#     hidden_states2 = llm_inputs2["hidden_states"]
#     video_start_idx1 = llm_inputs1["video_start_idx"]
#     video_start_idx2 = llm_inputs2["video_start_idx"]
#     video_token_len1 = llm_inputs1["video_token_len"]
#     video_token_len2 = llm_inputs2["video_token_len"]
#     seq_length1 = llm_inputs1["seq_length"]
#     seq_length2 = llm_inputs2["seq_length"]
#     if not torch.equal(position_ids1, position_ids2):
#         print(doc_id, "position_ids not equal")
#     if not torch.equal(cache_position1, cache_position2):
#         print(doc_id, "cache_position not equal")
#     if not torch.equal(position_embeddings1[0], position_embeddings2[0]):
#         print(doc_id, "position_embeddings[0] not equal")
#         eq_num = (position_embeddings1[0] == position_embeddings2[0]).sum().item()
#         tot_num = position_embeddings1[0].numel()
#         print(f"{tot_num-eq_num}/{tot_num} elements not equal")
#         diff = (position_embeddings1[0] - position_embeddings2[0]) / (position_embeddings1[0].abs() + 1e-7)
#         print("max abs diff:", diff.abs().max().item())
#     if not torch.equal(position_embeddings1[1], position_embeddings2[1]):
#         print(doc_id, "position_embeddings[1] not equal")
#         eq_num = (position_embeddings1[1] == position_embeddings2[1]).sum().item()
#         tot_num = position_embeddings1[1].numel()
#         print(f"{tot_num-eq_num}/{tot_num} elements not equal")
#         diff = (position_embeddings1[1] - position_embeddings2[1]) / (position_embeddings1[1].abs() + 1e-7)
#         print("max abs diff:", diff.abs().max().item())
#     if not torch.equal(hidden_states1, hidden_states2):
#         print(doc_id, "hidden_states not equal")
#         eq_num = (hidden_states1 == hidden_states2).sum().item()
#         tot_num = hidden_states1.numel()
#         print(f"{tot_num-eq_num}/{tot_num} elements not equal")
#         diff = (hidden_states1 - hidden_states2) / (hidden_states1.abs() + 1e-7)
#         print("max abs diff:", diff.abs().max().item())
#         diffs.append(diff.abs().max().item())
#     if video_start_idx1 != video_start_idx2:
#         print(doc_id, "video_start_idx not equal")
#     if video_token_len1 != video_token_len2:
#         print(doc_id, "video_token_len not equal")
#     # if seq_length1 != seq_length2:
#     #     print(doc_id, "seq_length not equal")

# print("average max abs diff in hidden_states:", sum(diffs)/len(diffs))
# print("max abs diff in hidden_states:", max(diffs))


import json
results1 = []
with open('/lustre/hdd/LAS/yangli1-lab/haifengh/VidCom2/lmms-eval/logs/interval_consecutive_difference_change_l2_a800_sim_thres-1_whiten-false_attn_gamma0.0_merge-importance-2_pivotal_seed2718281828459045_Tsigma0_diff-110-70-0.4_25/lmms-lab__llava-onevision-qwen2-7b-ov/20260315_111904_samples_mlvu_dev.jsonl') as f:
    for line in f:
        results1.append(json.loads(line))

results2 = []
with open('/lustre/hdd/LAS/yangli1-lab/haifengh/VidCom2/lmms-eval/logs/interval_consecutive_difference_change_l2_a800_sim_thres-1_whiten-false_attn_gamma0.0_merge-importance-2_pivotal_seed2718281828459045_Tsigma0_diff-110-70-0.4_25/lmms-lab__llava-onevision-qwen2-7b-ov/20260315_131047_samples_mlvu_dev.jsonl') as f:
    for line in f:
        results2.append(json.loads(line))
        
tot = 0
acc1 = 0
acc2 = 0
for r1, r2 in zip(results1, results2):
    assert r1["doc_id"] == r2["doc_id"], f"doc_id not equal: {r1['doc_id']} vs {r2['doc_id']}"
    if r1['mlvu_percetion_score']['pred_answer'][0] != r2['mlvu_percetion_score']['pred_answer'][0]:
        # breakpoint()
        tot += 1
        print(f"doc_id {r1['doc_id']} pred_answer not equal: {r1['mlvu_percetion_score']['pred_answer']} vs {r2['mlvu_percetion_score']['pred_answer']}")
        print(f"gt_answer: {r1['mlvu_percetion_score']['answer']}")
        pred_logits1 = [float(x) for x in r1["filtered_resps"][0][-2][0]]
        pred_logits2 = [float(x) for x in r2["filtered_resps"][0][-2][0]]
        print("pred_logits1:", pred_logits1)
        print("pred_logits2:", pred_logits2)
    if r1['mlvu_percetion_score']['pred_answer'] == r1['mlvu_percetion_score']['answer']:
        acc1 += 1
    if r2['mlvu_percetion_score']['pred_answer'] == r2['mlvu_percetion_score']['answer']:
        acc2 += 1

print(tot)
print(f"acc1: {acc1}/{len(results1)}={acc1/len(results1):.4f}")
print(f"acc2: {acc2}/{len(results2)}={acc2/len(results2):.4f}")