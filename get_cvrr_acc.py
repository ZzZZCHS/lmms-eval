data = """
|                             Tasks                             |Version|Filter|n-shot|     Metric      |   | Value |   |Stderr|
|---------------------------------------------------------------|-------|------|-----:|-----------------|---|------:|---|------|
|cvrr                                                           |    N/A|      |      |                 |   |       |   |      |
| - cvrr_continuity_and_object_instance_count                   |      0|none  |     0|gpt_eval_accuracy|↑  |36.7232|±  |   N/A|
| - cvrr_continuity_and_object_instance_count                   |      0|none  |     0|gpt_eval_score   |↑  | 2.5367|±  |   N/A|
| - cvrr_fine_grained_action_understanding                      |      0|none  |     0|gpt_eval_accuracy|↑  |48.2609|±  |   N/A|
| - cvrr_fine_grained_action_understanding                      |      0|none  |     0|gpt_eval_score   |↑  | 2.8043|±  |   N/A|
| - cvrr_interpretation_of_social_context                       |      0|none  |     0|gpt_eval_accuracy|↑  |59.2857|±  |   N/A|
| - cvrr_interpretation_of_social_context                       |      0|none  |     0|gpt_eval_score   |↑  | 3.3107|±  |   N/A|
| - cvrr_interpretation_of_visual_context                       |      0|none  |     0|gpt_eval_accuracy|↑  |61.5385|±  |   N/A|
| - cvrr_interpretation_of_visual_context                       |      0|none  |     0|gpt_eval_score   |↑  | 3.3810|±  |   N/A|
| - cvrr_multiple_actions_in_a_single_video                     |      0|none  |     0|gpt_eval_accuracy|↑  |41.1950|±  |   N/A|
| - cvrr_multiple_actions_in_a_single_video                     |      0|none  |     0|gpt_eval_score   |↑  | 2.6195|±  |   N/A|
| - cvrr_non_existent_actions_with_existent_scene_depictions    |      0|none  |     0|gpt_eval_accuracy|↑  |55.0725|±  |   N/A|
| - cvrr_non_existent_actions_with_existent_scene_depictions    |      0|none  |     0|gpt_eval_score   |↑  | 3.1957|±  |   N/A|
| - cvrr_non_existent_actions_with_non_existent_scene_depictions|      0|none  |     0|gpt_eval_accuracy|↑  |43.7500|±  |   N/A|
| - cvrr_non_existent_actions_with_non_existent_scene_depictions|      0|none  |     0|gpt_eval_score   |↑  | 2.5903|±  |   N/A|
| - cvrr_partial_actions                                        |      0|none  |     0|gpt_eval_accuracy|↑  |66.5049|±  |   N/A|
| - cvrr_partial_actions                                        |      0|none  |     0|gpt_eval_score   |↑  | 3.6262|±  |   N/A|
| - cvrr_time_order_understanding                               |      0|none  |     0|gpt_eval_accuracy|↑  |37.5000|±  |   N/A|
| - cvrr_time_order_understanding                               |      0|none  |     0|gpt_eval_score   |↑  | 2.5461|±  |   N/A|
| - cvrr_understanding_emotional_context                        |      0|none  |     0|gpt_eval_accuracy|↑  |38.3562|±  |   N/A|
| - cvrr_understanding_emotional_context                        |      0|none  |     0|gpt_eval_score   |↑  | 2.4384|±  |   N/A|
| - cvrr_unusual_and_physically_anomalous_activities            |      0|none  |     0|gpt_eval_accuracy|↑  |36.3158|±  |   N/A|
| - cvrr_unusual_and_physically_anomalous_activities            |      0|none  |     0|gpt_eval_score   |↑  | 2.2895|±  |   N/A|

"""

accuracies = []

for line in data.splitlines():
    if "gpt_eval_accuracy" not in line:
        continue

    cols = [c.strip() for c in line.split("|")]
    # After splitting, the columns are roughly:
    # 0 empty, 1 task, 2 version, 3 filter, 4 n-shot, 5 metric, 6 arrow, 7 value, 8 ±, 9 stderr, 10 empty
    value = float(cols[7])
    accuracies.append(value)

average = sum(accuracies) / len(accuracies)

print("Accuracies:", accuracies)
print("Average:", average)