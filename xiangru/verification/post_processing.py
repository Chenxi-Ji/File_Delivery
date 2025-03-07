import csv
import re

def parse_text_file(input_file, output_csv):
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        id_counter = 0
        for i in range(len(lines)):
            if lines[i].strip().startswith('%%%%%%%%%%%%%%%%'):
                # Look for the next line that starts with 'Result:'
                for j in range(i + 1, len(lines)):
                    line = lines[j].strip()
                    match = re.search(r'Result:\s*(\S+)\s+in', line)
                    if match:
                        result_str = match.group(1)
                        if result_str == "safe-incomplete":
                            result_id = 0
                        elif result_str == "safe":
                            result_id = 1
                        elif result_str == "unknown":
                            result_id = 2
                        else:
                            result_id = 3
                        results.append([id_counter, result_id])
                        id_counter += 1
                        break
    
    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

# Example usage
parse_text_file('logs/tinydozer_6-0_6-1.log', 'logs/tinydozer_6-0_6-1_model_layer12_weights_advtrain_50_eps0.03_alpha0.007_iter10_exclude_yaw_0.288_0.388_l1reg.csv')