import json
import sys

def process_file(input_file, output_file, max_words=10000):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if 'meta' in data:
                del data['meta']
            text = data['text'].split(' ')
            # split overly long samples, otherwise tokenization might get stuck
            while len(text) > max_words:
                data['text'] = ' '.join(text[:max_words])
                outfile.write(json.dumps(data) + '\n')
                text = text[max_words:]
            data['text'] = ' '.join(text)
            outfile.write(json.dumps(data) + '\n')

i = int(sys.argv[1])
input_file = f"{i:02d}.jsonl"
output_file = f"{i:02d}.text.jsonl"
process_file(input_file, output_file)
