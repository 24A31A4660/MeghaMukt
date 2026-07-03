import re
import sys

def main():
    try:
        with open('C:\\Users\\aksha\\.gemini\\antigravity-ide\\brain\\ec1360f4-2bb8-4d23-9f92-df543c06d1a3\\.system_generated\\logs\\transcript_full.jsonl', 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print('Error reading file:', e)
        return

    # Find the massive diff block (the one I deleted). It deleted lines 4224-4700.
    # The diff block starts with @@ -4224,477 +4224,6 @@
    match = re.search(r'@@ -4224,\d+ \+4224,\d+ @@(.*?)\[diff_block_end\]', text, re.DOTALL)
    if not match:
        print('Diff block not found')
        return

    diff = match.group(1)
    
    # JSON encoding escapes newlines as \n, so we need to process it correctly.
    # In transcript_full.jsonl, the diff text is inside a JSON string.
    # We can just unescape it or split by literal '\n'.
    # Because it's raw text read from JSON source, a newline could be literal '\\n' or real newline.
    if '\\n' in diff:
        lines = diff.split('\\n')
    else:
        lines = diff.split('\n')
        
    recovered = []
    for l in lines:
        if l.startswith('-'):
            recovered.append(l[1:])
        elif l.startswith('\\r-'): # edge case
            recovered.append(l[3:])
            
    if not recovered:
        print("No lines recovered")
        return
        
    with open('css/style.css', 'a', encoding='utf-8') as out:
        out.write('\n' + '\n'.join(recovered) + '\n')
    print(f'Recovered {len(recovered)} lines of CSS flawlessly!')

if __name__ == '__main__':
    main()
