import json
import re
import sys

def main():
    try:
        with open('css/s02_dump.json', 'r', encoding='utf-8') as f:
            args = json.load(f)
    except Exception as e:
        print('Error reading JSON:', e)
        return

    cmd = args.get('CommandLine', '')
    
    # We need to extract the CSS string from the python command.
    # The command used by the previous agent was a python -c script.
    
    css_text = ''
    markers = [
        ('css_content = """', '"""'),
        ('css_text = """', '"""'),
        ("css_content = '''", "'''"),
        ("css_text = '''", "'''"),
        ('css_content = \"\"\"', '\"\"\"'),
        ('css_text = \"\"\"', '\"\"\"')
    ]
    
    for start_marker, end_marker in markers:
        if start_marker in cmd:
            start_idx = cmd.find(start_marker) + len(start_marker)
            end_idx = cmd.find(end_marker, start_idx)
            if end_idx != -1:
                css_text = cmd[start_idx:end_idx]
                break
                
    if not css_text:
        print('Could not find CSS block in command')
        with open('css/cmd_dump.txt', 'w', encoding='utf-8') as out:
            out.write(cmd)
        return

    with open('css/style.css', 'a', encoding='utf-8') as out:
        out.write('\n' + css_text + '\n')
    print(f'Successfully recovered {len(css_text)} chars of CSS and appended to style.css!')

if __name__ == '__main__':
    main()
