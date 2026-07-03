import json
with open('C:\\Users\\aksha\\.gemini\\antigravity-ide\\brain\\c0a66edb-e149-475f-a0a2-17a252617226\\.system_generated\\logs\\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'tool_calls' in data:
            for call in data['tool_calls']:
                args = call.get('args', {})
                args_str = json.dumps(args)
                if 'style.css' in args_str:
                    print(f"Tool: {call.get('name')}")
                    if call.get('name') == 'default_api:run_command':
                        print('CMD:', args.get('CommandLine')[:100])
                        if '.s02-parent-grid' in args.get('CommandLine', ''):
                            print('FOUND THE RUN COMMAND SCRIPT WITH S02-PARENT-GRID!')
                            with open('css/style_dump.json', 'w', encoding='utf-8') as out:
                                json.dump(args, out)
                    elif call.get('name') == 'default_api:multi_replace_file_content':
                        for c in args.get('ReplacementChunks', []):
                            print('REPLACE length:', len(c.get('ReplacementContent', '')))
                            if '.s02-parent-grid' in c.get('ReplacementContent', ''):
                                print('FOUND THE MULTI REPLACE CHUNK WITH S02-PARENT-GRID!')
                                with open('css/style_dump.json', 'w', encoding='utf-8') as out:
                                    json.dump(args, out)
