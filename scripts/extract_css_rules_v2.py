import re

with open('css/style.css', 'r', encoding='utf-8') as f:
    content = f.read()

classes = ['\.nav', '\.nav__logo', '\.nav-menu-btn', '\.hero-eyebrow', '\.hero-ui-overlay', '\.hero-content-left', '\.cinematic-title', '#earth-canvas']

for cls in classes:
    matches = re.finditer(f"{cls}[^{{]*{{[^}}]*}}", content)
    for match in matches:
        print(match.group(0))
