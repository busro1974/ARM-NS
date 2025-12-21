"""Simple script to strip trailing whitespace and remove lines that are only whitespace in src/"""
import pathlib

root = pathlib.Path(__file__).resolve().parents[1] / 'src'
for path in root.rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    new_lines = []
    changed = False
    for line in text.splitlines():
        new_line = line.rstrip()
        if new_line != line:
            changed = True
        new_lines.append(new_line)
    # Ensure file ends with a newline
    new_text = '\n'.join(new_lines) + '\n'
    if new_text != text:
        path.write_text(new_text, encoding='utf-8')
        print(f"Cleaned: {path}")
