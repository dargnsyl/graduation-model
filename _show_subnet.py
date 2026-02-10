from pathlib import Path
import re
p=Path('model/DHGFormer.py')
t=p.read_text(encoding='latin1')
lines=t.splitlines()
for i in range(260, 285):
    print(i+1, lines[i])
