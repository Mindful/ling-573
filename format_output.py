import sys

report = False
for line in sys.stdin:
	if report:
		print(line)
	if '---' in line:
		report = True
	if '...' in line:
		report = False

