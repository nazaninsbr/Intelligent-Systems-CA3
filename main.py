from sys import argv
from Q1 import main as run_Q1
from Q2 import main as run_Q2
from Q3 import main as run_Q3
from Q4 import main as run_Q4

def main():
	if '--All' in argv:
		run_Q1()
		run_Q2()
		run_Q3()
		run_Q4()
	if '--Q1' in argv:
		run_Q1()
	if '--Q2' in argv:
		run_Q2()
	if '--Q3' in argv:
		run_Q3()
	if '--Q4' in argv:
		run_Q4()

if __name__ == '__main__':
	main()