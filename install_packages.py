import os, sys

txtfile = sys.argv[1]

with open(txtfile,"r+") as fp:
	for line in fp:
		cmdline = str(line.strip())
		os.system(cmdline)
