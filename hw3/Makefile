all: core doc

core:
	python3 p1.py
	python3 p2.py
	python3 p3.py
	python3 p4.py
doc:
	pandoc -f markdown-implicit_figures report.md -o report.pdf -V geometry:margin=1in


clean:
	-rm *.dat
	-rm *.png
	-rm *.pdf