SRC_NAME = term
# OUT_NAME = main

all: build clean

build:
	latexmk -pdf -cd -time -verbose src/$(SRC_NAME).tex
clean: build
# 	latexmk -cd -c $(PAPER_PATH)/src/main.tex 
	mv "src/$(SRC_NAME).pdf" "$(SRC_NAME).pdf"