TYPST_ROOT ?= "/home/xiyuanyang/Note/TypstNote/"
SOURCE_FILE ?= /home/xiyuanyang/Note/TypstNote/LectureNote/AlgorithmsNote/algorithm.typ
OUTPUT_FILE ?= ./result/algorithm.pdf

TYPST_CMD := typst compile --root $(TYPST_ROOT)

.PHONY: all
all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SOURCE_FILE)
	$(TYPST_CMD) $< $@