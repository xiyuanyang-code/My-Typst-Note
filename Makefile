TYPST_ROOT ?= "/Users/xiyuanyang/Desktop/Dev/TypstNote"
SOURCE_FILE ?= /Users/xiyuanyang/Desktop/Dev/TypstNote/LectureNote/AlgorithmsNote/algorithm.typ
OUTPUT_FILE ?= ./result/algorithm.pdf

TYPST_CMD := typst compile --root $(TYPST_ROOT)

.PHONY: all
all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SOURCE_FILE)
	$(TYPST_CMD) $< $@