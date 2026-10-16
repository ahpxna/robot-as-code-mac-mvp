.PHONY: install test plan smoke headless sim interactive eval map
TASK ?= Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao.
install:
	python3 -m pip install -e .
test:
	python3 -m unittest discover -s tests -v
plan:
	python3 -m robotcode.cli --dry-run --task "$(TASK)"
smoke:
	python3 -m robotcode.cli --smoke --headless
headless:
	python3 -m robotcode.cli --headless --task "$(TASK)"
sim:
	mjpython -m robotcode.cli --task "$(TASK)"
interactive:
	mjpython -m robotcode.cli --interactive
eval:
	python3 -m robotcode.evaluate --episodes 20
map:
	python3 -c "from robotcode.mapping import get_prior_map; get_prior_map('apartment', rebuild=True); print('build/maps/ updated')"
