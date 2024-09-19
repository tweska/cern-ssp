.PHONY: all clean

SUBDIRS = $(dir $(wildcard */Makefile))

all:
	@ $(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) all;)

clean:
	@ $(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) clean;)
