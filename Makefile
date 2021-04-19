CC := nvcc
CFLAGS := -ccbin=mpic++ -Werror cross-execution-space-call -lm -lmpi -D_MWAITXINTRIN_H_INCLUDED -Xcompiler -fopenmp

CPPSTD ?= c++11
ifneq ($(CPPSTD),)
CFLAGS += --std=$(CPPSTD)
endif

ifeq ($(DEBUG),true)
CFLAGS += -DDEBUG
endif

TARGET := rays

SOURCES_DIR := ./

all: $(TARGET)

$(TARGET): $(SOURCES_DIR)/$(TARGET).cu
	$(CC) $(CFLAGS) -o "$@" $<

SIGNER_EMAIL := andranik.chakiryan@gmail.com

%.asc: %
	$(RM) $@
	gpg -u $(SIGNER_EMAIL) -ab "$*"

.PHONY: clean
clean:
	$(RM) $(TARGET)
