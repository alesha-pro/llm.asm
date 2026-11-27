obj-m += surgeon_crash.o
KDIR ?= /lib/modules/5.15.0-185-generic/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
