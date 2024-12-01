#include <linux/module.h>
#include <linux/kernel.h>
MODULE_LICENSE("GPL");

int init_new_module(void);
void exit_new_module(void);

int init_new_module(void)
{
	printk(KERN_INFO "Hello, world!\n");
	return 0;

}

void exit_new_module(void) {
	printk(KERN_INFO "Goodbye, world!\n");
}

module_init(init_new_module);
module_exit(exit_new_module);