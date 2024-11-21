#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

#define MAX_LEN 4096
static struct proc_dir_entry *proc_entry;
static char *info;
static int len;

// Prototypes
ssize_t read_proc(struct file *file, char __user *user_buf, size_t count, loff_t *off);
ssize_t write_proc(struct file *file, const char __user *user_buf, size_t count, loff_t *off);

// File operations using proc_ops
static const struct proc_ops proc_fops = {
    .proc_read = read_proc,
    .proc_write = write_proc,
};

// Read function
ssize_t read_proc(struct file *file, char __user *user_buf, size_t count, loff_t *off) {
    if (*off > 0 || len == 0) return 0;
    if (copy_to_user(user_buf, info, len)) return -EFAULT;
    *off = len;
    return len;
}

// Write function
ssize_t write_proc(struct file *file, const char __user *user_buf, size_t count, loff_t *off) {
    if (count > MAX_LEN) return -EINVAL;
    memset(info, 0, MAX_LEN);
    len = count;
    if (copy_from_user(info, user_buf, count)) return -EFAULT;
    return len;
}

// Module initialization
int init_module(void) {
    info = kmalloc(MAX_LEN, GFP_KERNEL);
    if (!info) {
        printk(KERN_ERR "Failed to allocate memory for info\n");
        return -ENOMEM;
    }
    memset(info, 0, MAX_LEN);
    proc_entry = proc_create("myproc", 0666, NULL, &proc_fops);
    if (!proc_entry) {
        kfree(info);
        printk(KERN_ERR "Failed to create /proc/myproc\n");
        return -ENOMEM;
    }
    printk(KERN_INFO "/proc/myproc created\n");
    return 0;
}

// Module cleanup
void cleanup_module(void) {
    remove_proc_entry("myproc", NULL);
    kfree(info);
    printk(KERN_INFO "/proc/myproc removed\n");
}

MODULE_LICENSE("GPL");
