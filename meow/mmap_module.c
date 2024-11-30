#include <linux/module.h>
#include <linux/list.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/kthread.h>
#include <linux/proc_fs.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <asm/io.h>


static struct proc_dir_entry *tempdir, *tempinfo;
static unsigned char *buffer; // Kernel buffer
static unsigned char array[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// Allocate kernel memory
static void allocate_memory(void) {
    buffer = (unsigned char *)kmalloc(PAGE_SIZE, GFP_KERNEL); // Allocate a page-sized buffer
    if (buffer) {
        SetPageReserved(virt_to_page(buffer)); // Mark the memory as reserved
        memcpy(buffer, array, sizeof(array)); // Copy initial data into buffer
    }
}

// Free kernel memory
static void clear_memory(void) {
    if (buffer) {
        ClearPageReserved(virt_to_page(buffer)); // Unmark the memory as reserved
        kfree(buffer); // Free the memory
    }
}

// mmap handler
static int my_map(struct file *filp, struct vm_area_struct *vma) {
    unsigned long pfn = virt_to_phys((void *)buffer) >> PAGE_SHIFT; // Physical address of the buffer
    if (remap_pfn_range(vma, vma->vm_start, pfn, vma->vm_end - vma->vm_start, vma->vm_page_prot)) {
        printk(KERN_ERR "Failed to map kernel memory to user space\n");
        return -EAGAIN;
    }
    printk(KERN_INFO "Kernel memory mapped to user space\n");
    return 0;
}

// Define file operations
static const struct proc_ops myproc_fops = {
    .proc_mmap = my_map, // Only mmap operation is required
};

// Initialize the module
static int __init init_myproc_module(void) {
    tempdir = proc_mkdir("mydir", NULL); // Create /proc/mydir
    if (!tempdir) {
        printk(KERN_ERR "Failed to create /proc/mydir\n");
        return -ENOMEM;
    }

    tempinfo = proc_create("myinfo", 0666, tempdir, &myproc_fops); // Create /proc/mydir/myinfo
    if (!tempinfo) {
        remove_proc_entry("mydir", NULL); // Clean up on failure
        printk(KERN_ERR "Failed to create /proc/mydir/myinfo\n");
        return -ENOMEM;
    }

    allocate_memory(); // Allocate memory for the buffer
    printk(KERN_INFO "Kernel module initialized successfully\n");
    return 0;
}

// Cleanup the module
static void __exit exit_myproc_module(void) {
    clear_memory(); // Free allocated memory
    remove_proc_entry("myinfo", tempdir); // Remove /proc/mydir/myinfo
    remove_proc_entry("mydir", NULL); // Remove /proc/mydir
    printk(KERN_INFO "Kernel module removed successfully\n");
}

module_init(init_myproc_module);
module_exit(exit_myproc_module);

MODULE_LICENSE("GPL");
