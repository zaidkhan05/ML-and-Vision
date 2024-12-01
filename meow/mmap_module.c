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

// Declare proc directory and file pointers
static struct proc_dir_entry *tempdir, *tempinfo; // `/proc/mydir` and `/proc/mydir/myinfo`

// Kernel buffer for data storage
static unsigned char *buffer; // Dynamically allocated buffer
static unsigned char array[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Example data to initialize the buffer

// Allocate kernel memory
static void allocate_memory(void) {
    // Allocate a page-sized kernel buffer
    buffer = (unsigned char *)kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (buffer) {
        // Mark the allocated memory as reserved to prevent swapping
        SetPageReserved(virt_to_page(buffer));
        // Copy predefined data into the allocated buffer
        memcpy(buffer, array, sizeof(array));
    }
}

// Free kernel memory
static void clear_memory(void) {
    if (buffer) {
        // Clear the reserved status for the memory page
        ClearPageReserved(virt_to_page(buffer));
        // Free the allocated memory
        kfree(buffer);
    }
}

// mmap handler: Maps kernel memory to user space
static int my_map(struct file *filp, struct vm_area_struct *vma) {
    // Get the physical frame number (PFN) for the kernel buffer
    unsigned long pfn = virt_to_phys((void *)buffer) >> PAGE_SHIFT;
    
    // Map the kernel memory to the user space address range
    if (remap_pfn_range(vma, vma->vm_start, pfn, vma->vm_end - vma->vm_start, vma->vm_page_prot)) {
        printk(KERN_ERR "Failed to map kernel memory to user space\n");
        return -EAGAIN; // Return error if mapping fails
    }
    
    printk(KERN_INFO "Kernel memory mapped to user space\n");
    return 0; // Indicate success
}

// Define file operations structure for the proc file
static const struct proc_ops myproc_fops = {
    .proc_mmap = my_map, // Only mmap operation is implemented
};

// Module initialization function
static int __init init_myproc_module(void) {
    // Create a directory under /proc
    tempdir = proc_mkdir("mydir", NULL);
    if (!tempdir) {
        printk(KERN_ERR "Failed to create /proc/mydir\n");
        return -ENOMEM; // Return error for memory allocation failure
    }

    // Create a file under /proc/mydir
    tempinfo = proc_create("myinfo", 0666, tempdir, &myproc_fops);
    if (!tempinfo) {
        // Remove the directory if file creation fails
        remove_proc_entry("mydir", NULL);
        printk(KERN_ERR "Failed to create /proc/mydir/myinfo\n");
        return -ENOMEM; // Return error for memory allocation failure
    }

    // Allocate kernel memory for the buffer
    allocate_memory();
    printk(KERN_INFO "Kernel module initialized successfully\n");
    return 0; // Indicate success
}

// Module cleanup function
static void __exit exit_myproc_module(void) {
    // Free allocated kernel memory
    clear_memory();
    
    // Remove the proc file and directory
    remove_proc_entry("myinfo", tempdir);
    remove_proc_entry("mydir", NULL);
    
    printk(KERN_INFO "Kernel module removed successfully\n");
}

module_init(init_myproc_module); // Register the initialization function
module_exit(exit_myproc_module); // Register the cleanup function

MODULE_LICENSE("GPL"); // Specify the license for the module
