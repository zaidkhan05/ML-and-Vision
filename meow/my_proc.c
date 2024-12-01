#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

// Define the maximum length for the buffer
#define MAX_LEN 4096

// Declare a proc file entry pointer and variables for storing data
static struct proc_dir_entry *proc_entry; // Pointer to the /proc entry
static char *info; // Buffer to store written data
static int len; // Length of the data in the buffer

// Function prototypes for the proc file operations
ssize_t read_proc(struct file *file, char __user *user_buf, size_t count, loff_t *off);
ssize_t write_proc(struct file *file, const char __user *user_buf, size_t count, loff_t *off);

// File operations structure for proc entry, using `proc_ops`
static const struct proc_ops proc_fops = {
    .proc_read = read_proc,  // Read operation for the proc file
    .proc_write = write_proc // Write operation for the proc file
};

// Implementation of the read function for the proc file
ssize_t read_proc(struct file *file, char __user *user_buf, size_t count, loff_t *off) {
    // If offset is non-zero (already read) or there's no data, return 0
    if (*off > 0 || len == 0) 
        return 0;
    
    // Copy the data from kernel buffer to user space
    if (copy_to_user(user_buf, info, len)) 
        return -EFAULT; // Return error if copy fails
    
    *off = len; // Update the file offset to indicate all data is read
    return len; // Return the length of the data read
}

// Implementation of the write function for the proc file
ssize_t write_proc(struct file *file, const char __user *user_buf, size_t count, loff_t *off) {
    // Ensure the data does not exceed the maximum length
    if (count > MAX_LEN) 
        return -EINVAL; // Return error for invalid argument
    
    // Clear the buffer before writing new data
    memset(info, 0, MAX_LEN);
    
    len = count; // Update the length of data
    // Copy the data from user space to kernel buffer
    if (copy_from_user(info, user_buf, count)) 
        return -EFAULT; // Return error if copy fails
    
    return len; // Return the number of bytes written
}

// Module initialization function
int init_module(void) {
    // Allocate memory for the buffer in kernel space
    info = kmalloc(MAX_LEN, GFP_KERNEL);
    if (!info) { // Check if allocation was successful
        printk(KERN_ERR "Failed to allocate memory for info\n");
        return -ENOMEM; // Return error for insufficient memory
    }
    
    // Initialize the buffer with zeros
    memset(info, 0, MAX_LEN);
    
    // Create the proc entry named "myproc" with read-write permissions
    proc_entry = proc_create("myproc", 0666, NULL, &proc_fops);
    if (!proc_entry) { // Check if creation was successful
        kfree(info); // Free the allocated memory
        printk(KERN_ERR "Failed to create /proc/myproc\n");
        return -ENOMEM; // Return error for failure
    }
    
    printk(KERN_INFO "/proc/myproc created\n"); // Log successful creation
    return 0; // Indicate success
}

// Module cleanup function
void cleanup_module(void) {
    // Remove the proc entry from the /proc file system
    remove_proc_entry("myproc", NULL);
    
    // Free the allocated buffer memory
    kfree(info);
    
    printk(KERN_INFO "/proc/myproc removed\n"); // Log successful removal
}

// Specify the module's license type
MODULE_LICENSE("GPL");
