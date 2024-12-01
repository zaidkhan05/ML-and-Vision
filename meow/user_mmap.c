#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <fcntl.h> 
#include <sys/mman.h>
#include <sys/ioctl.h>

#define PAGE_SIZE 4096 // Define the page size for memory mapping

int main(int argc, char *argv[]) {
    int fd;                // File descriptor for /proc/mydir/myinfo
    unsigned char *p_map;  // Pointer for the mapped memory

    // Open the /proc file for read-write access
    fd = open("/proc/mydir/myinfo", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /proc/mydir/myinfo"); // Print error if open fails
        exit(1); // Exit with error code
    } else {
        printf("open successfully by Zaid\n"); // Log success
    }

    // Map the kernel memory associated with the proc file to user space
    p_map = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p_map == MAP_FAILED) { // Check if mmap failed
        perror("Failed to mmap"); // Print error message
        close(fd); // Close the file descriptor before exiting
        exit(1); // Exit with error code
    }

    // Read and print the first 12 bytes of the mapped memory
    for (int i = 0; i < 12; i++) { 
        printf("%d\n", p_map[i]); // Print each byte as an integer
    }

    // Print a concluding message
    printf("Printed by Zaid\n");

    // Unmap the memory and check for errors
    if (munmap(p_map, PAGE_SIZE) == -1) {
        perror("Failed to munmap"); // Print error if munmap fails
    }

    // Close the file descriptor
    close(fd);

    return 0; // Return success code
}
